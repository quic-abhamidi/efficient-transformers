# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Cloud entrypoint for model pruning / NAS optimization workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from QEfficient.model_pruning import nas_pipeline
from QEfficient.model_pruning.logging_utils import get_logger

logger = get_logger("cloud")

DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"


def _run_workflow(argv: list[str]) -> None:
    logger.info("[model-pruning] %s", " ".join(argv))
    nas_pipeline.main(argv)


def run_analyze(args: argparse.Namespace) -> None:
    cmd = [
        "analyze",
        "--model",
        args.model,
        "--datasets",
        *args.analysis_datasets,
        "--num-samples",
        str(args.analysis_samples),
        "--batch-size",
        str(args.analysis_batch_size),
        "--metric",
        args.metric,
        "--dtype",
        args.dtype,
        "--device-map",
        _resolve_device_map(args),
        "--output-dir",
        args.output_dir,
    ]
    if args.verbose:
        cmd.append("--verbose")
    if args.revision:
        cmd.extend(["--revision", args.revision])
    _run_workflow(cmd)


def run_evaluate(args: argparse.Namespace) -> None:
    if args.skip_layers and args.candidate_plans:
        raise ValueError("Use either --candidate-plans or --skip-layers, not both")
    candidate_plans = args.candidate_plans or str(Path(args.output_dir) / "candidate_plans.json")
    cmd = [
        "evaluate",
        "--model",
        args.model,
    ]
    if args.skip_layers:
        cmd.extend(["--skip-layers", *[str(layer) for layer in args.skip_layers]])
    else:
        cmd.extend(["--candidate-plans", candidate_plans])
    cmd.extend([
        "--datasets",
        *args.eval_datasets,
        "--num-samples",
        str(args.eval_samples),
        "--max-candidates",
        str(args.max_candidates),
        "--eval-method",
        args.eval_method,
        "--generation-len",
        str(getattr(args, "eval_generation_len", getattr(args, "generation_len", 40))),
        "--accuracy-metric",
        args.accuracy_metric,
        "--lm-eval-batch-size",
        str(args.lm_eval_batch_size),
        "--dtype",
        args.dtype,
        "--device-map",
        _resolve_device_map(args),
        "--output-dir",
        args.output_dir,
    ])
    if not args.skip_layers:
        cmd.extend(["--accuracy-threshold", str(args.accuracy_threshold)])
    if args.lm_eval_limit is not None:
        cmd.extend(["--lm-eval-limit", str(args.lm_eval_limit)])
    _extend_videomme_args(cmd, args)
    if args.revision:
        cmd.extend(["--revision", args.revision])
    _run_workflow(cmd)


def _extend_videomme_args(cmd: list[str], args: argparse.Namespace) -> None:
    if getattr(args, "videomme_dataset_path", None):
        cmd.extend(["--videomme-dataset-path", args.videomme_dataset_path])
    if getattr(args, "videomme_video_root", None):
        cmd.extend(["--videomme-video-root", args.videomme_video_root])
    if getattr(args, "videomme_split", None):
        cmd.extend(["--videomme-split", args.videomme_split])
    if getattr(args, "videomme_num_frames", None) is not None:
        cmd.extend(["--videomme-num-frames", str(args.videomme_num_frames)])
    if getattr(args, "videomme_fps", None) is not None:
        cmd.extend(["--videomme-fps", str(args.videomme_fps)])
    if getattr(args, "videomme_use_subtitles", False):
        cmd.append("--videomme-use-subtitles")


def run_qaic(args: argparse.Namespace) -> None:
    qaic_output_dir = args.qaic_output_dir or str(Path(args.output_dir) / "qaic")
    cmd = [
        "qaic",
        "--model",
        args.model,
    ]
    if getattr(args, "skip_layers", None):
        cmd.extend(["--skip-layers", *[str(layer) for layer in args.skip_layers]])
    else:
        plan = args.plan or str(Path(args.output_dir) / "best_plan.json")
        cmd.extend(["--plan", plan])
    cmd.extend([
        "--device-group",
        args.device_group,
        "--batch-size",
        str(args.batch_size),
        "--ctx-len",
        str(args.ctx_len),
        "--prefill-seq-len",
        str(args.prefill_seq_len),
        "--generation-len",
        str(args.generation_len),
        "--num-cores",
        str(args.num_cores),
        "--output-dir",
        qaic_output_dir,
    ])
    if args.compile_dir_base:
        cmd.extend(["--compile-dir-base", args.compile_dir_base])
    if args.prompts:
        cmd.extend(["--prompts", *args.prompts])
    if args.no_mxfp6_matmul:
        cmd.append("--no-mxfp6-matmul")
    if args.no_mxint8_kv_cache:
        cmd.append("--no-mxint8-kv-cache")
    _extend_videomme_args(cmd, args)
    if getattr(args, "videomme_num_samples", None) is not None:
        cmd.extend(["--videomme-num-samples", str(args.videomme_num_samples)])
    _run_workflow(cmd)


def run_all(args: argparse.Namespace) -> None:
    run_analyze(args)
    run_evaluate(args)
    run_qaic(args)


def _resolve_device_map(args: argparse.Namespace) -> str:
    if args.device_map:
        return args.device_map
    if args.device == "cpu":
        return "cpu"
    return "auto"


def _add_hidden_alias(parser: argparse.ArgumentParser, *flags: str, **kwargs) -> None:
    parser.add_argument(*flags, help=argparse.SUPPRESS, **kwargs)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local model path.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face model revision, branch, tag, or commit SHA. Leave unset to use the model repo default revision.")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype used when loading the Hugging Face model, for example float16, bfloat16, or float32.")
    parser.add_argument(
        "--device",
        choices=["cuda", "gpu", "cpu", "auto"],
        default="cuda",
        help="Execution device. cpu maps to HF device-map cpu; cuda/gpu/auto map to auto.",
    )
    parser.add_argument("--device-map", default=None, help="Override Hugging Face device_map. Leave unset unless you need a custom placement such as cpu, auto, or a device map supported by transformers.")
    _add_hidden_alias(parser, "--device_map", dest="device_map")
    parser.add_argument("--output-dir", default="results/qwen25_32b_validation", help="Directory where reports, plans, logs, and benchmark artifacts are written.")
    _add_hidden_alias(parser, "--output_dir", dest="output_dir")
    parser.add_argument("--verbose", action="store_true", help="Print detailed stage progress and write model_pruning_<timestamp>.log in the output directory.")


def _add_analysis_args(
    parser: argparse.ArgumentParser,
    *,
    include_short_aliases: bool = False,
    include_num_data_alias: bool = True,
    sample_default: int | None = 64,
    batch_default: int | None = 1,
) -> None:
    if include_short_aliases:
        parser.add_argument("--datasets", dest="analysis_datasets", nargs="+", default=["gsm8k", "hellaswag"])
        _add_hidden_alias(parser, "--analysis-datasets", "--analysis_datasets", dest="analysis_datasets", nargs="+")
    else:
        parser.add_argument("--analysis-datasets", dest="analysis_datasets", nargs="+", default=["gsm8k", "hellaswag"])
        _add_hidden_alias(parser, "--analysis_datasets", dest="analysis_datasets", nargs="+")

    sample_flags = ["--num-samples"] if include_short_aliases else ["--analysis-samples"]
    parser.add_argument(*sample_flags, dest="analysis_samples", type=int, default=sample_default)
    analysis_sample_aliases = ["--analysis_samples", "--analysis-num-samples", "--analysis_num_samples"]
    if include_short_aliases:
        analysis_sample_aliases.insert(0, "--analysis-samples")
    _add_hidden_alias(parser, *analysis_sample_aliases, dest="analysis_samples", type=int)
    if include_num_data_alias:
        _add_hidden_alias(parser, "--num-data-samples", "--num_data_samples", dest="analysis_samples", type=int)
    if include_short_aliases:
        _add_hidden_alias(parser, "--num_samples", dest="analysis_samples", type=int)

    batch_flags = ["--batch-size"] if include_short_aliases else ["--analysis-batch-size"]
    parser.add_argument(*batch_flags, dest="analysis_batch_size", type=int, default=batch_default)
    analysis_batch_aliases = ["--analysis_batch_size"]
    if include_short_aliases:
        analysis_batch_aliases.insert(0, "--analysis-batch-size")
    _add_hidden_alias(parser, *analysis_batch_aliases, dest="analysis_batch_size", type=int)
    if include_short_aliases:
        _add_hidden_alias(parser, "--batch_size", dest="analysis_batch_size", type=int)
    parser.add_argument("--metric", choices=["cosine", "l2", "both"], default="cosine")


def _add_eval_args(
    parser: argparse.ArgumentParser,
    *,
    include_short_aliases: bool = False,
    include_num_data_alias: bool = True,
    sample_default: int | None = 50,
    generation_dest: str = "generation_len",
) -> None:
    parser.add_argument("--candidate-plans", default=None, help="Path to candidate_plans.json generated by analyze. Omit when using --skip-layers.")
    _add_hidden_alias(parser, "--candidate_plans", dest="candidate_plans")
    parser.add_argument("--skip-layers", nargs="+", type=int, default=None, help="Manual decoder layer indices to skip; evaluates baseline plus this manual skip plan.")
    _add_hidden_alias(parser, "--skip_layers", dest="skip_layers", nargs="+", type=int)

    if include_short_aliases:
        parser.add_argument("--datasets", dest="eval_datasets", nargs="+", default=["gsm8k", "hellaswag"], help="Evaluation datasets/tasks. For lm_eval these are resolved to lm_eval task names.")
        _add_hidden_alias(parser, "--eval-datasets", "--eval_datasets", dest="eval_datasets", nargs="+")
    else:
        parser.add_argument("--eval-datasets", dest="eval_datasets", nargs="+", default=["gsm8k", "hellaswag"], help="Evaluation datasets/tasks. For lm_eval these are resolved to lm_eval task names.")
        _add_hidden_alias(parser, "--eval_datasets", dest="eval_datasets", nargs="+")

    sample_flags = ["--num-samples"] if include_short_aliases else ["--eval-samples"]
    parser.add_argument(*sample_flags, dest="eval_samples", type=int, default=sample_default, help="Number of samples per evaluation dataset. Also used as lm_eval limit when --lm-eval-limit is unset.")
    eval_sample_aliases = ["--eval_samples", "--eval-num-samples", "--eval_num_samples"]
    if include_short_aliases:
        eval_sample_aliases.insert(0, "--eval-samples")
    _add_hidden_alias(parser, *eval_sample_aliases, dest="eval_samples", type=int)
    if include_num_data_alias:
        _add_hidden_alias(parser, "--num-data-samples", "--num_data_samples", dest="eval_samples", type=int)
    if include_short_aliases:
        _add_hidden_alias(parser, "--num_samples", dest="eval_samples", type=int)

    parser.add_argument("--max-candidates", type=int, default=5, help="Number of non-baseline candidate plans to evaluate from candidate_plans.json.")
    _add_hidden_alias(parser, "--max_candidates", dest="max_candidates", type=int)
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=5.0,
        help="Allowed quality loss in percent for candidate-plan selection. Ignored when --skip-layers is used because manual layers are always evaluated and reported.",
    )
    _add_hidden_alias(parser, "--accuracy_threshold", dest="accuracy_threshold", type=float)
    parser.add_argument("--eval-method", choices=["perplexity", "lm_eval", "videomme"], default="lm_eval", help="Evaluation backend. Use lm_eval for text accuracy, videomme for video QA accuracy, or perplexity for PPL checks.")
    _add_hidden_alias(parser, "--eval_method", dest="eval_method", choices=["perplexity", "lm_eval", "videomme"])
    if include_short_aliases:
        parser.add_argument("--generation-len", dest=generation_dest, type=int, default=40, help="Maximum new tokens for lm_eval generate_until tasks and sample completions.")
        _add_hidden_alias(parser, "--generation_len", dest=generation_dest, type=int)
    else:
        parser.add_argument("--eval-generation-len", dest=generation_dest, type=int, default=40, help="Maximum new tokens for lm_eval generate_until tasks and sample completions.")
        _add_hidden_alias(parser, "--eval_generation_len", dest=generation_dest, type=int)
    parser.add_argument("--accuracy-metric", default="auto", help="Metric extracted from lm_eval results. Use auto to prefer acc_norm, acc, exact_match, em, then mc2; set acc for plain accuracy.")
    _add_hidden_alias(parser, "--accuracy_metric", dest="accuracy_metric")
    parser.add_argument("--lm-eval-batch-size", type=int, default=1, help="Batch size passed to lm_eval. Keep 1 for large models or memory-limited GPU runs.")
    _add_hidden_alias(parser, "--lm_eval_batch_size", dest="lm_eval_batch_size", type=int)
    parser.add_argument("--lm-eval-limit", type=int, default=None, help="Optional lm_eval sample limit. Leave unset to use --num-samples.")
    _add_hidden_alias(parser, "--lm_eval_limit", dest="lm_eval_limit", type=int)
    parser.add_argument("--videomme-dataset-path", default=None, help="Local Video-MME JSON/JSONL file or directory. If omitted, loads lmms-lab/Video-MME from Hugging Face.")
    _add_hidden_alias(parser, "--videomme_dataset_path", dest="videomme_dataset_path")
    parser.add_argument("--videomme-video-root", default=None, help="Directory containing Video-MME video files referenced by the dataset rows.")
    _add_hidden_alias(parser, "--videomme_video_root", dest="videomme_video_root")
    parser.add_argument("--videomme-split", default="test", help="Hugging Face split for Video-MME when --videomme-dataset-path is omitted.")
    _add_hidden_alias(parser, "--videomme_split", dest="videomme_split")
    parser.add_argument("--videomme-num-frames", type=int, default=8, help="Number of uniformly sampled frames per video.")
    _add_hidden_alias(parser, "--videomme_num_frames", dest="videomme_num_frames", type=int)
    parser.add_argument("--videomme-fps", type=float, default=None, help="Optional fixed FPS sampling before frame-count downsampling.")
    _add_hidden_alias(parser, "--videomme_fps", dest="videomme_fps", type=float)
    parser.add_argument("--videomme-use-subtitles", action="store_true", help="Inject subtitles/captions into the Video-MME prompt when available.")
    _add_hidden_alias(parser, "--videomme_use_subtitles", dest="videomme_use_subtitles", action="store_true")


def _add_qaic_args(
    parser: argparse.ArgumentParser,
    *,
    include_batch_alias: bool = True,
    batch_default: int | None = 1,
    include_videomme_args: bool = True,
    include_skip_layers: bool = True,
) -> None:
    parser.add_argument("--plan", default=None, help="Defaults to <output-dir>/best_plan.json unless --skip-layers is provided.")
    if include_skip_layers:
        parser.add_argument("--skip-layers", nargs="+", type=int, default=None, help="Manual decoder layer indices to skip for a QAIC-only comparison. Skips HF evaluate and writes manual_best_plan.json in the QAIC output directory.")
        _add_hidden_alias(parser, "--skip_layers", dest="skip_layers", nargs="+", type=int)
    parser.add_argument("--qaic-output-dir", default=None, help="Defaults to <output-dir>/qaic.")
    _add_hidden_alias(parser, "--qaic_output_dir", dest="qaic_output_dir")
    parser.add_argument("--device-group", default="0", help="Comma-separated QAIC device ids, e.g. 0 or 0,1,2,3.")
    _add_hidden_alias(parser, "--device_group", dest="device_group")
    batch_flags = ["--batch-size"] if include_batch_alias else ["--qaic-batch-size"]
    parser.add_argument(*batch_flags, dest="batch_size", type=int, default=batch_default)
    qaic_batch_aliases = ["--qaic_batch_size"]
    if include_batch_alias:
        qaic_batch_aliases.insert(0, "--qaic-batch-size")
        qaic_batch_aliases.append("--batch_size")
    _add_hidden_alias(parser, *qaic_batch_aliases, dest="batch_size", type=int)
    parser.add_argument("--ctx-len", type=int, default=4096)
    _add_hidden_alias(parser, "--ctx_len", dest="ctx_len", type=int)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    _add_hidden_alias(parser, "--prefill_seq_len", dest="prefill_seq_len", type=int)
    parser.add_argument("--generation-len", type=int, default=60)
    _add_hidden_alias(parser, "--generation_len", dest="generation_len", type=int)
    parser.add_argument("--num-cores", type=int, default=16)
    _add_hidden_alias(parser, "--num_cores", dest="num_cores", type=int)
    parser.add_argument("--compile-dir-base", default=None)
    _add_hidden_alias(parser, "--compile_dir_base", dest="compile_dir_base")
    parser.add_argument("--prompts", nargs="+", default=None)
    parser.add_argument("--no-mxfp6-matmul", action="store_true")
    _add_hidden_alias(parser, "--no_mxfp6_matmul", dest="no_mxfp6_matmul", action="store_true")
    parser.add_argument("--no-mxint8-kv-cache", action="store_true")
    _add_hidden_alias(parser, "--no_mxint8_kv_cache", dest="no_mxint8_kv_cache", action="store_true")
    if include_videomme_args:
        parser.add_argument("--videomme-dataset-path", default=None, help="Local Video-MME JSON/JSONL file or directory for QAIC VLM accuracy.")
        _add_hidden_alias(parser, "--videomme_dataset_path", dest="videomme_dataset_path")
        parser.add_argument("--videomme-video-root", default=None, help="Directory containing Video-MME video files referenced by dataset rows.")
        _add_hidden_alias(parser, "--videomme_video_root", dest="videomme_video_root")
        parser.add_argument("--videomme-split", default="test")
        _add_hidden_alias(parser, "--videomme_split", dest="videomme_split")
        parser.add_argument("--videomme-num-samples", type=int, default=None, help="Number of Video-MME samples to run on QAIC. Defaults to all loaded rows.")
        _add_hidden_alias(parser, "--videomme_num_samples", dest="videomme_num_samples", type=int)
        parser.add_argument("--videomme-num-frames", type=int, default=8)
        _add_hidden_alias(parser, "--videomme_num_frames", dest="videomme_num_frames", type=int)
        parser.add_argument("--videomme-fps", type=float, default=None)
        _add_hidden_alias(parser, "--videomme_fps", dest="videomme_fps", type=float)
        parser.add_argument("--videomme-use-subtitles", action="store_true")
        _add_hidden_alias(parser, "--videomme_use_subtitles", dest="videomme_use_subtitles", action="store_true")


def _add_run_all_global_stage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-data-samples",
        "--num_data_samples",
        type=int,
        default=None,
        help="Number of data samples for both analysis and evaluation stages.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="global_batch_size",
        type=int,
        default=None,
        help="Batch size for both analysis and QAIC stages.",
    )


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    num_data_samples = getattr(args, "num_data_samples", None)
    if num_data_samples is not None:
        if getattr(args, "analysis_samples", None) is None:
            args.analysis_samples = num_data_samples
        if getattr(args, "eval_samples", None) is None:
            args.eval_samples = num_data_samples

    global_batch_size = getattr(args, "global_batch_size", None)
    if global_batch_size is not None:
        if getattr(args, "analysis_batch_size", None) is None:
            args.analysis_batch_size = global_batch_size
        if getattr(args, "batch_size", None) is None:
            args.batch_size = global_batch_size

    if hasattr(args, "analysis_samples") and args.analysis_samples is None:
        args.analysis_samples = 64
    if hasattr(args, "eval_samples") and args.eval_samples is None:
        args.eval_samples = 50
    if hasattr(args, "analysis_batch_size") and args.analysis_batch_size is None:
        args.analysis_batch_size = 1
    if hasattr(args, "batch_size") and args.batch_size is None:
        args.batch_size = 1
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Generate weak-layer report and candidate plans.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(analyze)
    _add_analysis_args(analyze, include_short_aliases=True, include_num_data_alias=True)
    analyze.set_defaults(func=run_analyze)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate candidate plans and select best_plan.json.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(evaluate)
    _add_eval_args(evaluate, include_short_aliases=True, include_num_data_alias=True)
    evaluate.set_defaults(func=run_evaluate)

    qaic = subparsers.add_parser("qaic", help="Compile and benchmark the selected plan on QAIC.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(qaic)
    _add_qaic_args(qaic, include_batch_alias=True)
    qaic.set_defaults(func=run_qaic)

    run_all_parser = subparsers.add_parser("run-all", help="Run analyze, evaluate, and QAIC stages.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(run_all_parser)
    _add_run_all_global_stage_args(run_all_parser)
    _add_analysis_args(
        run_all_parser,
        include_short_aliases=False,
        include_num_data_alias=False,
        sample_default=None,
        batch_default=None,
    )
    _add_eval_args(
        run_all_parser,
        include_short_aliases=False,
        include_num_data_alias=False,
        sample_default=None,
        generation_dest="eval_generation_len",
    )
    _add_qaic_args(run_all_parser, include_batch_alias=False, batch_default=None, include_videomme_args=False, include_skip_layers=False)
    run_all_parser.set_defaults(func=run_all)

    return parser


def main() -> None:
    args = _normalize_args(build_parser().parse_args())
    args.func(args)


if __name__ == "__main__":
    main()
