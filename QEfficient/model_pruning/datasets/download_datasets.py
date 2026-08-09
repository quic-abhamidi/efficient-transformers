#!/usr/bin/env python3
"""Download datasets used by QEfficient model-pruning workflows.

The script mirrors the supported dataset aliases in
``qeff_model_optimizer.analysis.datasets.SUPPORTED_DATASETS`` and adds
``videomme`` for VLM evaluation. It downloads Hugging Face datasets, saves them
under ``QEfficient/model_pruning/datasets/downloaded`` by default, writes a
manifest, and can optionally export JSONL files for direct evaluator use.

Video-MME metadata is downloaded like any other Hugging Face dataset. The video
files are much larger and require ``yt-dlp``, so they are downloaded only when
``--download-videomme-videos`` is passed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "downloaded"


@dataclass(frozen=True)
class DatasetSpec:
    alias: str
    repo_id: str
    config: str | None
    split: str
    description: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec("gsm8k", "openai/gsm8k", "main", "train", "Grade-school math word problems"),
    "mbpp": DatasetSpec(
        "mbpp", "google-research-datasets/mbpp", "sanitized", "train", "Python programming problems"
    ),
    "wikitext": DatasetSpec(
        "wikitext", "Salesforce/wikitext", "wikitext-2-raw-v1", "train", "Language modeling text"
    ),
    "hellaswag": DatasetSpec("hellaswag", "Rowan/hellaswag", None, "train", "Commonsense sentence completion"),
    "winogrande": DatasetSpec(
        "winogrande", "allenai/winogrande", "winogrande_xl", "train", "Commonsense pronoun resolution"
    ),
    "arc_challenge": DatasetSpec(
        "arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "train", "AI2 ARC Challenge science QA"
    ),
    "arc_easy": DatasetSpec("arc_easy", "allenai/ai2_arc", "ARC-Easy", "train", "AI2 ARC Easy science QA"),
    "openbookqa": DatasetSpec("openbookqa", "allenai/openbookqa", "main", "train", "Open-book science QA"),
    "piqa": DatasetSpec("piqa", "baber/piqa", None, "train", "Physical interaction QA"),
    "mmlu": DatasetSpec("mmlu", "cais/mmlu", "all", "test", "Massive Multitask Language Understanding"),
    "boolq": DatasetSpec("boolq", "google/boolq", None, "train", "Boolean question answering"),
    "truthfulqa": DatasetSpec(
        "truthfulqa",
        "truthfulqa/truthful_qa",
        "multiple_choice",
        "validation",
        "Truthfulness multiple-choice QA",
    ),
    "lambada": DatasetSpec("lambada", "EleutherAI/lambada_openai", None, "test", "LAMBADA language modeling"),
    "mmlu_pro": DatasetSpec("mmlu_pro", "TIGER-Lab/MMLU-Pro", None, "test", "Harder MMLU-style reasoning"),
    "bbh_causal": DatasetSpec("bbh_causal", "lukaemon/bbh", "causal_judgement", "test", "BBH causal judgement"),
    "bbh_logical_deduction": DatasetSpec(
        "bbh_logical_deduction",
        "lukaemon/bbh",
        "logical_deduction_five_objects",
        "test",
        "BBH logical deduction",
    ),
    "ifeval": DatasetSpec("ifeval", "HuggingFaceH4/ifeval", None, "train", "Instruction-following evaluation prompts"),
    "helpsteer2": DatasetSpec(
        "helpsteer2", "nvidia/HelpSteer2", None, "train", "HelpSteer2 preference/instruction data"
    ),
    "gsm_hard": DatasetSpec("gsm_hard", "reasoning-machines/gsm-hard", None, "train", "Harder GSM arithmetic prompts"),
    "orca_math": DatasetSpec(
        "orca_math", "microsoft/orca-math-word-problems-200k", None, "train", "Math word-problem corpus"
    ),
    "humanevalpack": DatasetSpec(
        "humanevalpack", "bigcode/humanevalpack", "python", "test", "Python HumanEvalPack prompts"
    ),
    "metamathqa": DatasetSpec("metamathqa", "meta-math/MetaMathQA", None, "train", "MetaMathQA reasoning prompts"),
    "videomme": DatasetSpec(
        "videomme", "lmms-lab/Video-MME", "videomme", "test", "Video-MME VLM multiple-choice QA metadata"
    ),
}


DEFAULT_DATASETS = [
    "gsm8k",
    "hellaswag",
    "mbpp",
    "wikitext",
    "winogrande",
    "arc_challenge",
    "arc_easy",
    "openbookqa",
    "piqa",
    "mmlu",
    "boolq",
    "truthfulqa",
    "lambada",
    "mmlu_pro",
    "bbh_causal",
    "bbh_logical_deduction",
    "ifeval",
    "helpsteer2",
    "gsm_hard",
    "orca_math",
    "humanevalpack",
    "metamathqa",
    "videomme",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download model-pruning datasets into QEfficient/model_pruning/datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=(
            "Dataset aliases to download. Use 'all' for every supported alias, "
            "or 'text' for all non-Video-MME datasets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where dataset artifacts are written.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional row limit saved/exported locally after the dataset is loaded. "
            "Hugging Face may still download the source dataset shard."
        ),
    )
    parser.add_argument(
        "--export-jsonl",
        action="store_true",
        help="Also export each loaded dataset split to <alias>/<alias>.jsonl.",
    )
    parser.add_argument(
        "--save-to-disk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save each loaded dataset with Dataset.save_to_disk().",
    )
    parser.add_argument(
        "--download-videomme-videos",
        action="store_true",
        help="Download Video-MME video files with yt-dlp using URLs from the metadata rows.",
    )
    parser.add_argument(
        "--videomme-video-dir",
        type=Path,
        default=None,
        help="Directory for Video-MME videos. Defaults to <output-dir>/videomme/videos.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download/rewrite local artifacts even if manifest files already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported aliases and exit.",
    )
    return parser.parse_args()


def resolve_aliases(raw_aliases: list[str]) -> list[str]:
    requested = []
    for alias in raw_aliases:
        if alias == "all":
            requested.extend(DEFAULT_DATASETS)
        elif alias == "text":
            requested.extend(name for name in DEFAULT_DATASETS if name != "videomme")
        else:
            requested.append(alias)
    unknown = sorted(set(requested) - set(DATASET_SPECS))
    if unknown:
        raise ValueError(f"Unsupported dataset aliases: {unknown}. Supported: {sorted(DATASET_SPECS)}")
    deduped = []
    seen = set()
    for alias in requested:
        if alias not in seen:
            deduped.append(alias)
            seen.add(alias)
    return deduped


def load_hf_dataset(spec: DatasetSpec, cache_dir: Path | None):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": spec.split}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    try:
        if spec.config is None:
            return load_dataset(spec.repo_id, **kwargs)
        return load_dataset(spec.repo_id, spec.config, **kwargs)
    except Exception:
        if spec.alias == "videomme" and spec.config is not None:
            # Some datasets versions expose Video-MME without an explicit config.
            return load_dataset(
                spec.repo_id,
                split=spec.split,
                **({"cache_dir": str(cache_dir)} if cache_dir else {}),
            )
        raise


def maybe_limit_dataset(dataset, limit: int | None):
    if limit is None:
        return dataset
    if limit <= 0:
        raise ValueError("--limit must be positive when provided")
    return dataset.select(range(min(limit, len(dataset))))


def write_jsonl(dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_manifest(alias_dir: Path, spec: DatasetSpec, dataset, artifacts: dict[str, str | None]) -> None:
    manifest = {
        "alias": spec.alias,
        "repo_id": spec.repo_id,
        "config": spec.config,
        "split": spec.split,
        "description": spec.description,
        "num_rows": len(dataset),
        "columns": list(getattr(dataset, "column_names", []) or []),
        "artifacts": artifacts,
    }
    (alias_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def download_dataset(
    alias: str,
    *,
    output_dir: Path,
    cache_dir: Path | None,
    limit: int | None,
    export_jsonl: bool,
    save_to_disk: bool,
    force: bool,
):
    spec = DATASET_SPECS[alias]
    alias_dir = output_dir / alias
    manifest_path = alias_dir / "manifest.json"
    if manifest_path.exists() and not force:
        print(f"[skip] {alias}: {manifest_path} exists; pass --force to rewrite")
        return None

    print(f"[download] {alias}: {spec.repo_id} config={spec.config} split={spec.split}")
    dataset = maybe_limit_dataset(load_hf_dataset(spec, cache_dir), limit)
    alias_dir.mkdir(parents=True, exist_ok=True)

    saved_path = None
    jsonl_path = None
    if save_to_disk:
        saved = alias_dir / "dataset"
        if saved.exists() and force:
            shutil.rmtree(saved)
        dataset.save_to_disk(str(saved))
        saved_path = str(saved)
    if export_jsonl or alias == "videomme":
        jsonl = alias_dir / f"{alias}.jsonl"
        write_jsonl(dataset, jsonl)
        jsonl_path = str(jsonl)

    write_manifest(alias_dir, spec, dataset, {"dataset_dir": saved_path, "jsonl": jsonl_path})
    print(f"[done] {alias}: rows={len(dataset)} dir={alias_dir}")
    return dataset


def extract_videomme_url(row: dict[str, Any]) -> tuple[str | None, str | None]:
    url = row.get("url") or row.get("video_url") or row.get("videoUrl")
    video_id = row.get("video_id") or row.get("videoID") or row.get("youtube_id") or row.get("video") or url
    return (str(url) if url else None, str(video_id) if video_id else None)


def write_videomme_url_file(videomme_jsonl: Path, url_file: Path) -> int:
    seen = set()
    url_file.parent.mkdir(parents=True, exist_ok=True)
    with videomme_jsonl.open(encoding="utf-8") as src, url_file.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            url, video_id = extract_videomme_url(json.loads(line))
            if url and video_id not in seen:
                seen.add(video_id)
                dst.write(f"{url}\n")
    return len(seen)


def download_videomme_videos(output_dir: Path, video_dir: Path | None) -> None:
    alias_dir = output_dir / "videomme"
    jsonl = alias_dir / "videomme.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"Video-MME metadata JSONL not found: {jsonl}")
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp is required for --download-videomme-videos but was not found on PATH")

    target_dir = video_dir or alias_dir / "videos"
    target_dir.mkdir(parents=True, exist_ok=True)
    url_file = alias_dir / "video_urls.txt"
    count = write_videomme_url_file(jsonl, url_file)
    if count == 0:
        raise RuntimeError(f"No Video-MME URLs found in {jsonl}")
    print(f"[videomme] wrote {count} URLs to {url_file}")

    cmd = [
        "yt-dlp",
        "--ignore-errors",
        "--continue",
        "--no-overwrites",
        "--format",
        "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "--output",
        str(target_dir / "%(id)s.%(ext)s"),
        "--batch-file",
        str(url_file),
    ]
    print("[videomme] running:", " ".join(cmd))
    subprocess.run(cmd, check=False)
    downloaded = sum(1 for path in target_dir.rglob("*") if path.is_file())
    print(f"[videomme] video files under {target_dir}: {downloaded}")


def print_supported() -> None:
    for alias in DEFAULT_DATASETS:
        spec = DATASET_SPECS[alias]
        config = f"/{spec.config}" if spec.config else ""
        print(f"{alias:24s} {spec.repo_id}{config} [{spec.split}] - {spec.description}")


def main() -> int:
    args = parse_args()
    if args.list:
        print_supported()
        return 0

    aliases = resolve_aliases(args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    failures: dict[str, str] = {}
    for alias in aliases:
        try:
            download_dataset(
                alias,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                limit=args.limit,
                export_jsonl=args.export_jsonl,
                save_to_disk=args.save_to_disk,
                force=args.force,
            )
        except Exception as exc:
            failures[alias] = f"{type(exc).__name__}: {exc}"
            print(f"[fail] {alias}: {failures[alias]}", file=sys.stderr)

    if args.download_videomme_videos:
        try:
            download_videomme_videos(args.output_dir, args.videomme_video_dir)
        except Exception as exc:
            failures["videomme_videos"] = f"{type(exc).__name__}: {exc}"
            print(f"[fail] videomme_videos: {failures['videomme_videos']}", file=sys.stderr)

    summary = {
        "output_dir": str(args.output_dir),
        "requested": aliases,
        "failures": failures,
    }
    (args.output_dir / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] wrote {args.output_dir / 'download_summary.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
