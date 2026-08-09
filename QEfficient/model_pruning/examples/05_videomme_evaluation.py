#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run baseline vs optimized Video-MME accuracy for a VLM pruning plan.

Example:
    python QEfficient/model_pruning/examples/05_videomme_evaluation.py \
      --model Qwen/Qwen3-VL-8B-Instruct \
      --videomme-dataset-path /data/Video-MME/videomme.jsonl \
      --videomme-video-root /data/Video-MME/videos \
      --skip-layers 16 5 15 14 17 \
      --num-samples 50 \
      --output-dir results/model_pruning/qwen3_vl_videomme_top5
"""

from __future__ import annotations

import argparse

from QEfficient.model_pruning import nas_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--videomme-dataset-path", required=True)
    parser.add_argument("--videomme-video-root", default=None)
    parser.add_argument("--skip-layers", nargs="+", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--generation-len", type=int, default=16)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--use-subtitles", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        "evaluate",
        "--model",
        args.model,
        "--skip-layers",
        *[str(layer) for layer in args.skip_layers],
        "--datasets",
        "videomme",
        "--num-samples",
        str(args.num_samples),
        "--eval-method",
        "videomme",
        "--videomme-dataset-path",
        args.videomme_dataset_path,
        "--videomme-num-frames",
        str(args.num_frames),
        "--generation-len",
        str(args.generation_len),
        "--dtype",
        args.dtype,
        "--device-map",
        args.device_map,
        "--output-dir",
        args.output_dir,
    ]
    if args.videomme_video_root:
        cmd.extend(["--videomme-video-root", args.videomme_video_root])
    if args.fps is not None:
        cmd.extend(["--videomme-fps", str(args.fps)])
    if args.use_subtitles:
        cmd.append("--videomme-use-subtitles")
    if args.verbose:
        cmd.append("--verbose")
    nas_pipeline.main(cmd)


if __name__ == "__main__":
    main()
