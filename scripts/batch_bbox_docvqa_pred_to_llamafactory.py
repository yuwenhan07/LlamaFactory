#!/usr/bin/env python3

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert bbox prediction jsonl files into pred-crop LlamaFactory datasets."
    )
    parser.add_argument(
        "--prediction-dir",
        default="api/output/pred_bbox",
        help="Directory containing bbox prediction jsonl files.",
    )
    parser.add_argument(
        "--source-jsonl",
        default="data/bbox_docvqa_bbox_out_1000/bbox_docvqa_bbox_1000.jsonl",
        help="Source bbox dataset used to generate the predictions.",
    )
    parser.add_argument(
        "--output-root",
        default="data/bbox_docvqa_pred_crop",
        help="Root directory for converted pred-crop datasets.",
    )
    parser.add_argument(
        "--script-path",
        default="scripts/bbox_docvqa_pred_to_llamafactory.py",
        help="Conversion script path.",
    )
    parser.add_argument("--num-workers", type=int, default=16, help="Worker count passed to the converter.")
    parser.add_argument("--chunksize", type=int, default=8, help="Chunksize passed to the converter.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--verbose", action="store_true", help="Print skipped-sample reasons in converter output.")
    return parser.parse_args()


def model_name_from_prediction(path: Path) -> str:
    return path.stem.removeprefix("pred_bbox__")


def dataset_key(model_name: str) -> str:
    return model_name.replace(".", "_").replace("-", "_")


def build_command(args: argparse.Namespace, prediction_path: Path) -> list[str]:
    model_name = model_name_from_prediction(prediction_path)
    model_key = dataset_key(model_name)
    dataset_dir = Path(args.output_root) / model_key
    dataset_name = f"bbox_docvqa_pred_crop_{model_key}"

    command = [
        sys.executable,
        args.script_path,
        "--source-jsonl",
        args.source_jsonl,
        "--prediction-jsonl",
        str(prediction_path),
        "--dataset-dir",
        str(dataset_dir),
        "--dataset-name",
        dataset_name,
        "--num-workers",
        str(args.num_workers),
        "--chunksize",
        str(args.chunksize),
    ]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.verbose:
        command.append("--verbose")
    return command


def main() -> None:
    args = parse_args()
    prediction_dir = Path(args.prediction_dir).resolve()
    prediction_files = sorted(prediction_dir.glob("*.jsonl"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in {prediction_dir}")

    for prediction_path in prediction_files:
        command = build_command(args, prediction_path)
        print("[run]", shlex.join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
