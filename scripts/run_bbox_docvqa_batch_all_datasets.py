#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def load_dataset_info(dataset_dir):
    dataset_info_path = dataset_dir / "dataset_info.json"
    with dataset_info_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid dataset_info.json: {dataset_info_path}")
    return data


def select_dataset_name(dataset_dir, dataset_info, preferred_name=None):
    if preferred_name:
        if preferred_name not in dataset_info:
            raise KeyError(f"Dataset name {preferred_name!r} not found in {dataset_dir / 'dataset_info.json'}")
        return preferred_name

    if len(dataset_info) == 1:
        return next(iter(dataset_info))

    candidates = [
        f"bbox_docvqa_pred_crop_{dataset_dir.name}",
        "bbox_docvqa_pred_crop",
    ]
    for candidate in candidates:
        if candidate in dataset_info:
            return candidate

    raise ValueError(
        f"Cannot determine dataset name for {dataset_dir}. "
        f"Available names: {', '.join(dataset_info.keys())}"
    )


def discover_dataset_dirs(datasets_root):
    return sorted(
        path for path in datasets_root.iterdir() if path.is_dir() and (path / "dataset_info.json").is_file()
    )


def has_complete_dataset_outputs(dataset_output_root):
    required_files = [
        dataset_output_root / "summary.json",
        dataset_output_root / "summary.csv",
    ]
    return all(path.is_file() for path in required_files)


def write_summary(rows, output_root):
    output_root.mkdir(parents=True, exist_ok=True)

    summary_json = output_root / "batch_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    summary_csv = output_root / "batch_summary.csv"
    fieldnames = [
        "dataset_dir_name",
        "dataset_dir",
        "dataset_name",
        "output_root",
        "status",
        "returncode",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"Batch summary written to {summary_json}")
    print(f"Batch summary written to {summary_csv}")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Run scripts/run_bbox_docvqa_batch.py over every dataset under a root directory."
    )
    parser.add_argument(
        "--models-json",
        required=True,
        help="Path to the model config JSON used by run_bbox_docvqa_batch.py.",
    )
    parser.add_argument(
        "--datasets-root",
        default=str(repo_root / "data" / "bbox_docvqa_pred_crop"),
        help="Root directory that contains one subdirectory per converted dataset.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Base output directory. Each dataset will write to a separate subdirectory here.",
    )
    parser.add_argument(
        "--runner",
        default=str(script_dir / "run_bbox_docvqa_batch.py"),
        help="Path to the existing single-dataset batch runner.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name override. If unset, it is inferred from each dataset_info.json.",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=[],
        help="Dataset directory names to skip, for example: internvl3_38b.",
    )
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--cutoff-len", type=int, default=20480)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--image-max-pixels", type=int, default=26214400)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--vllm-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--hf-precision", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running the remaining datasets if one dataset fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip finished datasets and pass --resume to the inner runner.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets_root = Path(args.datasets_root).resolve()
    output_root = Path(args.output_root).resolve()
    runner = Path(args.runner).resolve()
    models_json = Path(args.models_json).resolve()

    dataset_dirs = discover_dataset_dirs(datasets_root)
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories with dataset_info.json found under {datasets_root}")

    summary_rows = []
    for dataset_dir in dataset_dirs:
        if dataset_dir.name in set(args.exclude_datasets):
            print(f"[skip] {dataset_dir.name} (excluded)")
            summary_rows.append(
                {
                    "dataset_dir_name": dataset_dir.name,
                    "dataset_dir": str(dataset_dir),
                    "dataset_name": None,
                    "output_root": str(output_root / dataset_dir.name),
                    "status": "excluded",
                    "returncode": 0,
                }
            )
            continue

        dataset_info = load_dataset_info(dataset_dir)
        dataset_name = select_dataset_name(dataset_dir, dataset_info, args.dataset_name)
        dataset_output_root = output_root / dataset_dir.name

        command = [
            sys.executable,
            str(runner),
            "--models-json",
            str(models_json),
            "--dataset-dir",
            str(dataset_dir),
            "--dataset-name",
            dataset_name,
            "--output-root",
            str(dataset_output_root),
            "--backend",
            args.backend,
            "--cutoff-len",
            str(args.cutoff_len),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--image-max-pixels",
            str(args.image_max_pixels),
            "--vllm-batch-size",
            str(args.vllm_batch_size),
            "--per-device-eval-batch-size",
            str(args.per_device_eval_batch_size),
            "--hf-precision",
            args.hf_precision,
        ]
        if args.max_samples is not None:
            command.extend(["--max-samples", str(args.max_samples)])
        if args.resume:
            command.append("--resume")

        print(f"[dataset] {dataset_dir.name}")
        row = {
            "dataset_dir_name": dataset_dir.name,
            "dataset_dir": str(dataset_dir),
            "dataset_name": dataset_name,
            "output_root": str(dataset_output_root),
            "status": "dry_run" if args.dry_run else "success",
            "returncode": 0,
        }

        if args.resume and has_complete_dataset_outputs(dataset_output_root):
            row["status"] = "skipped"
            print(f"[skip] {dataset_dir.name}")
            summary_rows.append(row)
            continue

        print(" ".join(command))

        if not args.dry_run:
            result = subprocess.run(command, check=False)
            row["returncode"] = result.returncode
            row["status"] = "success" if result.returncode == 0 else "failed"
            if result.returncode != 0 and not args.continue_on_error:
                summary_rows.append(row)
                write_summary(summary_rows, output_root)
                raise SystemExit(result.returncode)

        summary_rows.append(row)

    write_summary(summary_rows, output_root)


if __name__ == "__main__":
    main()
