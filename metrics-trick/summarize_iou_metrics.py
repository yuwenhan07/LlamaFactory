#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODEL_NAME_MAP = {
    "qwen3-vl-8b-instruct": "Qwen3-VL-8B",
    "qwen3-vl-32b-instruct": "Qwen3-VL-32B",
    "qwen3-vl-235b-a22b-instruct": "Qwen3-VL-235B-A22B",
    "qwen3.5-27b": "Qwen3.5-27B",
    "qwen3.5-122b-a10b": "Qwen35-122B-A10B",
    "qwen3.5-397b-a17b": "Qwen3.5-397B-A17B",
    "internvl3-38b": "InternVL3-38B",
    "ernie-5.0": "Erine-5",
    "qwen3-vl-8b-sft-0313": "qwen3-vl-8b-sft-0313",
}

MODEL_ORDER = [
    "Qwen3-VL-8B",
    "Qwen3-VL-32B",
    "Qwen3-VL-235B-A22B",
    "Qwen3.5-27B",
    "Qwen35-122B-A10B",
    "Qwen3.5-397B-A17B",
    "InternVL3-38B",
    "Erine-5",
    "qwen3-vl-8b-sft-0313",
]


def extract_model_name(metrics_path: Path, metrics: dict[str, Any]) -> str:
    input_file = str(metrics.get("input_file", ""))
    if input_file:
        input_name = Path(input_file).stem
        prefix = "pred_bbox__"
        if input_name.startswith(prefix):
            return input_name[len(prefix) :]

    stem = metrics_path.stem
    suffix = "_iou_metrics"
    prefix = "pred_bbox__"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix) : -len(suffix)]
    return stem


def build_summary_row(metrics_path: Path) -> dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    thresholds = metrics.get("sample_iou_thresholds", {})
    return {
        "model": MODEL_NAME_MAP.get(extract_model_name(metrics_path, metrics), extract_model_name(metrics_path, metrics)),
        "mean_sample_iou": metrics.get("mean_sample_iou", 0.0),
        "iou@0.3": thresholds.get("iou@0.3", {}).get("rate", 0.0),
        "iou@0.5": thresholds.get("iou@0.5", {}).get("rate", 0.0),
        "iou@0.7": thresholds.get("iou@0.7", {}).get("rate", 0.0),
        "sample_iou_thresholds": thresholds,
    }


def write_json(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = ["model", "mean_sample_iou", "iou@0.3", "iou@0.5", "iou@0.7"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize *_iou_metrics.json files.")
    parser.add_argument(
        "metrics_dir",
        nargs="?",
        default="metrics-trick/output/pred_bbox_iou",
        help="Directory containing *_iou_metrics.json files.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to write the summary json. Defaults to <metrics_dir>/iou_metrics_summary.json.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to write the summary csv. Defaults to <metrics_dir>/iou_metrics_summary.csv.",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir).resolve()
    metrics_files = sorted(metrics_dir.glob("*_iou_metrics.json"))
    summary_rows = [build_summary_row(path) for path in metrics_files]
    summary_rows = [row for row in summary_rows if row["model"] in MODEL_ORDER]
    order_index = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    summary_rows.sort(key=lambda item: (order_index.get(item["model"], len(MODEL_ORDER)), item["model"]))

    output_json = Path(args.output_json).resolve() if args.output_json else metrics_dir / "iou_metrics_summary.json"
    output_csv = Path(args.output_csv).resolve() if args.output_csv else metrics_dir / "iou_metrics_summary.csv"

    write_json(summary_rows, output_json)
    write_csv(summary_rows, output_csv)

    print(json.dumps({"summary_json": str(output_json), "summary_csv": str(output_csv), "models": len(summary_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
