#!/usr/bin/env bash

set -euo pipefail

INPUT_DIR="api/output/pred_bbox"
OUTPUT_DIR="metrics-trick/output/pred_bbox_iou"

mkdir -p "$OUTPUT_DIR"

for f in "$INPUT_DIR"/*.jsonl; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    echo "[run] $base"
    python metrics-trick/eval_iou.py "$f" --output-dir "$OUTPUT_DIR"
done

python metrics-trick/summarize_iou_metrics.py "$OUTPUT_DIR"
