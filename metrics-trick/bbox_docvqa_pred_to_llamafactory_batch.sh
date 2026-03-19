#!/usr/bin/env bash

set -euo pipefail

SOURCE_JSONL="${SOURCE_JSONL:-../data/bbox_docvqa_bbox_out_1000/bbox_docvqa_bbox_1000.jsonl}"
PREDICTION_DIR="${PREDICTION_DIR:-../api/output/pred_bbox}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./bbox_docvqa_pred_crop}"
NUM_WORKERS="${NUM_WORKERS:-16}"
CHUNKSIZE="${CHUNKSIZE:-8}"
MIN_CROP_EDGE="${MIN_CROP_EDGE:-28}"
PROMPT_PREFIX="${PROMPT_PREFIX:-Answer the question using only the document image(s). Return only the final answer with no explanation.}"

mkdir -p "$OUTPUT_ROOT"

for prediction_jsonl in "$PREDICTION_DIR"/*.jsonl; do
    [ -e "$prediction_jsonl" ] || continue

    base_name="$(basename "$prediction_jsonl" .jsonl)"
    model_name="${base_name#pred_bbox__}"
    dataset_dir="${OUTPUT_ROOT}/${model_name}"
    dataset_name="pred_bbox_crop_${model_name}"

    echo "[run] ${base_name}.jsonl -> ${dataset_dir}"

    python bbox_docvqa_pred_to_llamafactory.py \
        --source-jsonl "$SOURCE_JSONL" \
        --prediction-jsonl "$prediction_jsonl" \
        --dataset-dir "$dataset_dir" \
        --dataset-name "$dataset_name" \
        --num-workers "$NUM_WORKERS" \
        --chunksize "$CHUNKSIZE" \
        --min-crop-edge "$MIN_CROP_EDGE" \
        --prompt-prefix "$PROMPT_PREFIX"
done
