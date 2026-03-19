#!/usr/bin/env bash

set -euo pipefail

OUTPUT_DIR="./output/pred_crop_answer_qwen3.5-27b"
mkdir -p "./output/pred_crop_answer_qwen3.5-27b"

# pred crop answer | qwen3-vl-8b-instruct
python batch_test_bbox_docvqa_api.py \
  --dataset-jsonl ./bbox_docvqa_pred_crop/qwen3-vl-8b-instruct/pred_bbox_crop_qwen3-vl-8b-instruct.jsonl \
  --output "./output/pred_crop_answer_qwen3.5-27b/pred_bbox_crop_qwen3-vl-8b-instruct.jsonl" \
  --model qwen3.5-27b \
  --base-url https://qianfan.baidubce.com/v2 

# pred crop answer | qwen3-vl-32b-instruct
python batch_test_bbox_docvqa_api.py \
  --dataset-jsonl ./bbox_docvqa_pred_crop/qwen3-vl-32b-instruct/pred_bbox_crop_qwen3-vl-32b-instruct.jsonl \
  --output "./output/pred_crop_answer_qwen3.5-27b/pred_bbox_crop_qwen3-vl-32b-instruct.jsonl" \
  --model qwen3.5-27b \
  --base-url https://qianfan.baidubce.com/v2 \
  --max-workers 5 \
  --resume

# pred crop answer | qwen3-vl-235b-a22b-instruct
python batch_test_bbox_docvqa_api.py \
  --dataset-jsonl ./bbox_docvqa_pred_crop/qwen3-vl-235b-a22b-instruct/pred_bbox_crop_qwen3-vl-235b-a22b-instruct.jsonl \
  --output "./output/pred_crop_answer_qwen3.5-27b/pred_bbox_crop_qwen3-vl-235b-a22b-instruct.jsonl" \
  --model qwen3.5-27b \
  --base-url https://qianfan.baidubce.com/v2 \
  --max-workers 5 \
  --resume

# pred crop answer | qwen3.5-27b
python batch_test_bbox_docvqa_api.py \
  --dataset-jsonl ./bbox_docvqa_pred_crop/qwen3.5-27b/pred_bbox_crop_qwen3.5-27b.jsonl \
  --output "./output/pred_crop_answer_qwen3.5-27b/pred_bbox_crop_qwen3.5-27b.jsonl" \
  --model qwen3.5-27b \
  --base-url https://qianfan.baidubce.com/v2 \
  --max-workers 5 \
  --resume

# pred crop answer | qwen3.5-122b-a5b
python batch_test_bbox_docvqa_api.py \
  --dataset-jsonl ./bbox_docvqa_pred_crop/qwen3.5-122b-a5b/pred_bbox_crop_qwen3.5-122b-a5b.jsonl \
  --output "./output/pred_crop_answer_qwen3.5-27b/pred_bbox_crop_qwen3.5-122b-a5b.jsonl" \
  --model qwen3.5-27b \
  --base-url https://qianfan.baidubce.com/v2 \
  --max-workers 5 \
  --resume

# pred crop answer | qwen3.5-397b-a17b
python batch_test_bbox_docvqa_api.py \
  --dataset-jsonl ./bbox_docvqa_pred_crop/qwen3.5-397b-a17b/pred_bbox_crop_qwen3.5-397b-a17b.jsonl \
  --output "./output/pred_crop_answer_qwen3.5-27b/pred_bbox_crop_qwen3.5-397b-a17b.jsonl" \
  --model qwen3.5-27b \
  --base-url https://qianfan.baidubce.com/v2 \
  --max-workers 5 \
  --resume

# pred crop answer | internvl3-38b
python batch_test_bbox_docvqa_api.py \
  --dataset-jsonl ./bbox_docvqa_pred_crop/internvl3-38b/pred_bbox_crop_internvl3-38b.jsonl \
  --output "./output/pred_crop_answer_qwen3.5-27b/pred_bbox_crop_internvl3-38b.jsonl" \
  --model qwen3.5-27b \
  --base-url https://qianfan.baidubce.com/v2 \
  --max-workers 5 \
  --resume

# pred crop answer | ernie-5.0
python batch_test_bbox_docvqa_api.py \
  --dataset-jsonl ./bbox_docvqa_pred_crop/ernie-5.0/pred_bbox_crop_ernie-5.0.jsonl \
  --output "./output/pred_crop_answer_qwen3.5-27b/pred_bbox_crop_ernie-5.0.jsonl" \
  --model qwen3.5-27b \
  --base-url https://qianfan.baidubce.com/v2 \
  --max-workers 5 \
  --resume

# pred crop answer | qwen3-vl-8b-sft-0313
python batch_test_bbox_docvqa_api.py \
  --dataset-jsonl ./bbox_docvqa_pred_crop/qwen3-vl-8b-sft-0313/pred_bbox_crop_qwen3-vl-8b-sft-0313.jsonl \
  --output "./output/pred_crop_answer_qwen3.5-27b/pred_bbox_crop_qwen3-vl-8b-sft-0313.jsonl" \
  --model qwen3.5-27b \
  --base-url https://qianfan.baidubce.com/v2 \
  --max-workers 5 \
  --resume

# pred crop answer | qwen3-vl-8b-sft-0313-2
python batch_test_bbox_docvqa_api.py \
  --dataset-jsonl ./bbox_docvqa_pred_crop/qwen3-vl-8b-sft-0313-2/pred_bbox_crop_qwen3-vl-8b-sft-0313-2.jsonl \
  --output "./output/pred_crop_answer_qwen3.5-27b/pred_bbox_crop_qwen3-vl-8b-sft-0313-2.jsonl" \
  --model qwen3.5-27b \
  --base-url https://qianfan.baidubce.com/v2 \
  --max-workers 5 \
  --resume
