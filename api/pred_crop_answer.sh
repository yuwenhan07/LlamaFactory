#!/usr/bin/env bash

set -euo pipefail

# pred crop answer | qwen3.5-397b-a17b
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_pred_crop/qwen3_5_397b_a17b/bbox_docvqa_pred_crop_qwen3_5_397b_a17b.jsonl --output output/pred_crop_answer/pred_crop_answer__qwen3.5-397b-a17b.jsonl --model qwen3.5-397b-a17b --base-url https://qianfan.baidubce.com/v2 &

# pred crop answer | qwen3.5-122b-a10b
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_pred_crop/qwen3_5_122b_a10b/bbox_docvqa_pred_crop_qwen3_5_122b_a10b.jsonl --output output/pred_crop_answer/pred_crop_answer__qwen3.5-122b-a10b.jsonl --model qwen3.5-122b-a10b --base-url https://qianfan.baidubce.com/v2 &

# pred crop answer | qwen3.5-27b
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_pred_crop/qwen3_5_27b/bbox_docvqa_pred_crop_qwen3_5_27b.jsonl --output output/pred_crop_answer/pred_crop_answer__qwen3.5-27b.jsonl --model qwen3.5-27b --base-url https://qianfan.baidubce.com/v2 &

# pred crop answer | qwen3-vl-235b-a22b-instruct
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_pred_crop/qwen3_vl_235b_a22b_instruct/bbox_docvqa_pred_crop_qwen3_vl_235b_a22b_instruct.jsonl --output output/pred_crop_answer/pred_crop_answer__qwen3-vl-235b-a22b-instruct.jsonl --model qwen3-vl-235b-a22b-instruct --base-url https://qianfan.baidubce.com/v2 &

# pred crop answer | qwen3-vl-32b-instruct
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_pred_crop/qwen3_vl_32b_instruct/bbox_docvqa_pred_crop_qwen3_vl_32b_instruct.jsonl --output output/pred_crop_answer/pred_crop_answer__qwen3-vl-32b-instruct.jsonl --model qwen3-vl-32b-instruct --base-url https://qianfan.baidubce.com/v2 &

# pred crop answer | qwen3-vl-8b-instruct
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_pred_crop/qwen3_vl_8b_instruct/bbox_docvqa_pred_crop_qwen3_vl_8b_instruct.jsonl --output output/pred_crop_answer/pred_crop_answer__qwen3-vl-8b-instruct.jsonl --model qwen3-vl-8b-instruct --base-url https://qianfan.baidubce.com/v2 &

# pred crop answer | internvl3-38b
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_pred_crop/internvl3_38b/bbox_docvqa_pred_crop_internvl3_38b.jsonl --output output/pred_crop_answer/pred_crop_answer__internvl3-38b.jsonl --model internvl3-38b --base-url https://qianfan.baidubce.com/v2 &

# pred crop answer | ernie-5.0
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_pred_crop/ernie_5_0/bbox_docvqa_pred_crop_ernie_5_0.jsonl --output output/pred_crop_answer/pred_crop_answer__ernie-5.0.jsonl --model ernie-5.0 --base-url https://qianfan.baidubce.com/v2 &

wait
