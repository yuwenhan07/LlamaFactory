#!/usr/bin/env bash

set -euo pipefail


# document | qwen3.5-122b-a10b
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_document/bbox_docvqa_document.jsonl --output output/doc/doc__qwen3.5-122b-a10b.jsonl --model qwen3.5-122b-a10b --base-url https://qianfan.baidubce.com/v2 --resume &

# document | qwen3.5-27b
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_document/bbox_docvqa_document.jsonl --output output/doc/doc__qwen3.5-27b.jsonl --model qwen3.5-27b --base-url https://qianfan.baidubce.com/v2 --resume &

# document | qwen3-vl-235b-a22b-instruct
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_document/bbox_docvqa_document.jsonl --output output/doc/doc__qwen3-vl-235b-a22b-instruct.jsonl --model qwen3-vl-235b-a22b-instruct --base-url https://qianfan.baidubce.com/v2 --resume &

# document | qwen3-vl-32b-instruct
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_document/bbox_docvqa_document.jsonl --output output/doc/doc__qwen3-vl-32b-instruct.jsonl --model qwen3-vl-32b-instruct --base-url https://qianfan.baidubce.com/v2 --resume &

# document | qwen3-vl-8b-instruct
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_document/bbox_docvqa_document.jsonl --output output/doc/doc__qwen3-vl-8b-instruct.jsonl --model qwen3-vl-8b-instruct --base-url https://qianfan.baidubce.com/v2 --resume &

# document | internvl3-38b
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_document/bbox_docvqa_document.jsonl --output output/doc/doc__internvl3-38b.jsonl --model internvl3-38b --base-url https://qianfan.baidubce.com/v2 --resume &

# document | ernie-5.0
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_document/bbox_docvqa_document.jsonl --output output/doc/doc__ernie-5.0.jsonl --model ernie-5.0 --base-url https://qianfan.baidubce.com/v2 --resume &

wait
