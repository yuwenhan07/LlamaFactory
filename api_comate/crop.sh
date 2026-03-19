#!/usr/bin/env bash

set -euo pipefail

# crop | "Claude Sonnet 4.6"
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_crop/bbox_docvqa_rel_crop.jsonl --output output/crop/crop__Claude-Sonnet-4.6.jsonl --temperature 0 --model "Claude Sonnet 4.6" --base-url https://oneapi-comate.baidu-int.com/v1 --api-key-env COMATE_API_KEY --resume &



# crop | "GLM-5"
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_crop/bbox_docvqa_rel_crop.jsonl --output output/crop/crop__GLM-5.jsonl  --model "GLM-5" --base-url https://oneapi-comate.baidu-int.com/v1 --api-key-env COMATE_API_KEY &

# crop | "Kimi-K2.5"
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_crop/bbox_docvqa_rel_crop.jsonl --output output/crop/crop__Kimi-K2.5.jsonl --model "Kimi-K2.5" --base-url https://oneapi-comate.baidu-int.com/v1 --api-key-env COMATE_API_KEY &


python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_crop/bbox_docvqa_rel_crop.jsonl --output output/crop/crop__MiniMax-M2.5.jsonl  --model "MiniMax-M2.5"  --temperature 0 --base-url https://oneapi-comate.baidu-int.com/v1 --api-key-env COMATE_API_KEY 
