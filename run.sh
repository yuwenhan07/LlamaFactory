# crop level test
python scripts/run_bbox_docvqa_batch.py   --models-json examples/inference/bbox_docvqa_models.example.json   --dataset-dir data/bbox_docvqa_crop   --dataset-name bbox_docvqa_rel_crop   --output-root saves/bbox_docvqa_crop --vllm-batch-size 8 
# page level test
python scripts/run_bbox_docvqa_batch.py   --models-json examples/inference/bbox_docvqa_models.example.json   --dataset-dir data/bbox_docvqa_page   --dataset-name bbox_docvqa_rel_page   --output-root saves/bbox_docvqa_page --vllm-batch-size 4 
# doc level test
python scripts/run_bbox_docvqa_batch.py   --models-json examples/inference/bbox_docvqa_models.example.json   --dataset-dir data/bbox_docvqa_document  --dataset-name bbox_docvqa_document   --output-root saves/bbox_docvqa_page --vllm-batch-size 1 