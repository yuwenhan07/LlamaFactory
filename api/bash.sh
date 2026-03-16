# crop
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_crop/bbox_docvqa_rel_crop.jsonl --output qianfan_debug_crop.jsonl --model qwen3.5-397b-a17b --base-url https://qianfan.baidubce.com/v2 

# page
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_page/bbox_docvqa_rel_page.jsonl --output qianfan_debug_page.jsonl --model qwen3.5-397b-a17b --base-url https://qianfan.baidubce.com/v2 

# document
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_document/bbox_docvqa_document.jsonl --output qianfan_debug_doc.jsonl --model qwen3.5-397b-a17b --base-url https://qianfan.baidubce.com/v2 

# predict bbox
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_bbox_out_1000/bbox_docvqa_bbox_1000.jsonl --output qianfan_debug_pred_bbox.jsonl --model qwen3.5-397b-a17b --base-url https://qianfan.baidubce.com/v2 