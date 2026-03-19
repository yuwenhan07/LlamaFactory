# crop | "Claude Sonnet 4.6"
python test_bbox_docvqa_api.py --dataset-jsonl ../data/bbox_docvqa_crop/bbox_docvqa_rel_crop.jsonl --output output/crop/crop__Claude-Sonnet-4.6.jsonl --temperature 0 --model "Claude Sonnet 4.6" --base-url https://oneapi-comate.baidu-int.com/v1 --api-key-env COMATE_API_KEY --resume 


# page 
python test_bbox_docvqa_api.py --dataset-jsonl  ../data/bbox_docvqa_page/bbox_docvqa_rel_page.jsonl --output output/page/page__Claude-Sonnet-4.6.jsonl --temperature 0 --model "Claude Sonnet 4.6" --base-url https://oneapi-comate.baidu-int.com/v1 --api-key-env COMATE_API_KEY --resume 

# pred_bbox
python test_bbox_docvqa_api.py --dataset-jsonl  ../data/bbox_docvqa_bbox_out_1000/bbox_docvqa_bbox_1000.jsonl --output output/pred_bbox/pred_bbox__Claude-Sonnet-4.6.jsonl --temperature 0 --model "Claude Sonnet 4.6" --base-url https://oneapi-comate.baidu-int.com/v1 --api-key-env COMATE_API_KEY --resume 

# doc
python test_bbox_docvqa_api.py --dataset-jsonl  ../data/bbox_docvqa_document/bbox_docvqa_document.jsonl  --output output/doc/doc__Claude-Sonnet-4.6.jsonl --temperature 0 --model "Claude Sonnet 4.6" --base-url https://oneapi-comate.baidu-int.com/v1 --api-key-env COMATE_API_KEY --resume 
