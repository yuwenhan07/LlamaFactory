# crop | "Kimi-K2.5"
# pred_bbox
python test_bbox_docvqa_api.py --dataset-jsonl  ../data/bbox_docvqa_bbox_out_1000/bbox_docvqa_bbox_1000.jsonl --output output/pred_bbox/pred_bbox__gpt-5.2.jsonl --model "gpt-5.2" --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --resume 

# doc
nohup python test_bbox_docvqa_api.py --dataset-jsonl  ../data/bbox_docvqa_document/bbox_docvqa_document.jsonl  --output output/doc/doc__Kimi-K2.5.jsonl --model "Kimi-K2.5" --base-url https://oneapi-comate.baidu-int.com/v1 --api-key-env COMATE_API_KEY --resume > log/kimi-k2.5-doc-1.log 2>&1 &

python test_bbox_docvqa_api.py --dataset-jsonl  ../data/bbox_docvqa_document/bbox_docvqa_document.jsonl  --output output/doc/doc__gpt-5.2.jsonl --model "gpt-5.2" --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --resume 
