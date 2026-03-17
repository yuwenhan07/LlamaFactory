# for f in api/output/crop/*.jsonl; 
#     do     base=$(basename "$f");     python metrics/judge.py "$f" "metrics/output_new/crop_judged/$base";   
# done

for f in api/output/page/*.jsonl; 
    do     base=$(basename "$f");     python metrics/judge.py "$f" "metrics/output_new/page_judged-1/$base";   
done


for f in api/output/pred_crop_answer//*.jsonl; 
    do     base=$(basename "$f");    python metrics/judge.py "$f" "metrics/output_new/pred_bbox_answer-models/$base";   
done