# for f in api/output/crop/*.jsonl; 
#     do     base=$(basename "$f");     python metrics/judge.py "$f" "metrics/output_new/crop_judged/$base";   
# done

for f in api/output/page/*.jsonl; 
    do     base=$(basename "$f");     python metrics/judge.py "$f" "metrics/output_new/page_judged-1/$base";   
done


for f in api/output/pred_crop_answer_qwen3.5-27b/*.jsonl; 
    do     base=$(basename "$f");    python metrics/judge.py "$f" "metrics/judge_output/pred_crop_answer_qwen3.5-27b/$base";   
done


for f in api/output/doc_clean/*.jsonl; 
    do     base=$(basename "$f");    python metrics/judge.py "$f" "metrics/judge_output/doc_clean/$base";   
done