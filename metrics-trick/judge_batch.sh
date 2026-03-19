for f in ./output/pred_crop_answer_qwen3.5-27b/*.jsonl; 
    do     base=$(basename "$f");    python judge.py "$f" "./judge_output/pred_crop_answer_qwen3.5-27b/$base";   
done