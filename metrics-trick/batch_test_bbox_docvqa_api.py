#!/usr/bin/env python3

import argparse
import base64
import json
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

from openai import OpenAI
from transformers.utils.versions import require_version


require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


THREAD_LOCAL = threading.local()


def to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    image_bytes = Path(image_path).read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def build_multimodal_content(text: str, image_paths: list[str], max_images: int | None) -> list[dict]:
    image_token = "<image>"
    image_count = text.count(image_token)
    usable_images = image_paths[:image_count]
    if max_images is not None:
        usable_images = usable_images[:max_images]
        image_count = min(image_count, max_images)

    parts = text.split(image_token)
    content: list[dict] = []

    for index, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})

        if index < image_count:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": to_data_url(usable_images[index])},
                }
            )

    return content


def extract_prompt_and_label(sample: dict) -> tuple[str, str]:
    messages = sample.get("messages", [])
    user_message = next(msg for msg in messages if msg.get("role") == "user")
    assistant_message = next(msg for msg in messages if msg.get("role") == "assistant")
    return user_message["content"], assistant_message["content"]


def get_client(base_url: str, api_key: str) -> OpenAI:
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAI(api_key=api_key, base_url=base_url)
        THREAD_LOCAL.client = client
    return client


def process_one_sample(
    sample_idx: int,
    sample: dict,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float | None,
    max_images: int | None,
    enable_thinking: bool,
) -> dict:
    prompt_text, label = extract_prompt_and_label(sample)
    images = sample.get("images", [])
    request_images = images if max_images is None else images[:max_images]
    content = build_multimodal_content(prompt_text, request_images, max_images)

    record = {
        "sample_idx": sample_idx,
        "bbox_docvqa_id": sample.get("bbox_docvqa_id"),
        "num_images": len(images),
        "num_images_sent": len(request_images),
        "query": sample.get("query"),
        "label": label,
    }

    try:
        extra_body = None
        if enable_thinking:
            extra_body = {
                "stop": [],
                "enable_thinking": True,
            }

        client = get_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
        )
        record["predict"] = response.choices[0].message.content
        record["error"] = None
    except Exception as exc:
        record["predict"] = None
        record["error"] = str(exc)

    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test BBox-DocVQA data through OpenAI-style API.")
    parser.add_argument("--dataset-jsonl", required=True, help="Path to the dataset jsonl file.")
    parser.add_argument("--output", required=True, help="Path to the output jsonl file.")
    parser.add_argument("--model", default="qwen3.5-397b-a17b", help="Model name exposed by the API.")
    parser.add_argument(
        "--base-url",
        default="https://qianfan.baidubce.com/v2",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="QIANFAN_API_KEY",
        help="Environment variable name that stores the API key.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="0-based start sample index.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to send.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for images per prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional top-p sampling value.")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=true in extra_body for Qianfan-compatible models.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output jsonl: skip successful records and retry failed ones.",
    )
    parser.add_argument("--max-workers", type=int, default=8, help="Number of worker threads.")
    return parser.parse_args()


def load_existing_records(output_path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not output_path.exists():
        return records

    with output_path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_idx = record.get("sample_idx")
            if isinstance(sample_idx, int):
                records[sample_idx] = record

    return records


def rewrite_existing_records(output_path: Path, records: dict[int, dict]) -> None:
    with output_path.open("w", encoding="utf-8", buffering=1) as writer:
        for sample_idx in sorted(records):
            writer.write(json.dumps(records[sample_idx], ensure_ascii=False) + "\n")
            writer.flush()


def main() -> None:
    args = parse_args()
    api_key = os.environ[args.api_key_env]

    dataset_path = Path(args.dataset_jsonl)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_records: dict[int, dict] = {}
    if args.resume:
        existing_records = load_existing_records(output_path)
        if existing_records:
            rewrite_existing_records(output_path, existing_records)

    with dataset_path.open("r", encoding="utf-8") as reader:
        rows = [json.loads(line) for line in reader]

    end_index = len(rows)
    if args.max_samples is not None:
        end_index = min(end_index, args.start_index + args.max_samples)
    results_by_idx = dict(existing_records)
    pending_indices: list[int] = []
    for sample_idx in range(args.start_index, end_index):
        existing_record = existing_records.get(sample_idx)
        if existing_record is not None and existing_record.get("error") is None:
            print(
                f"[{sample_idx}] bbox_docvqa_id={existing_record.get('bbox_docvqa_id')} "
                f"resume_skip=True"
            )
            continue
        pending_indices.append(sample_idx)

    total_pending = len(pending_indices)
    if total_pending:
        print(f"Processing {total_pending} sample(s) with {args.max_workers} threads")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_one_sample,
                sample_idx,
                rows[sample_idx],
                args.model,
                args.base_url,
                api_key,
                args.temperature,
                args.top_p,
                args.max_images,
                args.enable_thinking,
            ): sample_idx
            for sample_idx in pending_indices
        }

        finished = 0
        for future in as_completed(future_to_idx):
            sample_idx = future_to_idx[future]
            try:
                record = future.result()
            except Exception as exc:
                sample = rows[sample_idx]
                images = sample.get("images", [])
                request_images = images if args.max_images is None else images[: args.max_images]
                record = {
                    "sample_idx": sample_idx,
                    "bbox_docvqa_id": sample.get("bbox_docvqa_id"),
                    "num_images": len(images),
                    "num_images_sent": len(request_images),
                    "query": sample.get("query"),
                    "label": None,
                    "predict": None,
                    "error": str(exc),
                }

            results_by_idx[sample_idx] = record
            finished += 1
            print(
                f"[{sample_idx}] bbox_docvqa_id={record['bbox_docvqa_id']} "
                f"images={record['num_images_sent']}/{record['num_images']} "
                f"error={record['error'] is not None} "
                f"progress={finished}/{total_pending}"
            )

    with output_path.open("w", encoding="utf-8", buffering=1) as writer:
        for sample_idx in sorted(results_by_idx):
            writer.write(json.dumps(results_by_idx[sample_idx], ensure_ascii=False) + "\n")
            writer.flush()


if __name__ == "__main__":
    main()
