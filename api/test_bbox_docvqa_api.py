#!/usr/bin/env python3

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path

from openai import OpenAI
from transformers.utils.versions import require_version


require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = OpenAI(
        api_key=os.environ[args.api_key_env],
        base_url=args.base_url,
    )

    dataset_path = Path(args.dataset_jsonl)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("r", encoding="utf-8") as reader:
        rows = [json.loads(line) for line in reader]

    end_index = len(rows)
    if args.max_samples is not None:
        end_index = min(end_index, args.start_index + args.max_samples)

    with output_path.open("w", encoding="utf-8") as writer:
        for sample_idx in range(args.start_index, end_index):
            sample = rows[sample_idx]
            prompt_text, label = extract_prompt_and_label(sample)
            images = sample.get("images", [])
            request_images = images if args.max_images is None else images[: args.max_images]
            content = build_multimodal_content(prompt_text, request_images, args.max_images)

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
                if args.enable_thinking:
                    extra_body = {
                        "stop": [],
                        "enable_thinking": True,
                    }

                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    extra_body=extra_body,
                )
                record["predict"] = response.choices[0].message.content
                record["error"] = None
            except Exception as exc:
                record["predict"] = None
                record["error"] = str(exc)

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{sample_idx}] bbox_docvqa_id={record['bbox_docvqa_id']} "
                f"images={record['num_images_sent']}/{record['num_images']} "
                f"error={record['error'] is not None}"
            )


if __name__ == "__main__":
    main()
