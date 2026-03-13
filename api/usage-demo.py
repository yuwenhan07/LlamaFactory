#!/usr/bin/env python3

import argparse
import base64
import mimetypes
import os
from pathlib import Path

from openai import OpenAI


def to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    image_bytes = Path(image_path).read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal multimodal example for an OpenAI-compatible API.")
    parser.add_argument("--image", required=True, help="Path to the local image file.")
    parser.add_argument(
        "--question",
        default="Describe this image and list the most important visible details.",
        help="Question sent together with the image.",
    )
    parser.add_argument(
        "--model",
        default="nvidia/nemotron-nano-12b-v2-vl:free",
        help="Multimodal model name exposed by the API.",
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible API base URL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    content = [
        {"type": "text", "text": args.question},
        {"type": "image_url", "image_url": {"url": to_data_url(args.image)}},
    ]

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": content}],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
