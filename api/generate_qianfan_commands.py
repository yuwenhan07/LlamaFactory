#!/usr/bin/env python3

import argparse
import re
import shlex
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate expanded commands from bash.sh templates and qianfan.txt models."
    )
    parser.add_argument("--command-file", default="bash.sh", help="Template command file.")
    parser.add_argument("--model-file", default="qianfan.txt", help="Model config file.")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path to save generated commands. Prints to stdout when omitted.",
    )
    return parser.parse_args()


def parse_model_file(path: Path) -> tuple[str | None, list[str]]:
    base_url = None
    in_model_list = False
    models: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("base_url"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                base_url = parts[1].strip()
            continue
        if line == "model_list":
            in_model_list = True
            continue
        if in_model_list:
            models.append(line)

    if not models:
        raise ValueError(f"No models found in {path}")
    return base_url, models


def parse_command_file(path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    current_label = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_label = line[1:].strip() or "task"
            continue
        if line.startswith("python "):
            entries.append(
                {
                    "label": current_label or f"task_{len(entries) + 1}",
                    "command": line,
                }
            )
            current_label = None

    if not entries:
        raise ValueError(f"No runnable commands found in {path}")
    return entries


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "item"


def replace_arg(tokens: list[str], arg_name: str, value: str) -> list[str]:
    replaced = False
    result: list[str] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if token == arg_name:
            result.extend([arg_name, value])
            index += 2
            replaced = True
            continue
        if token.startswith(arg_name + "="):
            result.append(f"{arg_name}={value}")
            index += 1
            replaced = True
            continue
        result.append(token)
        index += 1

    if not replaced:
        result.extend([arg_name, value])
    return result


def read_arg_value(tokens: list[str], arg_name: str) -> str | None:
    for index, token in enumerate(tokens):
        if token == arg_name and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(arg_name + "="):
            return token.split("=", 1)[1]
    return None


def build_commands(command_entries: list[dict[str, str]], models: list[str], base_url: str | None) -> list[str]:
    lines: list[str] = []

    for command_entry in command_entries:
        for model in models:
            tokens = shlex.split(command_entry["command"])
            tokens = replace_arg(tokens, "--model", model)
            if base_url:
                tokens = replace_arg(tokens, "--base-url", base_url)

            output_value = read_arg_value(tokens, "--output")
            if output_value:
                output_path = Path(output_value)
                output_name = f"{output_path.stem}__{slugify(model)}{output_path.suffix or '.jsonl'}"
                tokens = replace_arg(tokens, "--output", output_name)

            lines.append(f"# {command_entry['label']} | {model}")
            lines.append(shlex.join(tokens))
            lines.append("")

    return lines


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    command_file = (script_dir / args.command_file).resolve()
    model_file = (script_dir / args.model_file).resolve()

    base_url, models = parse_model_file(model_file)
    command_entries = parse_command_file(command_file)
    lines = build_commands(command_entries, models, base_url)
    content = "\n".join(lines).rstrip() + "\n"

    if args.output_file:
        output_file = Path(args.output_file)
        if not output_file.is_absolute():
            output_file = script_dir / output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
