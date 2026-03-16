#!/usr/bin/env bash

python - "$@" <<'PY'
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-run commands from bash.sh with models from qianfan.txt."
    )
    parser.add_argument("--command-file", default="bash.sh", help="Template command file.")
    parser.add_argument("--model-file", default="qianfan.txt", help="Model config file.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Directory used to store ordered logs and manifest. Defaults to api/batch_runs/<timestamp>.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum worker threads when more than one task exists.",
    )
    return parser.parse_args()


def parse_model_file(path: Path):
    base_url = None
    in_model_list = False
    models = []

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


def parse_command_file(path: Path):
    entries = []
    current_label = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_label = line[1:].strip() or "task"
            continue
        if line.startswith("python "):
            entries.append({
                "label": current_label or f"task_{len(entries) + 1}",
                "command": line,
            })
            current_label = None

    if not entries:
        raise ValueError(f"No runnable commands found in {path}")
    return entries


def slugify(text: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "item"


def replace_arg(tokens, arg_name, value):
    replaced = False
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == arg_name:
            result.extend([arg_name, value])
            i += 2
            replaced = True
            continue
        if token.startswith(arg_name + "="):
            result.append(f"{arg_name}={value}")
            i += 1
            replaced = True
            continue
        result.append(token)
        i += 1

    if not replaced:
        result.extend([arg_name, value])
    return result


def read_arg_value(tokens, arg_name):
    for index, token in enumerate(tokens):
        if token == arg_name and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(arg_name + "="):
            return token.split("=", 1)[1]
    return None


def build_tasks(command_entries, models, base_url):
    tasks = []
    task_index = 0

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
            else:
                output_name = f"{slugify(command_entry['label'])}__{slugify(model)}.jsonl"
                tokens = replace_arg(tokens, "--output", output_name)

            tasks.append({
                "index": task_index,
                "label": command_entry["label"],
                "model": model,
                "tokens": tokens,
                "output_file": output_name,
            })
            task_index += 1

    return tasks


def run_task(task, workdir: Path, log_dir: Path):
    prefix = f"{task['index'] + 1:03d}__{slugify(task['label'])}__{slugify(task['model'])}"
    stdout_path = log_dir / f"{prefix}.stdout.log"
    stderr_path = log_dir / f"{prefix}.stderr.log"

    completed = subprocess.run(
        task["tokens"],
        cwd=workdir,
        text=True,
        capture_output=True,
    )

    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    result = dict(task)
    result.update(
        {
            "returncode": completed.returncode,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "command": shlex.join(task["tokens"]),
        }
    )
    return result


def main():
    args = parse_args()
    script_dir = Path.cwd()

    command_file = (script_dir / args.command_file).resolve()
    model_file = (script_dir / args.model_file).resolve()

    base_url, models = parse_model_file(model_file)
    command_entries = parse_command_file(command_file)
    tasks = build_tasks(command_entries, models, base_url)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    default_run_dir = script_dir / "batch_runs" / timestamp
    run_dir = Path(args.run_dir).resolve() if args.run_dir else default_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    max_workers = 1 if len(tasks) <= 1 else max(1, min(args.max_workers, len(tasks)))
    results = [None] * len(tasks)

    if max_workers == 1:
        for task in tasks:
            results[task["index"]] = run_task(task, script_dir, run_dir)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(run_task, task, script_dir, run_dir): task["index"] for task in tasks
            }
            for future in as_completed(future_map):
                index = future_map[future]
                results[index] = future.result()

    manifest_path = run_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as writer:
        for result in results:
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"run_dir={run_dir}")
    print(f"task_count={len(tasks)}")
    print(f"workers={max_workers}")
    print(f"manifest={manifest_path}")

    failed = [result for result in results if result["returncode"] != 0]
    if failed:
        print(f"failed={len(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
PY
