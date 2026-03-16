#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().split())


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def anls_score(prediction: Any, reference: Any) -> float:
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    dist = levenshtein_distance(pred, ref)
    max_len = max(len(pred), len(ref))
    if max_len == 0:
        return 1.0

    nl = dist / max_len
    if nl >= 0.5:
        return 0.0
    return 1.0 - nl


def exact_match(prediction: Any, reference: Any) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def contains_match(prediction: Any, reference: Any) -> float:
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not pred or not ref:
        return 0.0
    return 1.0 if ref in pred or pred in ref else 0.0


def compute_metrics(prediction_path: Path, scored_path: Path) -> dict[str, Any]:
    total = 0
    exact = 0.0
    contains = 0.0
    anls = 0.0

    scored_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("r", encoding="utf-8") as infile, scored_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            pred = row.get("predict", "")
            label = row.get("label", "")
            row["normalized_predict"] = normalize_text(pred)
            row["normalized_label"] = normalize_text(label)
            row["exact_match"] = exact_match(pred, label)
            row["contains_match"] = contains_match(pred, label)
            row["anls"] = anls_score(pred, label)
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

            total += 1
            exact += row["exact_match"]
            contains += row["contains_match"]
            anls += row["anls"]

    if total == 0:
        return {"samples": 0, "exact_match": 0.0, "contains_match": 0.0, "anls": 0.0}

    return {
        "samples": total,
        "exact_match": exact / total,
        "contains_match": contains / total,
        "anls": anls / total,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False, indent=2)
        writer.write("\n")


def collect_inputs(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.glob(pattern) if path.is_file())


def default_name_for_file(path: Path) -> str:
    if path.parent.name and path.parent.name not in (".", ""):
        if path.name == "generated_predictions.jsonl":
            return path.parent.name
    return path.stem


def evaluate_one_file(
    input_file: Path,
    scored_output: Path,
    metrics_output: Path,
    name: str | None,
    backend: str | None,
    model_name_or_path: str | None,
    template: str | None,
) -> dict[str, Any]:
    metrics = compute_metrics(input_file, scored_output)
    metrics.update(
        {
            "name": name or default_name_for_file(input_file),
            "backend": backend,
            "model_name_or_path": model_name_or_path,
            "template": template,
        }
    )
    write_json(metrics_output, metrics)
    return metrics


def write_summary(metrics_list: list[dict[str, Any]], output_root: Path) -> None:
    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"
    write_json(summary_json, metrics_list)

    fieldnames = [
        "name",
        "backend",
        "model_name_or_path",
        "template",
        "samples",
        "exact_match",
        "contains_match",
        "anls",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in metrics_list:
            csv_writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute exact match / contains match / ANLS for prediction jsonl.")
    parser.add_argument("input", help="Input jsonl file or a directory of jsonl files.")
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern used when input is a directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for scored outputs and metrics. Defaults to the input file parent or the input directory.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional run name for single-file evaluation.",
    )
    parser.add_argument("--backend", default=None, help="Optional backend metadata.")
    parser.add_argument("--model-name-or-path", default=None, help="Optional model path metadata.")
    parser.add_argument("--template", default=None, help="Optional template metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    input_files = collect_inputs(input_path, args.pattern)
    if not input_files:
        raise FileNotFoundError(f"No files matched: {input_path} ({args.pattern})")

    if input_path.is_file():
        output_root = Path(args.output_dir).resolve() if args.output_dir else input_path.parent
        scored_output = output_root / f"{input_path.stem}_scored.jsonl"
        metrics_output = output_root / f"{input_path.stem}_metrics.json"
        metrics = evaluate_one_file(
            input_file=input_path,
            scored_output=scored_output,
            metrics_output=metrics_output,
            name=args.name,
            backend=args.backend,
            model_name_or_path=args.model_name_or_path,
            template=args.template,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"scored_predictions={scored_output}")
        print(f"metrics_json={metrics_output}")
        return

    output_root = Path(args.output_dir).resolve() if args.output_dir else input_path
    output_root.mkdir(parents=True, exist_ok=True)

    metrics_list: list[dict[str, Any]] = []
    for input_file in input_files:
        rel_name = input_file.stem
        scored_output = output_root / f"{rel_name}_scored.jsonl"
        metrics_output = output_root / f"{rel_name}_metrics.json"
        metrics = evaluate_one_file(
            input_file=input_file,
            scored_output=scored_output,
            metrics_output=metrics_output,
            name=rel_name,
            backend=args.backend,
            model_name_or_path=args.model_name_or_path,
            template=args.template,
        )
        metrics_list.append(metrics)
        print(f"[done] {input_file.name} -> anls={metrics['anls']:.6f}")

    metrics_list.sort(key=lambda item: (-item["anls"], -item["contains_match"], -item["exact_match"], item["name"]))
    write_summary(metrics_list, output_root)
    print(f"summary_json={output_root / 'summary.json'}")
    print(f"summary_csv={output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
