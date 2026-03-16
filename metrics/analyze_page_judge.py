#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Any


GROUP_ORDER = ["SPSR", "SPMR", "MPMR", "ALL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze page judge outputs by SPSR / SPMR / MPMR / ALL."
    )
    parser.add_argument(
        "--dataset-jsonl",
        default="data/bbox_docvqa_page/bbox_docvqa_rel_page.jsonl",
        help="BBox-DocVQA page dataset jsonl.",
    )
    parser.add_argument(
        "--judge-dir",
        default="metrics/judge_output/page_judged",
        help="Directory containing judge result jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        default="metrics/judge_output/page_analysis",
        help="Directory used to store summary tables.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_sample(sample: dict[str, Any]) -> str:
    pages = len(sample.get("evidence_page", []))
    bbox = sample.get("bbox") or sample.get("rel_bbox") or []
    regions = sum(len(page) for page in bbox if isinstance(page, list))

    if pages == 1 and regions == 1:
        return "SPSR"
    if pages == 1 and regions > 1:
        return "SPMR"
    if pages > 1 and regions > 1:
        return "MPMR"
    raise ValueError(
        f"Unsupported page sample structure: bbox_docvqa_id={sample.get('bbox_docvqa_id')} "
        f"pages={pages}, regions={regions}"
    )


def build_group_map(dataset_path: Path) -> dict[int, str]:
    group_map: dict[int, str] = {}
    for sample in load_jsonl(dataset_path):
        sample_id = sample.get("bbox_docvqa_id")
        if not isinstance(sample_id, int):
            raise ValueError(f"Invalid bbox_docvqa_id: {sample_id!r}")
        group_map[sample_id] = classify_sample(sample)
    return group_map


def init_stats() -> dict[str, dict[str, float]]:
    return {
        group: {"total": 0.0, "correct": 0.0}
        for group in GROUP_ORDER
    }


def is_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() == "true"


def analyze_one_file(judge_file: Path, group_map: dict[int, str]) -> dict[str, Any]:
    stats = init_stats()

    for row in load_jsonl(judge_file):
        sample_id = row.get("id")
        if not isinstance(sample_id, int):
            continue
        group = group_map.get(sample_id)
        if group is None:
            continue

        correct = 1.0 if is_correct(row.get("judge")) else 0.0
        stats[group]["total"] += 1
        stats[group]["correct"] += correct
        stats["ALL"]["total"] += 1
        stats["ALL"]["correct"] += correct

    result: dict[str, Any] = {
        "model": judge_file.stem.removeprefix("page__"),
    }
    for group in GROUP_ORDER:
        total = int(stats[group]["total"])
        correct = int(stats[group]["correct"])
        ratio = (correct / total) if total else 0.0
        result[f"{group}_correct"] = correct
        result[f"{group}_total"] = total
        result[f"{group}_ratio"] = ratio

    return result


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = ["model"]
    for group in GROUP_ORDER:
        fieldnames.extend([f"{group}_correct", f"{group}_total", f"{group}_ratio"])

    with output_path.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow(row)


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    headers = ["model"] + GROUP_ORDER
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        values = [row["model"]]
        for group in GROUP_ORDER:
            values.append(
                f"{row[f'{group}_correct']}/{row[f'{group}_total']} "
                f"({row[f'{group}_ratio']:.2%})"
            )
        lines.append("| " + " | ".join(values) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_jsonl).resolve()
    judge_dir = Path(args.judge_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    group_map = build_group_map(dataset_path)
    judge_files = sorted(judge_dir.glob("*.jsonl"))
    if not judge_files:
        raise FileNotFoundError(f"No judge files found in {judge_dir}")

    rows = [analyze_one_file(path, group_map) for path in judge_files]
    rows.sort(key=lambda row: row["ALL_ratio"], reverse=True)

    csv_path = output_dir / "page_judge_summary.csv"
    md_path = output_dir / "page_judge_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f"csv={csv_path}")
    print(f"markdown={md_path}")
    print((md_path.read_text(encoding='utf-8')))


if __name__ == "__main__":
    main()
