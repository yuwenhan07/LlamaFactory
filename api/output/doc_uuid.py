from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate doc result jsonl files by bbox_docvqa_id."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "doc",
        help="Directory containing source jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "doc_clean",
        help="Directory to write cleaned jsonl files.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=1623,
        help="Expected number of unique bbox_docvqa_id values per complete file.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return rows


def has_prediction(row: dict[str, Any]) -> bool:
    prediction = row.get("predict")
    if prediction is None:
        return False
    if isinstance(prediction, str):
        return bool(prediction.strip())
    return True


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int, int]:
        index, row = item
        return (
            1 if row.get("error") is None else 0,
            1 if has_prediction(row) else 0,
            int(row.get("num_images_sent", -1)),
            index,
        )

    _, best_row = max(enumerate(rows), key=score)
    return best_row


def clean_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("bbox_docvqa_id")].append(row)

    cleaned: list[dict[str, Any]] = []
    null_error_kept = 0
    null_error_dropped = 0
    duplicate_groups = 0

    for bbox_docvqa_id in sorted(grouped):
        candidates = grouped[bbox_docvqa_id]
        if len(candidates) > 1:
            duplicate_groups += 1

        best = choose_best(candidates)
        if best.get("error") is None:
            null_error_kept += 1

        for row in candidates:
            if row is not best and row.get("error") is None:
                null_error_dropped += 1

        cleaned.append(dict(best))

    stats = {
        "input_rows": len(rows),
        "output_rows": len(cleaned),
        "duplicate_rows_removed": len(rows) - len(cleaned),
        "duplicate_groups": duplicate_groups,
        "null_error_kept": null_error_kept,
        "null_error_dropped": null_error_dropped,
    }
    return cleaned, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    files = sorted(input_dir.glob("*.jsonl"))

    if not files:
        raise FileNotFoundError(f"No jsonl files found in {input_dir}")

    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Expected unique id count: {args.expected_count}")

    for path in files:
        rows = load_jsonl(path)
        cleaned_rows, stats = clean_rows(rows)
        output_path = output_dir / path.name
        write_jsonl(output_path, cleaned_rows)

        status = "OK"
        if stats["output_rows"] != args.expected_count:
            status = "WARN"

        print(
            f"[{status}] {path.name}: "
            f"{stats['input_rows']} -> {stats['output_rows']} "
            f"(removed {stats['duplicate_rows_removed']}, groups {stats['duplicate_groups']}, "
            f"kept_null_error {stats['null_error_kept']})"
        )


if __name__ == "__main__":
    main()
