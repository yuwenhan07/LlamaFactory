#!/usr/bin/env python3

import argparse
import ast
import json
import re
from pathlib import Path

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def strip_code_fence(text):
    if text is None:
        return ""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_numeric_box(box):
    if isinstance(box, dict):
        for key in ("bbox_2d", "bbox", "box", "coordinates"):
            if key in box:
                return parse_numeric_box(box[key])
        return None

    if isinstance(box, (list, tuple)) and len(box) == 4:
        values = []
        for item in box:
            if isinstance(item, (int, float)):
                values.append(float(item))
            elif isinstance(item, str):
                match = NUMBER_RE.fullmatch(item.strip())
                if not match:
                    return None
                values.append(float(match.group(0)))
            else:
                return None
        return values

    if isinstance(box, str):
        numbers = NUMBER_RE.findall(box)
        if len(numbers) == 4:
            return [float(num) for num in numbers]

    return None


def is_box_list(value):
    return isinstance(value, list) and all(parse_numeric_box(item) is not None for item in value)


def is_page_list(value):
    return isinstance(value, list) and all(is_box_list(item) for item in value)


def split_flat_number_sequence(value):
    if not isinstance(value, (list, tuple)):
        return None

    numbers = []
    for item in value:
        if isinstance(item, (int, float)):
            numbers.append(float(item))
        elif isinstance(item, str):
            match = NUMBER_RE.fullmatch(item.strip())
            if not match:
                return None
            numbers.append(float(match.group(0)))
        else:
            return None

    if not numbers or len(numbers) % 4 != 0:
        return None

    return [numbers[idx : idx + 4] for idx in range(0, len(numbers), 4)]


def normalize_pages_from_nested_lists(data):
    if is_page_list(data):
        return [[parse_numeric_box(box) for box in page] for page in data]
    if is_box_list(data):
        return [[parse_numeric_box(box) for box in data]]

    if isinstance(data, (list, tuple)):
        flat_boxes = split_flat_number_sequence(data)
        if flat_boxes is not None:
            return [flat_boxes]

    box = parse_numeric_box(data)
    if box is not None:
        return [[box]]

    return None


def extract_json_span(text):
    first_positions = [idx for idx in (text.find("["), text.find("{")) if idx != -1]
    start = min(first_positions) if first_positions else -1
    end = max(text.rfind("]"), text.rfind("}"))
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def parse_predicted_pages(text):
    cleaned = strip_code_fence(text)
    candidates = [cleaned]
    span = extract_json_span(cleaned)
    if span and span != cleaned:
        candidates.append(span)

    for candidate in candidates:
        if not candidate:
            continue
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
            except Exception:
                continue
            normalized = normalize_pages_from_nested_lists(parsed)
            if normalized is not None:
                return normalized

    matches = re.findall(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        cleaned,
    )
    if matches:
        return [[[float(value) for value in match] for match in matches]]

    return None


def expected_split_count(record):
    pages = record.get("evidence_page", []) or []
    rel_bbox = record.get("rel_bbox", []) or []

    if len(pages) > 1:
        return sum(1 for page_boxes in rel_bbox if page_boxes)
    if len(pages) == 1:
        return len(rel_bbox[0]) if rel_bbox else 0
    return 0


def combine_record_predictions(source_record, split_predictions):
    pages = source_record.get("evidence_page", []) or []
    rel_bbox = source_record.get("rel_bbox", []) or []
    num_pages = len(pages)
    page_predictions = [[] for _ in range(num_pages)]
    errors = []

    if num_pages > 1:
        split_idx = 0
        for page_idx, page_boxes in enumerate(rel_bbox):
            if not page_boxes:
                continue
            prediction_record = split_predictions[split_idx]
            parsed_pages = parse_predicted_pages(prediction_record.get("predict", ""))
            if parsed_pages is None or not parsed_pages or not parsed_pages[0]:
                errors.append(f"page {page_idx} parse_failed")
            else:
                page_predictions[page_idx] = parsed_pages[0]
            split_idx += 1
    elif num_pages == 1:
        combined_boxes = []
        for split_idx, prediction_record in enumerate(split_predictions, start=1):
            parsed_pages = parse_predicted_pages(prediction_record.get("predict", ""))
            if parsed_pages is None or not parsed_pages or not parsed_pages[0]:
                errors.append(f"box {split_idx} parse_failed")
                continue
            combined_boxes.extend(parsed_pages[0])
        page_predictions[0] = combined_boxes
    else:
        errors.append("missing_evidence_page")

    if errors:
        return None, "; ".join(errors)

    return json.dumps(page_predictions, ensure_ascii=False, separators=(",", ":")), None


def combine_predictions(source_jsonl, prediction_jsonl, output_jsonl):
    with source_jsonl.open("r", encoding="utf-8") as source_file:
        source_records = [json.loads(line) for line in source_file]
    with prediction_jsonl.open("r", encoding="utf-8") as prediction_file:
        prediction_records = [json.loads(line) for line in prediction_file]

    expected_total = sum(expected_split_count(record) for record in source_records)
    if expected_total != len(prediction_records):
        raise ValueError(
            f"Split prediction count mismatch: expected {expected_total} from source records, "
            f"but found {len(prediction_records)} prediction lines."
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    prediction_offset = 0

    with output_jsonl.open("w", encoding="utf-8") as output_file:
        for sample_idx, source_record in enumerate(source_records):
            split_count = expected_split_count(source_record)
            split_slice = prediction_records[prediction_offset : prediction_offset + split_count]
            prediction_offset += split_count

            combined_predict, error = combine_record_predictions(source_record, split_slice)
            output_record = {
                "sample_idx": sample_idx,
                "bbox_docvqa_id": source_record.get("bbox_docvqa_id", sample_idx + 1),
                "num_images": len(source_record.get("images", [])),
                "num_images_sent": len(source_record.get("images", [])),
                "query": source_record.get("query"),
                "label": source_record.get("bbox_target")
                or source_record.get("messages", [{}, {"content": None}])[1].get("content"),
                "predict": combined_predict,
                "error": error,
            }
            output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Combine split bbox predictions back into original BBox-DocVQA format.")
    parser.add_argument(
        "--prediction-jsonl",
        default="/home/work/workspace/ywh/LlamaFactory/split_and_combine/pred_bbox_sft_output/qwen3-vl-8b-sft-0313/generated_predictions.jsonl",
        help="Split prediction jsonl produced from the single-image bbox dataset.",
    )
    parser.add_argument(
        "--source-jsonl",
        default="../data/bbox_docvqa_bbox_out_1000/bbox_docvqa_bbox_1000.jsonl",
        help="Original unsplit bbox dataset jsonl.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/home/work/workspace/ywh/LlamaFactory/api/output/pred_bbox/pred_bbox__qwen3-vl-8b-sft-0313.jsonl",
        help="Path to write the merged prediction jsonl.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    combine_predictions(
        source_jsonl=Path(args.source_jsonl).resolve(),
        prediction_jsonl=Path(args.prediction_jsonl).resolve(),
        output_jsonl=Path(args.output_jsonl).resolve(),
    )
