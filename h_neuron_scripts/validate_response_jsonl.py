import argparse
import json
from typing import Any, Dict, List, Tuple


REQUIRED_FIELDS = {
    "question": str,
    "responses": list,
    "judges": list,
    "ground_truth": list,
}
VALID_JUDGES = {"true", "false", "uncertain", "error"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that a response-collection JSONL file has the one-qid-per-line "
            "shape expected by downstream H-neuron scripts."
        )
    )
    parser.add_argument("--input_path", required=True, help="Path to response samples JSONL file.")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Maximum rows to validate; 0 validates the whole file.",
    )
    return parser.parse_args()


def get_single_sample(row: Dict[str, Any], line_num: int) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(row, dict):
        raise ValueError(f"line {line_num}: row is not a JSON object")
    if len(row) != 1:
        raise ValueError(f"line {line_num}: expected exactly one top-level qid, got {len(row)}")

    qid = next(iter(row))
    content = row[qid]
    if not isinstance(qid, str) or not qid:
        raise ValueError(f"line {line_num}: qid must be a non-empty string")
    if not isinstance(content, dict):
        raise ValueError(f"line {line_num}: sample content for qid {qid!r} is not an object")
    return qid, content


def validate_content(qid: str, content: Dict[str, Any], line_num: int) -> None:
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in content:
            raise ValueError(f"line {line_num}, qid {qid}: missing required field {field!r}")
        if not isinstance(content[field], expected_type):
            raise ValueError(
                f"line {line_num}, qid {qid}: field {field!r} must be "
                f"{expected_type.__name__}, got {type(content[field]).__name__}"
            )

    question = content["question"]
    responses = content["responses"]
    judges = content["judges"]
    ground_truth = content["ground_truth"]

    if not question.strip():
        raise ValueError(f"line {line_num}, qid {qid}: question is empty")
    if not responses:
        raise ValueError(f"line {line_num}, qid {qid}: responses list is empty")
    if len(responses) != len(judges):
        raise ValueError(
            f"line {line_num}, qid {qid}: responses/judges length mismatch "
            f"({len(responses)} != {len(judges)})"
        )
    if not ground_truth:
        raise ValueError(f"line {line_num}, qid {qid}: ground_truth list is empty")

    bad_response_types = [type(resp).__name__ for resp in responses if not isinstance(resp, str)]
    if bad_response_types:
        raise ValueError(f"line {line_num}, qid {qid}: every response must be a string")

    normalized_judges: List[str] = [str(judge).strip().lower() for judge in judges]
    invalid_judges = sorted(set(normalized_judges) - VALID_JUDGES)
    if invalid_judges:
        raise ValueError(
            f"line {line_num}, qid {qid}: invalid judge labels {invalid_judges}; "
            f"expected one of {sorted(VALID_JUDGES)}"
        )


def main() -> None:
    args = parse_args()
    checked = 0
    qids = set()

    with open(args.input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if args.max_rows and checked >= args.max_rows:
                break

            row = json.loads(line)
            qid, content = get_single_sample(row, line_num)
            if qid in qids:
                raise ValueError(f"line {line_num}: duplicate qid {qid!r}")
            validate_content(qid, content, line_num)
            qids.add(qid)
            checked += 1

    if checked == 0:
        raise ValueError("no JSONL rows were validated")

    print(f"Validated response JSONL contract for {checked} rows from {args.input_path}")


if __name__ == "__main__":
    main()
