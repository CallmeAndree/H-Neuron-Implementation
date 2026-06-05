import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "pandas is required to write parquet files. Install it with: pip install pandas pyarrow"
    ) from exc


REQUIRED_FIELDS = {"question", "responses", "judges", "ground_truth"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine three one-qid-per-line JSONL response files into one parquet file. "
            "Each JSONL line must look like: {\"qid\": {\"question\": ..., "
            "\"responses\": [...], \"judges\": [...], \"ground_truth\": [...]}}."
        )
    )
    parser.add_argument(
        "--input_paths",
        nargs=3,
        required=True,
        help="Exactly three JSONL files to combine, in output order.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to write the combined parquet file.",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help=(
            "Write flat parquet columns qid/question/responses/judges/ground_truth. "
            "By default, the parquet has qid and sample columns, where sample is a dict."
        ),
    )
    parser.add_argument(
        "--allow_duplicates",
        action="store_true",
        help="Allow duplicate qids across input files. By default duplicates fail fast.",
    )
    return parser.parse_args()


def get_single_sample(row: Dict[str, Any], source: Path, line_num: int) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(row, dict):
        raise ValueError(f"{source} line {line_num}: row is not a JSON object")
    if len(row) != 1:
        raise ValueError(
            f"{source} line {line_num}: expected exactly one top-level qid, got {len(row)}"
        )

    qid = next(iter(row))
    content = row[qid]
    if not isinstance(qid, str) or not qid:
        raise ValueError(f"{source} line {line_num}: qid must be a non-empty string")
    if not isinstance(content, dict):
        raise ValueError(f"{source} line {line_num}, qid {qid}: sample content is not an object")
    return qid, content


def validate_sample(qid: str, content: Dict[str, Any], source: Path, line_num: int) -> None:
    missing = sorted(REQUIRED_FIELDS - set(content))
    if missing:
        raise ValueError(f"{source} line {line_num}, qid {qid}: missing fields {missing}")

    if not isinstance(content["question"], str) or not content["question"].strip():
        raise ValueError(f"{source} line {line_num}, qid {qid}: question must be a non-empty string")

    for field in ["responses", "judges", "ground_truth"]:
        if not isinstance(content[field], list):
            raise ValueError(f"{source} line {line_num}, qid {qid}: {field} must be a list")

    if not content["responses"]:
        raise ValueError(f"{source} line {line_num}, qid {qid}: responses list is empty")
    if not content["ground_truth"]:
        raise ValueError(f"{source} line {line_num}, qid {qid}: ground_truth list is empty")
    if len(content["responses"]) != len(content["judges"]):
        raise ValueError(
            f"{source} line {line_num}, qid {qid}: responses/judges length mismatch "
            f"({len(content['responses'])} != {len(content['judges'])})"
        )


def iter_jsonl_samples(input_path: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid, content = get_single_sample(row, input_path, line_num)
            validate_sample(qid, content, input_path, line_num)
            yield qid, content


def make_parquet_row(qid: str, content: Dict[str, Any], wide: bool) -> Dict[str, Any]:
    if wide:
        return {
            "qid": qid,
            "question": content["question"],
            "responses": content["responses"],
            "judges": content["judges"],
            "ground_truth": content["ground_truth"],
        }
    return {"qid": qid, "sample": content}


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.input_paths]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    seen_qids = set()

    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSONL file not found: {input_path}")
        for qid, content in iter_jsonl_samples(input_path):
            if not args.allow_duplicates and qid in seen_qids:
                raise ValueError(f"Duplicate qid found across inputs: {qid}")
            seen_qids.add(qid)
            rows.append(make_parquet_row(qid, content, args.wide))

    if not rows:
        raise ValueError("No JSONL rows were found in the input files")

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
