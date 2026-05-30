import argparse
import json
import os
from typing import Any, Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a JSONL samples file and keep only rows whose judges are "
            "homogeneous: all 'true' or all 'false'."
        )
    )
    parser.add_argument("--input_path", type=str, required=True, help="Input JSONL samples file.")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL file for consistent samples.")
    return parser.parse_args()


def get_single_sample(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if len(row) != 1:
        raise ValueError("Expected each JSONL row to contain exactly one top-level sample ID.")

    qid = next(iter(row))
    content = row[qid]

    if not isinstance(content, dict):
        raise ValueError("Expected sample content to be a JSON object.")

    return qid, content


def judge_label(content: Dict[str, Any]) -> str:
    judges = content.get("judges")

    if not isinstance(judges, list) or not judges:
        raise ValueError("Missing or empty judges list.")

    normalized = [str(j).strip().lower() for j in judges]
    labels = set(normalized)

    if labels == {"true"}:
        return "true"
    if labels == {"false"}:
        return "false"

    return "mixed"


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    total_rows = 0
    kept_true = 0
    kept_false = 0
    dropped_mixed = 0
    dropped_invalid = 0

    with open(args.input_path, "r", encoding="utf-8") as f_in, open(
        args.output_path, "w", encoding="utf-8"
    ) as f_out:
        for line_num, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            total_rows += 1

            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not a JSON object.")

                _, content = get_single_sample(row)
                label = judge_label(content)
            except Exception as exc:
                dropped_invalid += 1
                print(f"Skipping invalid row {line_num}: {exc}")
                continue

            if label == "true":
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_true += 1
            elif label == "false":
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_false += 1
            else:
                dropped_mixed += 1

    print(f"Input rows: {total_rows}")
    print(f"Kept all-true rows: {kept_true}")
    print(f"Kept all-false rows: {kept_false}")
    print(f"Dropped mixed/other rows: {dropped_mixed}")
    print(f"Dropped invalid rows: {dropped_invalid}")
    print(f"Output rows: {kept_true + kept_false}")
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
