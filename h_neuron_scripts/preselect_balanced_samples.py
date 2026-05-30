import argparse
import json
import os
import random
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preselect an equal number of all-true and all-false JSONL samples."
    )
    parser.add_argument("--input_path", type=str, required=True, help="Input JSONL samples file.")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL file.")
    parser.add_argument("--num_per_class", type=int, default=91, help="Number of all-true and all-false samples to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def get_label(row: Dict[str, Any]) -> str:
    if len(row) != 1:
        return "invalid"

    qid = next(iter(row))
    content = row.get(qid)
    if not isinstance(content, dict):
        return "invalid"

    judges = content.get("judges")
    if not isinstance(judges, list) or not judges:
        return "invalid"

    labels = {str(j).strip().lower() for j in judges}
    if labels == {"true"}:
        return "true"
    if labels == {"false"}:
        return "false"
    return "mixed"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    true_rows: List[Dict[str, Any]] = []
    false_rows: List[Dict[str, Any]] = []
    total_rows = 0
    mixed_rows = 0
    invalid_rows = 0

    with open(args.input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                print(f"Skipping invalid JSON at line {line_num}")
                continue

            label = get_label(row)
            if label == "true":
                true_rows.append(row)
            elif label == "false":
                false_rows.append(row)
            elif label == "mixed":
                mixed_rows += 1
            else:
                invalid_rows += 1

    available_per_class = min(len(true_rows), len(false_rows))
    selected_per_class = min(args.num_per_class, available_per_class)

    if selected_per_class < args.num_per_class:
        print(
            f"Warning: requested {args.num_per_class} per class, but only "
            f"{selected_per_class} per class can be selected."
        )

    selected_true = random.sample(true_rows, selected_per_class)
    selected_false = random.sample(false_rows, selected_per_class)
    selected_rows = selected_true + selected_false
    random.shuffle(selected_rows)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Input rows: {total_rows}")
    print(f"Available all-true rows: {len(true_rows)}")
    print(f"Available all-false rows: {len(false_rows)}")
    print(f"Dropped mixed rows: {mixed_rows}")
    print(f"Dropped invalid rows: {invalid_rows}")
    print(f"Selected all-true rows: {selected_per_class}")
    print(f"Selected all-false rows: {selected_per_class}")
    print(f"Output rows: {selected_per_class * 2}")
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
