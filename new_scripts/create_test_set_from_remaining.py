import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_PATH = Path("data/small_subset/train_qwen_samples_consistent.jsonl")
TRAIN_PATH = Path("data/small_subset/train_qwen_samples_91_true_91_false.jsonl")
OUTPUT_PATH = Path("data/small_subset/test_qwen_samples.jsonl")
SEED = 42
TARGET_TOTAL = 200
TARGET_PER_CLASS = TARGET_TOTAL // 2


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_qid(row):
    return next(iter(row.keys()))


def get_label(row):
    content = next(iter(row.values()))
    judges = content.get("judges", [])
    if not judges or len(set(judges)) != 1:
        return None
    label = judges[0]
    if label not in {"true", "false"}:
        return None
    return label


def main():
    random.seed(SEED)

    train_qids = {get_qid(row) for row in read_jsonl(TRAIN_PATH)}

    remaining_by_label = defaultdict(list)
    skipped = 0

    for row in read_jsonl(SOURCE_PATH):
        qid = get_qid(row)
        if qid in train_qids:
            continue

        label = get_label(row)
        if label is None:
            skipped += 1
            continue

        remaining_by_label[label].append(row)

    selected = []
    for label in ["true", "false"]:
        rows = remaining_by_label[label]
        random.shuffle(rows)
        selected.extend(rows[:TARGET_PER_CLASS])

    if len(selected) < TARGET_TOTAL:
        selected_qids = {get_qid(row) for row in selected}
        leftovers = []
        for rows in remaining_by_label.values():
            leftovers.extend(row for row in rows if get_qid(row) not in selected_qids)
        random.shuffle(leftovers)
        selected.extend(leftovers[: TARGET_TOTAL - len(selected)])

    random.shuffle(selected)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(get_label(row) for row in selected)
    print(f"Train qids excluded: {len(train_qids)}")
    print(f"Remaining true: {len(remaining_by_label['true'])}")
    print(f"Remaining false: {len(remaining_by_label['false'])}")
    print(f"Skipped non-consistent/unknown remaining rows: {skipped}")
    print(f"Saved total: {len(selected)}")
    print(f"Saved true: {counts['true']}")
    print(f"Saved false: {counts['false']}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
