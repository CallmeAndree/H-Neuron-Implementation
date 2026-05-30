import json
from collections import defaultdict
from pathlib import Path

input_path = Path("data/small_subset/test_answer_tokens_qwen.jsonl")
output_path = Path("data/small_subset/test_qwen_ids.json")

ids = defaultdict(list)
total = 0
skipped = 0

with input_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        qid = next(iter(row.keys()))
        content = row[qid]
        if "judge" in content:
            label = content["judge"]
        else:
            judges = content.get("judges", [])
            if not judges or len(set(judges)) != 1:
                skipped += 1
                continue
            label = judges[0]
        if label == "true":
            ids["t"].append(qid)
        elif label == "false":
            ids["f"].append(qid)
        else:
            skipped += 1
            continue
        total += 1

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump({"t": ids["t"], "f": ids["f"]}, f, ensure_ascii=False, indent=2)

print(f"Saved: {output_path}")
print(f"Total ids: {total}")
print(f"True ids: {len(ids['t'])}")
print(f"False ids: {len(ids['f'])}")
print(f"Skipped rows: {skipped}")
