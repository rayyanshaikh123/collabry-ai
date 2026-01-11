import json
from pathlib import Path

# Your dataset locations inside the data folder
FILES = [
    "data/intent_training.jsonl",
    "data/intent_training_augmented.jsonl"
]

OUTPUT = "data/dataset.jsonl"

items = []

for file in FILES:
    path = Path(file)
    if not path.exists():
        print(f"Missing file: {file}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                print("Skipping corrupted line:", line)

# Remove duplicates by hashing JSON strings
unique = {json.dumps(x, sort_keys=True): x for x in items}
final = list(unique.values())

# Write merged dataset
with open(OUTPUT, "w", encoding="utf-8") as out:
    for item in final:
        out.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Merged {len(final)} items → {OUTPUT}")
