from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = PROJECT_ROOT / "data" / "ner" / "labeled" / "batch_001_cleaned.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "labeled" / "manual_batch_001.json"

LIMIT = 60


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    records = load_jsonl(INPUT_PATH)
    selected = records[:LIMIT]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print("Исходных записей:", len(records))
    print("Экспортировано в manual batch:", len(selected))
    print("Файл:", OUTPUT_PATH)


if __name__ == "__main__":
    main()