from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATHS = [
    PROJECT_ROOT / "data" / "ner" / "labeled" / "manual_batch_001_final.jsonl",
    PROJECT_ROOT / "data" / "ner" / "labeled" / "manual_batch_002_final.jsonl",
]

OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "labeled" / "manual_corpus_v2.jsonl"


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    all_records = []
    seen_ids = set()
    duplicate_ids = []

    for path in INPUT_PATHS:
        records = load_jsonl(path)
        print(f"{path.name}: {len(records)} records")

        for record in records:
            record_id = record["id"]
            if record_id in seen_ids:
                duplicate_ids.append(record_id)
                continue
            seen_ids.add(record_id)
            all_records.append(record)

    all_records.sort(key=lambda x: x["id"])
    save_jsonl(all_records, OUTPUT_PATH)

    print()
    print("Итоговый корпус:", len(all_records))
    print("Дубликатов пропущено:", len(duplicate_ids))
    print("Файл:", OUTPUT_PATH)

    if duplicate_ids:
        print("Примеры duplicate ids:")
        for item in duplicate_ids[:10]:
            print(" -", item)


if __name__ == "__main__":
    main()