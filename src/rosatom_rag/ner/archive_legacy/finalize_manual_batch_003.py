from pathlib import Path
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "ner" / "labeled" / "manual_batch_003.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "labeled" / "manual_batch_003_final.jsonl"


BAD_DOC_NUM_PATTERNS = [
    re.compile(r"^N\s*[1-9]\d*$", re.IGNORECASE),
    re.compile(r"^№\s*[1-9]\d*$", re.IGNORECASE),
]

BAD_NORM_REF_PATTERNS = [
    re.compile(r"^приложение\s+\d+$", re.IGNORECASE),
]

BAD_ENTITY_TEXT_PATTERNS = [
    re.compile(r"^#+"),
    re.compile(r"^\d+$"),
    re.compile(r"^[\.\,\-\–—\:\;\(\)\[\]\s]+$"),
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def entity_is_bad(ent: dict) -> bool:
    text = normalize_spaces(ent["text"])
    label = ent["label"]

    if not text:
        return True

    for pattern in BAD_ENTITY_TEXT_PATTERNS:
        if pattern.match(text):
            return True

    if "##" in ent["text"]:
        return True

    if label == "DOC_NUM":
        for pattern in BAD_DOC_NUM_PATTERNS:
            if pattern.match(text):
                return True

    if label == "NORM_REF":
        for pattern in BAD_NORM_REF_PATTERNS:
            if pattern.match(text):
                return True

    return False


def remove_duplicates(entities):
    seen = set()
    result = []

    for ent in entities:
        key = (ent["label"], ent["start"], ent["end"], ent["text"])
        if key not in seen:
            seen.add(key)
            result.append(ent)

    return result


def sort_entities(entities):
    return sorted(entities, key=lambda x: (x["start"], x["end"], x["label"]))


def finalize_record(record):
    cleaned_entities = []
    removed_count = 0

    for ent in record.get("entities", []):
        ent = {
            "text": ent["text"],
            "label": ent["label"],
            "start": ent["start"],
            "end": ent["end"],
        }

        if entity_is_bad(ent):
            removed_count += 1
            continue

        cleaned_entities.append(ent)

    cleaned_entities = remove_duplicates(cleaned_entities)
    cleaned_entities = sort_entities(cleaned_entities)

    final_record = {
        "id": record["id"],
        "source_file": record["source_file"],
        "page_num": record["page_num"],
        "chunk_id": record["chunk_id"],
        "text": record["text"],
        "entities": cleaned_entities,
    }

    return final_record, removed_count


def main():
    records = load_json(INPUT_PATH)

    final_records = []
    total_removed = 0

    for record in records:
        final_record, removed_count = finalize_record(record)
        final_records.append(final_record)
        total_removed += removed_count

    save_jsonl(final_records, OUTPUT_PATH)

    print("Исходных записей:", len(records))
    print("Сохранено финальных записей:", len(final_records))
    print("Удалено некорректных entities:", total_removed)
    print("Файл:", OUTPUT_PATH)


if __name__ == "__main__":
    main()