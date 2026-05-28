from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = PROJECT_ROOT / "data" / "ner" / "labeled" / "manual_batch_001.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "labeled" / "manual_batch_001_final.jsonl"

ALLOWED_LABELS = {"NORM_DOC", "NORM_REF", "DATE", "DOC_NUM"}


def normalize_entity(ent: dict, text: str):
    label = ent["label"]
    start = ent["start"]
    end = ent["end"]
    ent_text = ent["text"]

    if label not in ALLOWED_LABELS:
        return None

    if not isinstance(start, int) or not isinstance(end, int):
        return None

    if start < 0 or end > len(text) or start >= end:
        return None

    actual = text[start:end]
    if actual != ent_text:
        return None

    return {
        "text": ent_text,
        "label": label,
        "start": start,
        "end": end,
    }


def deduplicate_entities(entities: list[dict]):
    seen = set()
    result = []

    for ent in entities:
        key = (ent["text"], ent["label"], ent["start"], ent["end"])
        if key in seen:
            continue
        seen.add(key)
        result.append(ent)

    result.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return result


def main():
    records = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    final_records = []
    dropped_entities = 0

    for record in records:
        text = record["text"]
        cleaned_entities = []

        for ent in record.get("entities", []):
            normalized = normalize_entity(ent, text)
            if normalized is None:
                dropped_entities += 1
                continue
            cleaned_entities.append(normalized)

        cleaned_entities = deduplicate_entities(cleaned_entities)

        final_records.append({
            "id": record["id"],
            "source_file": record["source_file"],
            "page_num": record["page_num"],
            "chunk_id": record["chunk_id"],
            "text": record["text"],
            "entities": cleaned_entities,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in final_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Исходных записей:", len(records))
    print("Сохранено финальных записей:", len(final_records))
    print("Удалено некорректных entities:", dropped_entities)
    print("Файл:", OUTPUT_PATH)


if __name__ == "__main__":
    main()