from pathlib import Path
import json
import re
from rosatom_rag.config import PROJECT_ROOT , CHUNKS_DIR, EMB_MODEL_DIR

CHUNKS_PATH = CHUNKS_DIR / "ntd_chunks.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "labeled" / "manual_corpus.json"
EXPORT_LIMIT = 1000
MIN_TEXT_LEN = 80
PATTERNS = {
    "NORM_DOC": [
        r"\bГОСТ(?:\s+Р)?(?:\s+IEC)?\s+\d[\d\.\-–—]*\b",
        r"\bГОСТ\s+[A-ZА-ЯЁ]+\s+\d[\d\.\-–—]*\b",
        r"\bСП\s+\d[\d\.\-–—]*\b",
        r"\bСНиП\s+\d[\d\.\-–—]*\b",
        r"\bНП-\d+-\d+\b",
        r"\bРТМ\s+\d[\d\.\-–—]*\b",
        r"\bПУЭ\b",
    ],
    "NORM_REF": [
        r"\bп\.\s*\d+(?:\.\d+)*\b",
        r"\bпп\.\s*\d+(?:\.\d+)*(?:\s*,\s*\d+(?:\.\d+)*)*\b",
        r"\bгл\.\s*\d+(?:\.\d+)*\b",
        r"\bглава\s+\d+(?:\.\d+)*\b",
        r"\bраздел\s+\d+(?:\.\d+)*\b",
        r"\bтабл\.\s*\d+(?:\.\d+)*\b",
        r"\bтаблица\s+\d+(?:\.\d+)*\b",
        r"\bприложение\s+[А-ЯA-Z0-9IVXLC]+\b",
    ],
    "DATE": [
        r"\b\d{2}\.\d{2}\.\d{4}\b",
        r"\b\d{1,2}\s+[А-Яа-яЁё]+\s+\d{4}\s*г\.?\b",
    ],
}


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def make_record_id(record: dict) -> str:
    return record.get("id") or record.get("chunk_id")


def find_entities_by_patterns(text: str):
    entities = []
    for label, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                entities.append({
                    "text": match.group(0),
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                })
    return entities


def remove_overlaps(entities):
    entities = sorted(
        entities,
        key=lambda x: (x["start"], -(x["end"] - x["start"]))
    )
    filtered = []
    occupied = []
    
    for ent in entities:
        s, e = ent["start"], ent["end"]

        has_overlap = False
        for os, oe in occupied:
            if not (e <= os or s >= oe):
                has_overlap = True
                break

        if not has_overlap:
            filtered.append(ent)
            occupied.append((s, e))

    filtered.sort(key=lambda x: (x["start"], x["end"]))
    return filtered


def score_record(text: str, entities: list[dict]) -> int:
    score = 0

    label_counts = {}
    for ent in entities:
        label_counts[ent["label"]] = label_counts.get(ent["label"], 0) + 1

    score += len(entities) * 2
    score += label_counts.get("NORM_DOC", 0) * 3
    score += label_counts.get("NORM_REF", 0) * 2
    score += label_counts.get("DATE", 0)

    if "ГОСТ" in text:
        score += 3
    if "СП" in text:
        score += 2
    if "ПУЭ" in text:
        score += 2
    if "табл." in text or "Таблица" in text or "таблица" in text:
        score += 1

    return score


def export_record(record: dict, entities: list[dict], auto_score: int):
    record_id = make_record_id(record)

    return {
        "id": record_id,
        "source_file": record["source_file"],
        "page_num": record["page_num"],
        "chunk_id": record["chunk_id"],
        "text": record["text"],
        "entities": entities,
        "auto_score": auto_score,
    }


def main():
    chunks = load_jsonl(CHUNKS_PATH)
    existing_records = load_json(OUTPUT_PATH)
    existing_ids = {make_record_id(r) for r in existing_records}

    candidates = []

    total_scanned = 0
    skipped_existing = 0
    skipped_short = 0
    with_entities = 0

    for record in chunks:
        total_scanned += 1

        record_id = make_record_id(record)
        if not record_id:
            continue

        if record_id in existing_ids:
            skipped_existing += 1
            continue

        text = record.get("text", "")
        if len(text.strip()) < MIN_TEXT_LEN:
            skipped_short += 1
            continue

        entities = find_entities_by_patterns(text)
        entities = remove_overlaps(entities)

        if not entities:
            continue

        with_entities += 1
        auto_score = score_record(text, entities)
        candidates.append(export_record(record, entities, auto_score))

    candidates.sort(
        key=lambda x: (
            -x["auto_score"],
            x["source_file"],
            x["page_num"],
            x["chunk_id"],
        )
    )

    selected = candidates[:EXPORT_LIMIT]
    combined = existing_records + selected
    save_json(combined, OUTPUT_PATH)

    print("Всего чанков просмотрено:", total_scanned)
    print("Уже были в manual_corpus:", skipped_existing)
    print("Пропущено коротких:", skipped_short)
    print("Кандидатов с сущностями:", with_entities)
    print("Добавлено новых записей:", len(selected))
    print("Итого записей в manual_corpus:", len(combined))
    print("Файл:", OUTPUT_PATH)
    print()

    for item in selected[:10]:
        print("=" * 80)
        print("id       :", item["id"])
        print("source   :", item["source_file"])
        print("page_num :", item["page_num"])
        print("auto_score:", item["auto_score"])
        print("entities :")
        for ent in item["entities"]:
            print(f"  - {ent['label']}: {ent['text']} [{ent['start']}, {ent['end']}]")
        print()


if __name__ == "__main__":
    main()