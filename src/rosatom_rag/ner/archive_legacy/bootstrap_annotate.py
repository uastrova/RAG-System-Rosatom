from pathlib import Path
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "ner" / "raw_samples" / "batch_001_candidates.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "labeled" / "batch_001_bootstrapped.jsonl"


PATTERNS = {
    "NORM_DOC": [
        r"\bГОСТ\s*Р\s*ИСО\s+\d[\d\.\-–—/]*\b",
        r"\bГОСТ\s*Р\s*МЭК\s+\d[\d\.\-–—/]*\b",
        r"\bГОСТ\s*IEC\s+\d[\d\.\-–—/]*\b",
        r"\bГОСТ\s*Р\s+\d[\d\.\-–—/]*\b",
        r"\bГОСТ(?:\s+[A-ZА-ЯЁ]+)?\s+\d[\d\.\-–—/]*\b",
        r"\bСП\s+\d[\d\.\-–—/]*\b",
        r"\bСанПиН\s+\d[\d\.\-–—/]*\b",
        r"\bСТО(?:\s+[A-ZА-ЯЁ\-]+)?\s+\d[\d\s\.\-–—/]*\b",
        r"\bТР\s+ТС\s+\d[\d\.\-–—/]*\b",
        r"\bНП-\d+-\d+\b",
        r"\bПУЭ\b",
    ],
    "NORM_REF": [
        r"\bп\.\s*\d+(?:\.\d+)*\b",
        r"\bпункт\s+\d+(?:\.\d+)*\b",
        r"\bгл\.\s*\d+(?:\.\d+)*\b",
        r"\bраздел\s+\d+(?:\.\d+)*\b",
        r"\bприложение\s+[А-ЯA-Z]\b",
        r"\bтаблица\s+\d+(?:\.\d+)*\b",
        r"\bТаблица\s+\d+(?:\.\d+)*\b",
        r"\bтабл\.\s*\d+\b",
        r"\bТабл\.\s*\d+\b",
        r"\bпоз\.\s*\d+\b",
        r"\bпоз\.\s*\d+\.\d+\b",
    ],
    "DATE": [
        r"\b\d{2}\.\d{2}\.\d{4}\b",
        r"\b\d{1,2}\s+[А-Яа-яЁё]+\s+\d{4}\s*г\.?\b",
    ],
    "DOC_NUM": [
        r"\bN\s*\d+(?:\s*[\/\-–—]\s*[A-Za-zА-Яа-яЁё0-9]+)*\b",
        r"\b№\s*\d+(?:\s*[\/\-–—]\s*[A-Za-zА-Яа-яЁё0-9]+)*\b",
    ],
}

MANUAL_ONLY_LABELS = ["ORG", "ADDRESS"]


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


def normalize_spaces(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_trash_doc_num(text: str) -> bool:
    text = normalize_spaces(text)
    trash_values = {
        "N 1", "N 2", "N 3", "N 4",
        "№ 1", "№ 2", "№ 3", "№ 4",
    }
    return text in trash_values


def deduplicate_entities(entities):
    seen = set()
    result = []

    for ent in entities:
        key = (ent["label"], ent["start"], ent["end"], ent["text"])
        if key in seen:
            continue
        seen.add(key)
        result.append(ent)

    return result


def find_entities_by_patterns(text: str):
    entities = []

    for label, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                entity_text = match.group(0)

                if label == "DOC_NUM" and is_trash_doc_num(entity_text):
                    continue

                entities.append({
                    "text": entity_text,
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                })

    entities = deduplicate_entities(entities)
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


def bootstrap_record(record):
    text = record["text"]
    entities = find_entities_by_patterns(text)
    entities = remove_overlaps(entities)

    bootstrapped = {
        "id": record["id"],
        "source_file": record["source_file"],
        "page_num": record["page_num"],
        "chunk_id": record["chunk_id"],
        "text": text,
        "entities": entities,
        "manual_labels_remaining": MANUAL_ONLY_LABELS,
        "needs_review": True,
        "auto_score": record.get("auto_score", 0),
    }
    return bootstrapped


def main():
    records = load_jsonl(INPUT_PATH)
    bootstrapped = [bootstrap_record(record) for record in records]
    save_jsonl(bootstrapped, OUTPUT_PATH)

    print("Исходных записей:", len(records))
    print("Сохранено в:", OUTPUT_PATH)
    print()

    for item in bootstrapped[:5]:
        print("=" * 80)
        print("id       :", item["id"])
        print("source   :", item["source_file"])
        print("page_num :", item["page_num"])
        print("entities :")
        for ent in item["entities"]:
            print(f"  - {ent['label']}: {ent['text']} [{ent['start']}, {ent['end']}]")
        print()


if __name__ == "__main__":
    main()