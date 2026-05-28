from pathlib import Path
import json
import re
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks" / "ntd_chunks.jsonl"
SEEN_PATH = PROJECT_ROOT / "data" / "ner" / "labeled" / "manual_batch_001_final.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "labeled" / "manual_batch_002.json"

TARGET_COUNTS = {
    "pue": 30,
    "gost": 25,
    "sp": 25,
}

PATTERNS = {
    "NORM_DOC": [
        r"\bГОСТ(?:\s+[A-ZА-ЯЁ]+)?\s+\d[\d\.\-–—/]*\b",
        r"\bГОСТ\s+Р\s+\d[\d\.\-–—/]*\b",
        r"\bГОСТ\s+IEC\s+\d[\d\.\-–—/]*\b",
        r"\bСП\s+\d[\d\.\-–—/]*\b",
        r"\bСП\s+\d+-\d+\b",
        r"\bНП-\d+-\d+\b",
        r"\bПУЭ\b",
    ],
    "NORM_REF": [
        r"\bп\.\s*\d+(?:\.\d+)*\b",
        r"\bгл\.\s*\d+(?:\.\d+)*\b",
        r"\bраздел\s+\d+(?:\.\d+)*\b",
        r"\bприложение\s+[А-ЯA-Z0-9]+\b",
        r"\bтаблица\s+[А-ЯA-Z0-9\.\-]+\b",
        r"\bтабл\.\s*[А-ЯA-Z0-9\.\-]+\b",
    ],
    "DATE": [
        r"\b\d{2}\.\d{2}\.\d{4}\b",
        r"\b\d{1,2}\s+[А-Яа-яЁё]+\s+\d{4}\s*г\.?\b",
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


def load_seen_ids(path: Path):
    if not path.exists():
        return set()
    seen = set()
    for rec in load_jsonl(path):
        if rec.get("chunk_id"):
            seen.add(rec["chunk_id"])
        elif rec.get("id"):
            seen.add(rec["id"])
    return seen


def save_json(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def find_entities_by_patterns(text: str):
    entities = []

    for label, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(0)
                entities.append({
                    "text": value,
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                })

    return entities


def remove_overlaps(entities):
    entities = sorted(entities, key=lambda x: (x["start"], -(x["end"] - x["start"])))
    filtered = []

    for ent in entities:
        s, e = ent["start"], ent["end"]
        overlap = False
        for prev in filtered:
            ps, pe = prev["start"], prev["end"]
            if not (e <= ps or s >= pe):
                overlap = True
                break
        if not overlap:
            filtered.append(ent)

    filtered.sort(key=lambda x: (x["start"], x["end"]))
    return filtered


def detect_bucket(source_file: str):
    s = (source_file or "").lower()
    if "пуэ" in s:
        return "pue"
    if "гост" in s:
        return "gost"
    if "сп " in s or s.startswith("сп") or "/сп" in s:
        return "sp"
    return None


def score_record(rec):
    text = rec.get("text", "")
    source = rec.get("source_file", "")

    score = 0

    score += min(len(text) // 200, 10)

    if "ГОСТ" in text:
        score += 3
    if "СП " in text or text.startswith("СП"):
        score += 3
    if "ПУЭ" in text:
        score += 3

    if "должен" in text.lower():
        score += 3
    if "следует" in text.lower():
        score += 3
    if "требован" in text.lower():
        score += 3
    if "испытан" in text.lower():
        score += 2
    if "применяется" in text.lower():
        score += 2

    newline_count = text.count("\n")
    if newline_count < 25:
        score += 3
    elif newline_count < 50:
        score += 1

    if source.lower().startswith("(пуэ)") or "пуэ" in source.lower():
        score += 2

    return score


def make_bootstrapped_record(rec):
    text = rec["text"]
    entities = remove_overlaps(find_entities_by_patterns(text))

    return {
        "id": rec.get("chunk_id") or rec.get("id"),
        "source_file": rec.get("source_file"),
        "page_num": rec.get("page_num"),
        "chunk_id": rec.get("chunk_id"),
        "text": text,
        "entities": entities,
        "manual_labels_remaining": MANUAL_ONLY_LABELS,
        "needs_review": True,
        "auto_score": score_record(rec),
    }


def main():
    all_chunks = load_jsonl(CHUNKS_PATH)
    seen_ids = load_seen_ids(SEEN_PATH)

    grouped = defaultdict(list)

    for rec in all_chunks:
        chunk_id = rec.get("chunk_id") or rec.get("id")
        if not chunk_id or chunk_id in seen_ids:
            continue

        bucket = detect_bucket(rec.get("source_file", ""))
        if bucket is None:
            continue

        text = rec.get("text", "")
        if len(text) < 300:
            continue

        item = make_bootstrapped_record(rec)
        grouped[bucket].append(item)

    for bucket in grouped:
        grouped[bucket].sort(key=lambda x: x["auto_score"], reverse=True)

    result = []

    for bucket, need_count in TARGET_COUNTS.items():
        selected = grouped[bucket][:need_count]
        result.extend(selected)

    result.sort(key=lambda x: (x["source_file"], x["page_num"], x["chunk_id"]))

    save_json(result, OUTPUT_PATH)

    print("Всего новых кандидатов после фильтрации:")
    for bucket in ["pue", "gost", "sp"]:
        print(f"  {bucket}: {len(grouped[bucket])}")

    print()
    print("Экспортировано в manual batch:", len(result))
    print("Файл:", OUTPUT_PATH)
    print()

    for item in result[:10]:
        print("=" * 80)
        print("id       :", item["id"])
        print("source   :", item["source_file"])
        print("page_num :", item["page_num"])
        print("auto_score:", item["auto_score"])
        print("entities :")
        if not item["entities"]:
            print("  - <empty>")
        else:
            for ent in item["entities"][:10]:
                print(f"  - {ent['label']}: {ent['text']} [{ent['start']}, {ent['end']}]")
        print()


if __name__ == "__main__":
    main()