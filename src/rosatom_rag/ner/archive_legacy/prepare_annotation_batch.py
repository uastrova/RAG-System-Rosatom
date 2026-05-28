from pathlib import Path
import json
import re
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks" / "ntd_chunks.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "raw_samples" / "batch_001_candidates.jsonl"


PATTERNS = [
    r"\bГОСТ\b",
    r"\bСП\b",
    r"\bНП-\d+-\d+\b",
    r"\bПУЭ\b",
    r"\bГОСТ\s+[A-ZА-ЯЁ]*\s*\d[\d\.\-–—]*\b",
    r"\bСП\s+\d[\d\.]*\b",
    r"\bп\.\s*\d+(\.\d+)*\b",
    r"\bгл\.\s*\d+(\.\d+)*\b",
    r"\bраздел\s+\d+(\.\d+)*\b",
    r"\bтабл\.\s*\d+\b",
    r"\bТабл\.\s*\d+\b",
    r"\bN\s*\d[\w\-–—]*\b",
    r"\b№\s*\d[\w\-–—]*\b",
    r"\b\d{2}\.\d{2}\.\d{4}\b",
    r"\b\d{1,2}\s+[а-яА-ЯЁё]+\s+\d{4}\s*г\.?\b",
]


def load_chunks(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score_text(text: str) -> int:
    score = 0
    for pattern in PATTERNS:
        score += len(re.findall(pattern, text))
    return score


def score_record(record: dict) -> int:
    text = record.get("text", "")
    base = score_text(text)

    # штрафуем слишком короткие куски
    if len(text) < 300:
        base -= 2

    # штрафуем слишком длинные куски
    if len(text) > 2000:
        base -= 1

    # бонус за сочетание разных типов паттернов
    has_doc = any(re.search(p, text) for p in [
        r"\bГОСТ\b", r"\bСП\b", r"\bНП-\d+-\d+\b", r"\bПУЭ\b"
    ])
    has_ref = any(re.search(p, text) for p in [
        r"\bп\.\s*\d+(\.\d+)*\b", r"\bгл\.\s*\d+(\.\d+)*\b", r"\bраздел\s+\d+(\.\d+)*\b"
    ])
    has_date = any(re.search(p, text) for p in [
        r"\b\d{2}\.\d{2}\.\d{4}\b",
        r"\b\d{1,2}\s+[а-яА-ЯЁё]+\s+\d{4}\s*г\.?\b"
    ])
    has_docnum = any(re.search(p, text) for p in [
        r"\bN\s*\d[\w\-–—]*\b", r"\b№\s*\d[\w\-–—]*\b"
    ])

    diversity_bonus = sum([has_doc, has_ref, has_date, has_docnum])
    base += diversity_bonus

    return base


def prepare_candidates(records, limit_total: int = 200, max_per_source: int = 3):
    scored = []

    for record in records:
        score = score_record(record)
        if score > 0:
            scored.append((score, record))

    scored.sort(key=lambda x: x[0], reverse=True)

    by_source = defaultdict(list)
    for score, record in scored:
        by_source[record["source_file"]].append((score, record))

    selected = []
    used_chunk_ids = set()

    # первый проход: берем top chunks из максимально разных source_file
    source_names = sorted(
        by_source.keys(),
        key=lambda s: max(x[0] for x in by_source[s]),
        reverse=True,
    )

    for source_name in source_names:
        taken_for_source = 0
        for score, record in by_source[source_name]:
            if taken_for_source >= max_per_source:
                break
            if record["chunk_id"] in used_chunk_ids:
                continue

            selected.append({
                "id": record["chunk_id"],
                "source_file": record["source_file"],
                "page_num": record["page_num"],
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "entities": [],
                "auto_score": score,
            })
            used_chunk_ids.add(record["chunk_id"])
            taken_for_source += 1

            if len(selected) >= limit_total:
                return selected

    return selected


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    records = load_chunks(CHUNKS_PATH)
    candidates = prepare_candidates(records, limit_total=200, max_per_source=3)
    save_jsonl(candidates, OUTPUT_PATH)

    print("Всего chunks в исходном файле:", len(records))
    print("Подготовлено кандидатов для разметки:", len(candidates))
    print("Сохранено в:", OUTPUT_PATH)

    print("\nПервые 10 кандидатов:")
    for item in candidates[:10]:
        print("-" * 80)
        print("id        :", item["id"])
        print("source    :", item["source_file"])
        print("page_num  :", item["page_num"])
        print("auto_score:", item["auto_score"])
        print("text:")
        print(item["text"][:700])


if __name__ == "__main__":
    main()