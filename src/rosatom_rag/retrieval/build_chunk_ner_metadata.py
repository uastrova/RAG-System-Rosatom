from pathlib import Path
import json
import re
from rosatom_rag.config import DATA_DIR
from rosatom_rag.utils import print_header


PREDICTIONS_PATH = DATA_DIR / "ner" / "predictions" / "chunk_entities_full_v2.jsonl"
OUTPUT_PATH = DATA_DIR / "ner" / "entity_index" / "chunk_ner_metadata_v2.jsonl"


BAD_ENTITY_TEXT_PATTERNS = [
    re.compile(r"^#+"),
    re.compile(r"^[\.\,\-\–—\:\;\(\)\[\]\s]+$"),
]

BAD_NORM_REF_PATTERNS = [
    re.compile(r"^\d+$"),
    re.compile(r"^\d+(?:\.\d+)*$"),
]

BAD_NORM_DOC_PATTERNS = [
    re.compile(r"^\d+$"),
    re.compile(r"^пуэ\s*[\.\-–—]?\s*\d+$", re.IGNORECASE),
]

ALLOWED_LABELS = {"NORM_DOC", "NORM_REF"}

GENERIC_REF_TEXTS = {
    "приложение",
    "приложение к",
    "форма",
    "рисунок",
    "таблица",
    "глава",
    "раздел",
    "пункт",
}


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_entity_text(text: str) -> str:
    text = text.lower().replace("ё", "е").replace("\xa0", " ")

    text = re.sub(r"\bтабл\.\s*", "таблица ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bрис\.\s*", "рисунок ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bпп\.\s*", "пп. ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bп\.\s*", "п. ", text, flags=re.IGNORECASE)

    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predicted_entity_is_bad(ent: dict) -> bool:
    text_raw = ent["text"]
    text = normalize_spaces(text_raw)
    label = ent["label"]

    if not text:
        return True

    if "##" in text_raw:
        return True

    for pattern in BAD_ENTITY_TEXT_PATTERNS:
        if pattern.match(text):
            return True

    if label == "NORM_REF":
        for pattern in BAD_NORM_REF_PATTERNS:
            if pattern.match(text):
                return True

    if label == "NORM_DOC":
        for pattern in BAD_NORM_DOC_PATTERNS:
            if pattern.match(text):
                return True

    return False


def is_informative_entity(label: str, text: str) -> bool:
    if label not in ALLOWED_LABELS:
        return False

    if not text:
        return False

    if label == "NORM_DOC":
        return True

    # NORM_REF
    if text in GENERIC_REF_TEXTS:
        return False

    if re.search(r"\d", text):
        return True

    informative_prefixes = (
        "глава ",
        "раздел ",
        "таблица ",
        "рисунок ",
        "п. ",
        "пп. ",
    )
    return text.startswith(informative_prefixes)


def clean_predicted_entities(entities):
    seen = set()
    result = []

    for ent in entities:
        if predicted_entity_is_bad(ent):
            continue

        label = ent["label"]
        text_norm = normalize_entity_text(ent["text"])

        if not is_informative_entity(label, text_norm):
            continue

        key = (label, text_norm)
        if key in seen:
            continue

        seen.add(key)
        result.append({
            "label": label,
            "text": text_norm,
        })

    result.sort(key=lambda x: (x["label"], x["text"]))
    return result


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
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    print_header("BUILD ENTITY INDEX V2")
    print("PREDICTIONS_PATH:", PREDICTIONS_PATH)
    print("OUTPUT_PATH     :", OUTPUT_PATH)

    records = load_jsonl(PREDICTIONS_PATH)
    result = []

    for rec in records:
        entities = clean_predicted_entities(rec.get("predicted_entities", []))
        entity_texts = [ent["text"] for ent in entities]
        entity_labels = [ent["label"] for ent in entities]

        result.append({
            "chunk_id": rec["chunk_id"],
            "source_file": rec.get("source_file"),
            "page_num": rec.get("page_num"),
            "entity_labels": entity_labels,
            "entity_texts": entity_texts,
        })

    save_jsonl(result, OUTPUT_PATH)

    print("Исходных записей:", len(records))
    print("Сохранено записей:", len(result))
    print("Готово.")


if __name__ == "__main__":
    main()