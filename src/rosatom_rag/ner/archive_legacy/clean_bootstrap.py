from pathlib import Path
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "ner" / "labeled" / "batch_001_bootstrapped.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "labeled" / "batch_001_cleaned.jsonl"


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


def normalize_entity_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" -", "-").replace("- ", "-")
    return text


def is_bad_entity(ent: dict) -> bool:
    text = normalize_entity_text(ent["text"])
    label = ent["label"]

    if not text:
        return True

    if len(text) < 2:
        return True

    if label == "NORM_DOC":
        if "ГОСТ ГОСТ" in text:
            return True

        if text.startswith("ГОСТ IEC") and not re.search(r"\d{4}", text) and not re.search(r"-\d", text):
            return True

        if text in {"ГОСТ", "СП", "ПУЭ"}:
            return text != "ПУЭ"

    return False


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    seen = set()
    result = []

    for ent in entities:
        text = normalize_entity_text(ent["text"])
        key = (text, ent["label"], ent["start"], ent["end"])

        if key in seen:
            continue

        seen.add(key)

        result.append({
            "text": text,
            "label": ent["label"],
            "start": ent["start"],
            "end": ent["end"],
        })

    result.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return result


def clean_record(record: dict) -> dict:
    entities = record.get("entities", [])
    cleaned = []

    for ent in entities:
        normalized = {
            "text": normalize_entity_text(ent["text"]),
            "label": ent["label"],
            "start": ent["start"],
            "end": ent["end"],
        }

        if not is_bad_entity(normalized):
            cleaned.append(normalized)

    cleaned = deduplicate_entities(cleaned)

    record["entities"] = cleaned
    record["needs_review"] = True
    return record


def main():
    records = load_jsonl(INPUT_PATH)
    cleaned = [clean_record(record) for record in records]
    save_jsonl(cleaned, OUTPUT_PATH)

    print("Исходных записей:", len(records))
    print("Сохранено в:", OUTPUT_PATH)
    print()

    for item in cleaned[:5]:
        print("=" * 80)
        print("id       :", item["id"])
        print("entities :")
        for ent in item["entities"]:
            print(f"  - {ent['label']}: {ent['text']} [{ent['start']}, {ent['end']}]")
        print()


if __name__ == "__main__":
    main()