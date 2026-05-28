from pathlib import Path
import json
import argparse

from rosatom_rag.ner.inference import get_ner_pipeline, predict_entities


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = PROJECT_ROOT / "models" / "ner" / "rubert_ner_manual_v3_full_finetune"
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks" / "ntd_chunks.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "predictions" / "no_docnum_full.jsonl"


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


def select_records(records, limit=None, source_substring=None, text_substring=None):
    filtered = []

    for record in records:
        source_ok = True
        text_ok = True

        if source_substring:
            source_ok = source_substring.lower() in record.get("source_file", "").lower()

        if text_substring:
            text_ok = text_substring.lower() in record.get("text", "").lower()

        if source_ok and text_ok:
            filtered.append(record)

        if limit is not None and len(filtered) >= limit:
            break

    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))

    args = parser.parse_args()
    output_path = Path(args.output)

    print("MODEL_DIR :", MODEL_DIR)
    print("INPUT_PATH :", INPUT_PATH)
    print("OUTPUT_PATH:", output_path)

    ner_pipe = get_ner_pipeline(MODEL_DIR)

    all_records = load_jsonl(INPUT_PATH)
    selected = select_records(
        all_records,
        limit=args.limit,
        source_substring=args.source,
        text_substring=args.text,
    )

    print("Всего чанков:", len(all_records))
    print("Выбрано чанков:", len(selected))
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    preview_items = []
    written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, record in enumerate(selected, start=1):
            text = record["text"]
            entities = predict_entities(text, ner_pipe=ner_pipe)

            result = {
                "id": record.get("id") or record.get("chunk_id"),
                "source_file": record.get("source_file"),
                "page_num": record.get("page_num"),
                "chunk_id": record.get("chunk_id"),
                "text": text,
                "predicted_entities": entities,
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1

            if len(preview_items) < 10:
                preview_items.append(result)

            if idx % 200 == 0:
                f.flush()
                print(f"Обработано: {idx}/{len(selected)}")

    for item in preview_items:
        print("=" * 80)
        print("id       :", item.get("id"))
        print("source   :", item["source_file"])
        print("page_num :", item["page_num"])
        print("entities :")
        if not item["predicted_entities"]:
            print("  - <empty>")
        else:
            for ent in item["predicted_entities"]:
                print(
                    f"  - {ent['label']}: {ent['text']} "
                    f"[{ent['start']}, {ent['end']}] "
                    f"score={ent['score']:.4f}"
                )
        print()

    print("Сохранено записей:", written)
    print("Сохранено в:", output_path)

if __name__ == "__main__":
    main()