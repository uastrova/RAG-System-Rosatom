from pathlib import Path
import json
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from rosatom_rag.ner.labels import LABEL_LIST, LABEL2ID
from rosatom_rag.config import LABELED_DIR, SPLITS_DIR


INPUT_PATH =  LABELED_DIR / "manual_corpus_final.jsonl"
OUTPUT_DIR = SPLITS_DIR / "hf_rubert_dataset_no_docnum_v1"
LABELS_PATH = SPLITS_DIR / "label_list_no_docnum_v1.json"

TOKENIZER_NAME = "ai-forever/ruBert-base"
MAX_LENGTH = 384
SEED = 42
DEV_SIZE = 0.2


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def clean_entities(record):
    text = record["text"]
    entities = []

    for ent in record.get("entities", []):
        start = ent["start"]
        end = ent["end"]
        label = ent["label"]
        ent_text = ent["text"]

        if start < 0 or end > len(text) or start >= end:
            continue

        entities.append({
            "text": ent_text,
            "label": label,
            "start": start,
            "end": end,
        })

    entities.sort(key=lambda x: (x["start"], x["end"]))
    record["entities"] = entities
    return record


def split_records(records, seed=SEED, dev_size=DEV_SIZE):
    records = records[:]
    random.Random(seed).shuffle(records)

    dev_count = max(1, int(len(records) * dev_size))
    dev_records = records[:dev_count]
    train_records = records[dev_count:]

    return train_records, dev_records


def align_record(tokenizer, record):
    text = record["text"]
    entities = record["entities"]

    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    offsets = enc["offset_mapping"]
    labels = [-100] * len(offsets)

    # Сначала всем реальным токенам ставим O
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        labels[i] = LABEL2ID["O"]

    # Затем накладываем entity labels
    for ent in entities:
        ent_start = ent["start"]
        ent_end = ent["end"]
        ent_label = ent["label"]

        token_indices = []

        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:
                continue

            overlap = max(0, min(tok_end, ent_end) - max(tok_start, ent_start))
            if overlap > 0:
                token_indices.append(i)

        if not token_indices:
            continue

        first_idx = token_indices[0]
        labels[first_idx] = LABEL2ID[f"B-{ent_label}"]

        for idx in token_indices[1:]:
            labels[idx] = LABEL2ID[f"I-{ent_label}"]

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    item = {
        "id": record["id"],
        "source_file": record["source_file"],
        "page_num": record["page_num"],
        "chunk_id": record["chunk_id"],
        "text": text,
        "tokens": tokens,
        "labels": labels,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }

    if "token_type_ids" in enc:
        item["token_type_ids"] = enc["token_type_ids"]

    return item


def main():
    records = load_jsonl(INPUT_PATH)
    records = [clean_entities(record) for record in records]

    print("Загружено records:", len(records))

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print("Tokenizer loaded:", TOKENIZER_NAME)

    train_records, dev_records = split_records(records)
    print("Train size:", len(train_records))
    print("Validation size:", len(dev_records))

    train_items = [align_record(tokenizer, record) for record in train_records]
    dev_items = [align_record(tokenizer, record) for record in dev_records]

    dataset = DatasetDict({
        "train": Dataset.from_list(train_items),
        "validation": Dataset.from_list(dev_items),
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(LABEL_LIST, f, ensure_ascii=False, indent=2)

    print("Dataset saved to:", OUTPUT_DIR)
    print("Labels saved to :", LABELS_PATH)
    print()

    print("Sample train item:")
    sample = train_items[0]
    print("id:", sample["id"])
    print("source_file:", sample["source_file"])
    print("tokens[:30]:", sample["tokens"][:30])
    print("labels[:30]:", sample["labels"][:30])


if __name__ == "__main__":
    main()