from functools import lru_cache
import json
import re

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from rosatom_rag.config import DATA_DIR


ENTITY_INDEX_PATH = DATA_DIR / "ner" / "entity_index" / "chunk_entity_index_v2.jsonl"

STOPWORDS = {
    "и", "в", "во", "на", "по", "о", "об", "от", "до", "из", "за", "с", "со",
    "к", "ко", "у", "для", "про", "при", "что", "какие", "какой", "какая",
    "какие", "указано", "сказано", "относится", "относятся", "требования",
    "предъявляются",
}


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е").replace("\xa0", " ")

    text = re.sub(r"\bтаблиц[аеиуы]\b", "таблица", text, flags=re.IGNORECASE)
    text = re.sub(r"\bглав[аеиуы]\b", "глава", text, flags=re.IGNORECASE)
    text = re.sub(r"\bраздел[аеиуы]?\b", "раздел", text, flags=re.IGNORECASE)
    text = re.sub(r"\bрис\.\s*", "рисунок ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bрисунк[аеиуы]?\b", "рисунок", text, flags=re.IGNORECASE)

    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    text = normalize_text(text)
    raw_tokens = re.findall(r"[a-zа-яё0-9\.]+", text, flags=re.IGNORECASE)

    tokens = []
    for token in raw_tokens:
        if token in STOPWORDS:
            continue

        if len(token) < 2 and not re.search(r"\d", token):
            continue

        tokens.append(token)

    return tokens


@lru_cache(maxsize=1)
def load_entity_records():
    records = []
    with open(ENTITY_INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@lru_cache(maxsize=1)
def load_entity_bm25():
    records = load_entity_records()
    corpus_tokens = [tokenize(rec["entity_blob"]) for rec in records]
    return BM25Okapi(corpus_tokens)


def similarity_search_bm25_entity(question: str, k: int = 10):
    records = load_entity_records()
    bm25 = load_entity_bm25()

    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked = sorted(
        zip(records, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    docs = []
    for rec, score in ranked[:k]:
        docs.append(
            Document(
                page_content=rec["entity_blob"],
                metadata={
                    "chunk_id": rec["chunk_id"],
                    "source_file": rec.get("source_file"),
                    "page_num": rec.get("page_num"),
                    "entity_texts": rec.get("entity_texts", []),
                    "entity_labels": rec.get("entity_labels", []),
                    "entity_bm25_score": float(score),
                },
            )
        )

    return docs