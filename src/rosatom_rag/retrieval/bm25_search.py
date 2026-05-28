from pathlib import Path
import json
import re
from functools import lru_cache
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from rosatom_rag.config import CHUNKS_DIR


# поиск BM25 - ловит точные совпадения слов без понимания смысла 


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е").replace("\xa0", " ")
    return text


def tokenize(text: str):
    text = normalize_text(text)
    return re.findall(r"[a-zа-я0-9]+(?:\.[a-zа-я0-9]+)*", text)


def load_chunk_records(chunks_path: Path):
    records = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_to_document(record: dict) -> Document:
    return Document(
        page_content=record["text"],
        metadata={
            "chunk_id": record["chunk_id"],
            "source_file": record["source_file"],
            "source_stem": record.get("source_stem", ""),
            "page_num": record["page_num"],
            "chunk_index_on_page": record.get("chunk_index_on_page", 0),
            "char_start": record.get("char_start", 0),
            "char_end": record.get("char_end", 0),
            "chunk_type": record.get("chunk_type", "text"),
        },
    )

# BM25 индекс
@lru_cache(maxsize=1)
def load_bm25_index():
    chunks_path = CHUNKS_DIR / "ntd_chunks.jsonl"
    records = load_chunk_records(chunks_path)

    tokenized_corpus = []
    documents = []

    for record in records:
        text = record.get("text", "")
        tokens = tokenize(text)

        if not tokens:
            tokens = ["__empty__"]

        # токены для поиска
        tokenized_corpus.append(tokens)

        # сам документ
        documents.append(record_to_document(record))

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, documents


def similarity_search_bm25(question: str, k: int = 6):
    bm25, documents = load_bm25_index()
    query_tokens = tokenize(question)

    if not query_tokens:
        return []
    
    # BM25 считает скор для каждого чанка (насколько соответствующий чанк похож на вопрос по словам)
    scores = bm25.get_scores(query_tokens)

    # первое число — индекс документа, второе — BM25 score
    scored = list(enumerate(scores))

    scored.sort(key=lambda x: x[1], reverse=True)

    top_docs = []
    for idx, score in scored[:k]:
        doc = documents[idx]
        doc.metadata["bm25_score"] = float(score)
        top_docs.append(doc)

    return top_docs