from functools import lru_cache
from pathlib import Path
import json
import re

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from rosatom_rag.config import VECTORSTORE_DIR, EMB_MODEL_DIR, CHUNKS_DIR
from rosatom_rag.retrieval.embeddings import LocalSentenceTransformerEmbeddings
from rosatom_rag.retrieval.bm25_search import similarity_search_bm25
from rosatom_rag.retrieval.reranker import rerank_documents


def normalize_ref_text(text: str) -> str:
    text = text.lower().replace("ё", "е").replace("\xa0", " ")
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bтаблиц[аеиуы]\b", "таблица", text)
    return text


def extract_table_refs(question: str):
    q = normalize_ref_text(question)
    refs = re.findall(r"таблица\s*(?:n|№)?\s*\d+(?:\.\d+)+", q, flags=re.IGNORECASE)
    return list(dict.fromkeys(refs))


@lru_cache(maxsize=1)
def load_vectorstore():
    embeddings = LocalSentenceTransformerEmbeddings(model_path=str(EMB_MODEL_DIR))
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


@lru_cache(maxsize=1)
def load_chunk_documents():
    path = CHUNKS_DIR / "ntd_chunks.jsonl"
    docs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            docs.append(
                Document(
                    page_content=rec["text"],
                    metadata={
                        "chunk_id": rec["chunk_id"],
                        "source_file": rec["source_file"],
                        "source_stem": rec.get("source_stem", ""),
                        "page_num": rec["page_num"],
                        "chunk_index_on_page": rec.get("chunk_index_on_page", 0),
                        "char_start": rec.get("char_start", 0),
                        "char_end": rec.get("char_end", 0),
                        "chunk_type": rec.get("chunk_type"),
                    },
                )
            )

    return docs


def similarity_search_faiss_raw(question: str, k: int = 20):
    vectorstore = load_vectorstore()
    return vectorstore.similarity_search(question, k=k)


def exact_table_reference_search(question: str, k: int = 10):
    refs = extract_table_refs(question)
    if not refs:
        return []

    docs = load_chunk_documents()
    matched = []

    for doc in docs:
        text_norm = normalize_ref_text(doc.page_content)

        matched_refs = [ref for ref in refs if ref in text_norm]
        if not matched_refs:
            continue

        score = float(len(matched_refs))
        if doc.metadata.get("chunk_type") == "table":
            score += 1.0

        doc.metadata["exact_ref_score"] = score
        matched.append(doc)

    matched.sort(
        key=lambda d: (
            d.metadata.get("exact_ref_score", 0.0),
            1 if d.metadata.get("chunk_type") == "table" else 0,
        ),
        reverse=True,
    )

    return matched[:k]


def collect_hybrid_candidates(
    question: str,
    faiss_k: int = 20,
    bm25_k: int = 20,
    exact_k: int = 10,
    rrf_k: int = 60,
    max_candidates: int = 20,
):
    faiss_docs = similarity_search_faiss_raw(question, k=faiss_k)
    bm25_docs = similarity_search_bm25(question, k=bm25_k)
    exact_docs = exact_table_reference_search(question, k=exact_k)

    merged = {}

    for rank, doc in enumerate(faiss_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id not in merged:
            merged[chunk_id] = {
                "doc": doc,
                "faiss_rank": None,
                "bm25_rank": None,
                "exact_rank": None,
            }
        merged[chunk_id]["faiss_rank"] = rank

    for rank, doc in enumerate(bm25_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id not in merged:
            merged[chunk_id] = {
                "doc": doc,
                "faiss_rank": None,
                "bm25_rank": None,
                "exact_rank": None,
            }
        merged[chunk_id]["bm25_rank"] = rank

    for rank, doc in enumerate(exact_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id not in merged:
            merged[chunk_id] = {
                "doc": doc,
                "faiss_rank": None,
                "bm25_rank": None,
                "exact_rank": None,
            }
        merged[chunk_id]["exact_rank"] = rank
        merged[chunk_id]["doc"].metadata["exact_ref_score"] = doc.metadata.get("exact_ref_score", 0.0)

    scored_docs = []

    for item in merged.values():
        doc = item["doc"]
        faiss_rank = item["faiss_rank"]
        bm25_rank = item["bm25_rank"]
        exact_rank = item["exact_rank"]

        hybrid_score = 0.0

        if faiss_rank is not None:
            hybrid_score += 1.0 / (rrf_k + faiss_rank)

        if bm25_rank is not None:
            hybrid_score += 1.0 / (rrf_k + bm25_rank)

        if exact_rank is not None:
            hybrid_score += 1.0 / exact_rank

        doc.metadata["faiss_rank"] = faiss_rank
        doc.metadata["bm25_rank"] = bm25_rank
        doc.metadata["exact_rank"] = exact_rank
        doc.metadata["hybrid_score"] = hybrid_score

        scored_docs.append(doc)

    scored_docs.sort(key=lambda d: d.metadata["hybrid_score"], reverse=True)
    return scored_docs[:max_candidates]


def similarity_search_hybrid_reranked(
    question: str,
    k: int = 6,
    faiss_k: int = 20,
    bm25_k: int = 20,
    exact_k: int = 10,
    rrf_k: int = 60,
    max_candidates: int = 20,
    reranker_max_length: int = 1024,
):
    candidates = collect_hybrid_candidates(
        question=question,
        faiss_k=faiss_k,
        bm25_k=bm25_k,
        exact_k=exact_k,
        rrf_k=rrf_k,
        max_candidates=max_candidates,
    )

    reranked = rerank_documents(
        question=question,
        docs=candidates,
        top_k=k,
        max_length=reranker_max_length,
    )
    return reranked