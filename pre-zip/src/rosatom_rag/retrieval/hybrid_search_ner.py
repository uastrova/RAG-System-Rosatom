from functools import lru_cache
from copy import deepcopy

from rosatom_rag.retrieval.hybrid_search import (
    similarity_search_faiss_raw,
    exact_table_reference_search,
    load_chunk_documents,
)
from rosatom_rag.retrieval.bm25_search import similarity_search_bm25
from rosatom_rag.retrieval.bm25_entity_search import similarity_search_bm25_entity
from rosatom_rag.retrieval.reranker import rerank_documents


@lru_cache(maxsize=1)
def load_chunk_doc_map():
    docs = load_chunk_documents()
    return {doc.metadata["chunk_id"]: doc for doc in docs}


def clone_doc(doc):
    return deepcopy(doc)


def get_full_doc_by_chunk_id(chunk_id: str):
    doc_map = load_chunk_doc_map()
    doc = doc_map.get(chunk_id)
    if doc is None:
        return None
    return clone_doc(doc)


def collect_hybrid_ner_candidates(
    question: str,
    faiss_k: int = 20,
    bm25_k: int = 20,
    exact_k: int = 10,
    entity_k: int = 5,
    rrf_k: int = 60,
    max_candidates: int = 20,
    entity_rrf_weight: float = 0.35,
):
    faiss_docs = similarity_search_faiss_raw(question, k=faiss_k)
    bm25_docs = similarity_search_bm25(question, k=bm25_k)
    exact_docs = exact_table_reference_search(question, k=exact_k)
    entity_docs = similarity_search_bm25_entity(question, k=entity_k)

    merged = {}

    def ensure_item(chunk_id, full_doc):
        if chunk_id not in merged:
            merged[chunk_id] = {
                "doc": full_doc,
                "faiss_rank": None,
                "bm25_rank": None,
                "exact_rank": None,
                "entity_rank": None,
                "entity_bm25_score": None,
            }

    for rank, doc in enumerate(faiss_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        ensure_item(chunk_id, clone_doc(doc))
        merged[chunk_id]["faiss_rank"] = rank

    for rank, doc in enumerate(bm25_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        ensure_item(chunk_id, clone_doc(doc))
        merged[chunk_id]["bm25_rank"] = rank

    for rank, doc in enumerate(exact_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        ensure_item(chunk_id, clone_doc(doc))
        merged[chunk_id]["exact_rank"] = rank

    for rank, doc in enumerate(entity_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        full_doc = get_full_doc_by_chunk_id(chunk_id)
        if full_doc is None:
            continue

        ensure_item(chunk_id, full_doc)
        merged[chunk_id]["entity_rank"] = rank
        merged[chunk_id]["entity_bm25_score"] = doc.metadata.get("entity_bm25_score")
        merged[chunk_id]["doc"].metadata["entity_texts"] = doc.metadata.get("entity_texts", [])
        merged[chunk_id]["doc"].metadata["entity_labels"] = doc.metadata.get("entity_labels", [])

    scored_docs = []

    for item in merged.values():
        doc = item["doc"]
        faiss_rank = item["faiss_rank"]
        bm25_rank = item["bm25_rank"]
        exact_rank = item["exact_rank"]
        entity_rank = item["entity_rank"]
        entity_bm25_score = item["entity_bm25_score"]

        hybrid_score = 0.0

        if faiss_rank is not None:
            hybrid_score += 1.0 / (rrf_k + faiss_rank)

        if bm25_rank is not None:
            hybrid_score += 1.0 / (rrf_k + bm25_rank)

        if exact_rank is not None:
            hybrid_score += 1.0 / exact_rank

        if entity_rank is not None:
            hybrid_score += entity_rrf_weight * (1.0 / (rrf_k + entity_rank))

        doc.metadata["faiss_rank"] = faiss_rank
        doc.metadata["bm25_rank"] = bm25_rank
        doc.metadata["exact_rank"] = exact_rank
        doc.metadata["entity_rank"] = entity_rank
        doc.metadata["entity_bm25_score"] = entity_bm25_score
        doc.metadata["hybrid_score"] = hybrid_score

        scored_docs.append(doc)

    scored_docs.sort(key=lambda d: d.metadata["hybrid_score"], reverse=True)
    return scored_docs[:max_candidates]


def similarity_search_hybrid_ner_reranked(
    question: str,
    k: int = 6,
    faiss_k: int = 20,
    bm25_k: int = 20,
    exact_k: int = 10,
    entity_k: int = 5,
    rrf_k: int = 60,
    max_candidates: int = 20,
    reranker_max_length: int = 1024,
):
    candidates = collect_hybrid_ner_candidates(
        question=question,
        faiss_k=faiss_k,
        bm25_k=bm25_k,
        exact_k=exact_k,
        entity_k=entity_k,
        rrf_k=rrf_k,
        max_candidates=max_candidates,
    )

    reranked = rerank_documents(
        question=question,
        docs=candidates,
        top_k=len(candidates),
        max_length=reranker_max_length,
    )

    reranked.sort(key=lambda d: d.metadata.get("reranker_score", 0.0), reverse=True)
    return reranked[:k]