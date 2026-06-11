from rosatom_rag.retrieval.hybrid_search import collect_hybrid_candidates
from rosatom_rag.retrieval.ner_faiss_rerank import rerank_hybrid_docs_with_ner


# Сильный retrieval baseline и NER-enhanced вариант:
# 1. hybrid: FAISS + BM25 + RRF
# 2. hybrid_ner: FAISS + BM25 + RRF + rule-based query entities + chunk NER metadata


def retrieve_hybrid(
    question: str,
    k: int = 10,
    faiss_k: int = 100,
    bm25_k: int = 100,
    rrf_k: int = 60,
):
    docs = collect_hybrid_candidates(
        question=question,
        faiss_k=faiss_k,
        bm25_k=bm25_k,
        rrf_k=rrf_k,
        max_candidates=k,
    )

    for rank, doc in enumerate(docs, start=1):
        doc.metadata["final_rank"] = rank
        doc.metadata["retrieval_pipeline"] = "hybrid"

    return docs


def retrieve_hybrid_ner(
    question: str,
    k: int = 10,
    candidates_k: int = 100,
    faiss_k: int = 100,
    bm25_k: int = 100,
    rrf_k: int = 60,
    ner_weight: float = 0.5,
):
    candidates = collect_hybrid_candidates(
        question=question,
        faiss_k=faiss_k,
        bm25_k=bm25_k,
        rrf_k=rrf_k,
        max_candidates=candidates_k,
    )
    reranked_docs = rerank_hybrid_docs_with_ner(
        question=question,
        docs=candidates,
        top_k=k,
        ner_weight=ner_weight,
    )

    for rank, doc in enumerate(reranked_docs, start=1):
        doc.metadata["final_rank"] = rank
        doc.metadata["retrieval_pipeline"] = "hybrid_ner"

    return reranked_docs
