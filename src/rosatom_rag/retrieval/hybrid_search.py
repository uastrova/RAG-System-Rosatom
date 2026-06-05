from functools import lru_cache

from langchain_community.vectorstores import FAISS

from rosatom_rag.config import VECTORSTORE_DIR, EMB_MODEL_DIR
from rosatom_rag.retrieval.embeddings import LocalSentenceTransformerEmbeddings
from rosatom_rag.retrieval.bm25_search import similarity_search_bm25
from rosatom_rag.retrieval.reranker import rerank_documents


# FAISS + BM25 + RRF + reranker

@lru_cache(maxsize=1)
def load_vectorstore():
    embeddings = LocalSentenceTransformerEmbeddings(model_path=str(EMB_MODEL_DIR))
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def similarity_search_faiss_raw(question: str, k: int = 20):
    vectorstore = load_vectorstore()
    return vectorstore.similarity_search(question, k=k)


def collect_hybrid_candidates(
    question: str,
    faiss_k: int = 20,
    bm25_k: int = 20,
    rrf_k: int = 60,
    max_candidates: int = 20,
):
    faiss_docs = similarity_search_faiss_raw(question, k=faiss_k)
    bm25_docs = [] if bm25_k <= 0 else similarity_search_bm25(question, k=bm25_k)

    merged = {}

    for rank, doc in enumerate(faiss_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id not in merged:
            merged[chunk_id] = {
                "doc": doc,
                "faiss_rank": None,
                "bm25_rank": None,
            }

        merged[chunk_id]["faiss_rank"] = rank

    for rank, doc in enumerate(bm25_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id not in merged:
            merged[chunk_id] = {
                "doc": doc,
                "faiss_rank": None,
                "bm25_rank": None,
            }

        merged[chunk_id]["bm25_rank"] = rank

    scored_docs = []

    for item in merged.values():
        doc = item["doc"]
        faiss_rank = item["faiss_rank"]
        bm25_rank = item["bm25_rank"]

        hybrid_score = 0.0

        if faiss_rank is not None:
            hybrid_score += 1.0 / (rrf_k + faiss_rank)

        if bm25_rank is not None:
            hybrid_score += 1.0 / (rrf_k + bm25_rank)

        doc.metadata["faiss_rank"] = faiss_rank
        doc.metadata["bm25_rank"] = bm25_rank
        doc.metadata["hybrid_score"] = hybrid_score

        scored_docs.append(doc)

    scored_docs.sort(key=lambda d: d.metadata["hybrid_score"], reverse=True)
    return scored_docs[:max_candidates]


def similarity_search_hybrid_reranked(
    question: str,
    k: int = 6,
    faiss_k: int = 20,
    bm25_k: int = 20,
    rrf_k: int = 60,
    max_candidates: int = 20,
    reranker_max_length: int = 1024,
):
    candidates = collect_hybrid_candidates(
        question=question,
        faiss_k=faiss_k,
        bm25_k=bm25_k,
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