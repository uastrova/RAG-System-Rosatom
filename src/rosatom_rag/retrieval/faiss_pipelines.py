from functools import lru_cache
from langchain_community.vectorstores import FAISS

from rosatom_rag.config import VECTORSTORE_DIR, EMB_MODEL_DIR
from rosatom_rag.retrieval.embeddings import LocalSentenceTransformerEmbeddings
from rosatom_rag.retrieval.ner_faiss_rerank import rerank_faiss_docs_with_ner


# пайплайны для сравнения:
# 1. FAISS
# 2. FAISS + NER reranking


@lru_cache(maxsize=1)
def load_faiss_vectorstore():
    embeddings = LocalSentenceTransformerEmbeddings(model_path=str(EMB_MODEL_DIR))
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def similarity_search_faiss(question: str, k: int = 10):
    vectorstore = load_faiss_vectorstore()
    return vectorstore.similarity_search(question, k=k)


def retrieve_faiss(question: str, k: int = 10):
    docs = similarity_search_faiss(question, k=k)

    for rank, doc in enumerate(docs, start=1):
        doc.metadata["faiss_rank"] = rank
        doc.metadata["retrieval_pipeline"] = "faiss"

    return docs


def retrieve_faiss_ner(
    question: str,
    k: int = 10,
    faiss_candidates_k: int = 50,
    ner_weight: float = 0.5,
):
    candidates = similarity_search_faiss(question, k=faiss_candidates_k)
    reranked_docs = rerank_faiss_docs_with_ner(
        question=question,
        docs=candidates,
        top_k=k,
        ner_weight=ner_weight,
    )

    for rank, doc in enumerate(reranked_docs, start=1):
        doc.metadata["final_rank"] = rank
        doc.metadata["retrieval_pipeline"] = "faiss_ner"

    return reranked_docs
