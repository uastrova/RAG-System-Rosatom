from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def load_reranker_components():
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device


def rerank_documents(question: str, docs, top_k: int = 6, max_length: int = 1024, batch_size: int = 8):
    if not docs:
        return []

    tokenizer, model, device = load_reranker_components()

    scored_docs = []

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        pairs = [(question, doc.page_content) for doc in batch_docs]

        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits.view(-1)
            scores = torch.sigmoid(logits).detach().cpu().tolist()

        for doc, score in zip(batch_docs, scores):
            doc.metadata["reranker_score"] = float(score)
            scored_docs.append(doc)

    scored_docs.sort(key=lambda d: d.metadata["reranker_score"], reverse=True)
    return scored_docs[:top_k]