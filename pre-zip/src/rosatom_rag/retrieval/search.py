from pathlib import Path
import json
import re
from functools import lru_cache

from langchain_community.vectorstores import FAISS

from rosatom_rag.config import VECTORSTORE_DIR, EMB_MODEL_DIR, TOP_K
from rosatom_rag.retrieval.embeddings import LocalSentenceTransformerEmbeddings
from rosatom_rag.ner.inference import get_ner_pipeline, predict_entities


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHUNK_ENTITIES_PATH = PROJECT_ROOT / "data" / "ner" / "predictions" / "chunk_entities_full.jsonl"

TOP_K_RAW = 40

LABEL_WEIGHTS = {
    "NORM_REF": 5.0,
    "NORM_DOC": 4.0,
    "DATE": 2.5,
    "ORG": 1.5,
    "ADDRESS": 1.0,
}


def normalize_entity_text(text: str) -> str:
    text = text.lower().replace("ё", "е").replace("№", "n")
    text = re.sub(r"\s+", "", text)
    return text.strip()


@lru_cache(maxsize=1)
def load_chunk_entities_map():
    entity_map = {}

    with open(CHUNK_ENTITIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            chunk_id = record["chunk_id"]
            entities = record.get("predicted_entities", [])
            entity_map[chunk_id] = entities

    return entity_map


def build_entity_key(ent: dict):
    return ent["label"], normalize_entity_text(ent["text"])


def score_entity_overlap(question_entities, chunk_entities):
    if not question_entities:
        return 0.0

    chunk_keys = {build_entity_key(ent) for ent in chunk_entities}
    score = 0.0

    used = set()
    for ent in question_entities:
        key = build_entity_key(ent)
        if key in chunk_keys and key not in used:
            score += LABEL_WEIGHTS.get(ent["label"], 1.0)
            used.add(key)

    return score


def load_vectorstore():
    embeddings = LocalSentenceTransformerEmbeddings(model_path=str(EMB_MODEL_DIR))

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def similarity_search(question: str, k: int = TOP_K, top_k_raw: int = TOP_K_RAW):
    vectorstore = load_vectorstore()

    dense_results = vectorstore.similarity_search_with_score(question, k=top_k_raw)

    ner_pipe = get_ner_pipeline()
    question_entities = predict_entities(question, ner_pipe=ner_pipe)
    chunk_entity_map = load_chunk_entities_map()

    scored_docs = []

    for rank, (doc, _) in enumerate(dense_results, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        chunk_entities = chunk_entity_map.get(chunk_id, [])

        dense_rank_score = top_k_raw - rank + 1
        ner_score = score_entity_overlap(question_entities, chunk_entities)

        final_score = dense_rank_score + 10.0 * ner_score

        doc.metadata["dense_rank"] = rank
        doc.metadata["ner_score"] = ner_score
        doc.metadata["final_score"] = final_score
        doc.metadata["question_entities"] = question_entities
        doc.metadata["chunk_entities"] = chunk_entities

        scored_docs.append(doc)

    scored_docs.sort(key=lambda d: d.metadata["final_score"], reverse=True)
    return scored_docs[:k]