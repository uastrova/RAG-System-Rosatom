from functools import lru_cache
import re
from pathlib import Path
import json

from rosatom_rag.config import ENTITY_INDEX_DIR


CHUNK_NER_METADATA_PATH = ENTITY_INDEX_DIR / "chunk_ner_metadata_v2.jsonl"


# NER/rule-based reranking для retrieval-кандидатов.
# Документные сущности берём из заранее построенной chunk NER metadata,
# а query-side сущности извлекаем регулярками: это устойчивее для коротких
# вопросов с формальными ссылками на СП/НП/СТО/ПУЭ, пункты, главы и таблицы.


QUERY_ENTITY_PATTERNS = [
    re.compile(r"\bСП\s+\d+(?:\.\d+)*(?:-\d+)?\b", re.IGNORECASE),
    re.compile(r"\bНП[-\s]?\d+[-–—]\d+\b", re.IGNORECASE),
    re.compile(r"\bСТО\s+\d+\s+\d+[-–—]\d+\b", re.IGNORECASE),
    re.compile(r"\bПУЭ\b", re.IGNORECASE),
    re.compile(r"\bглав[аеуы]\s+\d+(?:\.\d+)*\b", re.IGNORECASE),
    re.compile(r"\bраздел[ае]?\s+\d+(?:\.\d+)*\b", re.IGNORECASE),
    re.compile(r"\bп\.\s*\d+(?:\.\d+)*\b", re.IGNORECASE),
    re.compile(r"\bпункт[ае]?\s+\d+(?:\.\d+)*\b", re.IGNORECASE),
    re.compile(r"\bтабл\.\s*[А-ЯA-Z]?\d+(?:\.\d+)*\b", re.IGNORECASE),
    re.compile(r"\bтаблиц[аеуы]\s+[А-ЯA-Z]?\d+(?:\.\d+)*\b", re.IGNORECASE),
]


def normalize_entity_text(text: str) -> str:
    text = text.lower().replace("ё", "е").replace("\xa0", " ")

    text = re.sub(r"\bтабл\.\s*", "таблица ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bтаблиц[аеуы]\s+", "таблица ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bрис\.\s*", "рисунок ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bпп\.\s*", "пп. ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bпункт[ае]?\s+", "п. ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bп\.\s*", "п. ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bглав[аеуы]\s+", "глава ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bраздел[ае]?\s+", "раздел ", text, flags=re.IGNORECASE)

    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    text = re.sub(r"\s*[-–—]\s*", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@lru_cache(maxsize=1)
def load_chunk_ner_metadata(metadata_path: str | Path = CHUNK_NER_METADATA_PATH):
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Chunk NER metadata file not found: {metadata_path}. "
            "Run: python -m rosatom_rag.retrieval.build_chunk_ner_metadata"
        )

    records = load_jsonl(metadata_path)
    return {record["chunk_id"]: record for record in records}


def extract_query_entity_texts_by_regex(question: str) -> set[str]:
    entities = set()

    for pattern in QUERY_ENTITY_PATTERNS:
        for match in pattern.finditer(question):
            entity_text = normalize_entity_text(match.group(0))
            if entity_text:
                entities.add(entity_text)

    return entities


def extract_query_entity_texts(question: str, use_model_fallback: bool = False) -> set[str]:
    entities = extract_query_entity_texts_by_regex(question)

    if not use_model_fallback:
        return entities

    from rosatom_rag.ner.inference import predict_entities

    model_entities = {
        normalize_entity_text(entity["text"])
        for entity in predict_entities(question)
        if entity.get("label") in {"NORM_DOC", "NORM_REF"}
        and normalize_entity_text(entity.get("text", ""))
    }
    return entities | model_entities


def score_ner_overlap(query_entities: set[str], chunk_entities: set[str]) -> tuple[float, list[str]]:
    if not query_entities:
        return 0.0, []

    overlap = sorted(query_entities & chunk_entities)
    score = len(overlap) / len(query_entities)
    return score, overlap


def get_chunk_entity_texts(chunk_id: str, metadata_by_chunk_id: dict) -> set[str]:
    chunk_metadata = metadata_by_chunk_id.get(chunk_id, {})
    chunk_entities = {
        normalize_entity_text(entity_text)
        for entity_text in chunk_metadata.get("entity_texts", [])
        if normalize_entity_text(entity_text)
    }

    # Добавляем rule-based сущности из chunk_id/source_file. Это важно для чанков,
    # где номер документа есть в имени файла, но не повторяется в тексте чанка.
    source_text = " ".join(
        str(value)
        for value in [chunk_id, chunk_metadata.get("source_file")]
        if value
    )
    chunk_entities.update(extract_query_entity_texts_by_regex(source_text))
    return chunk_entities


def add_ner_scores_to_docs(
    question: str,
    docs,
    ner_weight: float = 0.5,
    base_score_key: str | None = None,
    output_score_key: str = "ner_enhanced_score",
):
    query_entities = extract_query_entity_texts(question)
    metadata_by_chunk_id = load_chunk_ner_metadata()
    scored_docs = []

    for rank, doc in enumerate(docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        chunk_entities = get_chunk_entity_texts(chunk_id, metadata_by_chunk_id)
        ner_score, ner_overlap = score_ner_overlap(query_entities, chunk_entities)

        if base_score_key is None:
            base_score = 1.0 / rank
        else:
            base_score = float(doc.metadata.get(base_score_key, 0.0))

        final_score = base_score + ner_weight * ner_score

        doc.metadata["query_ner_entities"] = sorted(query_entities)
        doc.metadata["chunk_ner_entities"] = sorted(chunk_entities)
        doc.metadata["ner_overlap"] = ner_overlap
        doc.metadata["ner_score"] = float(ner_score)
        doc.metadata[output_score_key] = float(final_score)

        scored_docs.append(doc)

    scored_docs.sort(key=lambda d: d.metadata[output_score_key], reverse=True)
    return scored_docs


def rerank_faiss_docs_with_ner(
    question: str,
    docs,
    top_k: int = 10,
    ner_weight: float = 0.5,
):
    for faiss_rank, doc in enumerate(docs, start=1):
        doc.metadata["faiss_rank"] = faiss_rank

    scored_docs = add_ner_scores_to_docs(
        question=question,
        docs=docs,
        ner_weight=ner_weight,
        base_score_key=None,
        output_score_key="faiss_ner_score",
    )
    return scored_docs[:top_k]


def rerank_hybrid_docs_with_ner(
    question: str,
    docs,
    top_k: int = 10,
    ner_weight: float = 0.5,
):
    scored_docs = add_ner_scores_to_docs(
        question=question,
        docs=docs,
        ner_weight=ner_weight,
        base_score_key="hybrid_score",
        output_score_key="hybrid_ner_score",
    )
    return scored_docs[:top_k]
