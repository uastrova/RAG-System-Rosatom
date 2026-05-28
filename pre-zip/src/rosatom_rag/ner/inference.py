from pathlib import Path
import re

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "ner" / "rubert_ner_manual_v3_full_finetune"

BAD_ENTITY_TEXT_PATTERNS = [
    re.compile(r"^#+"),
    re.compile(r"^[\.\,\-\–—\:\;\(\)\[\]\s]+$"),
]

BAD_NORM_REF_PATTERNS = [
    re.compile(r"^\d+$"),
    re.compile(r"^\d+(?:\.\d+)*$"),
]

BAD_NORM_DOC_PATTERNS = [
    re.compile(r"^\d+$"),
    re.compile(r"^пуэ\s*[\.\-–—]?\s*\d+$", re.IGNORECASE),
]

_PIPELINE_CACHE = {}


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def predicted_entity_is_bad(ent: dict) -> bool:
    text_raw = ent["text"]
    text = normalize_spaces(text_raw)
    label = ent["label"]

    if not text:
        return True

    if "##" in text_raw:
        return True

    for pattern in BAD_ENTITY_TEXT_PATTERNS:
        if pattern.match(text):
            return True

    if label == "NORM_REF":
        for pattern in BAD_NORM_REF_PATTERNS:
            if pattern.match(text):
                return True

    if label == "NORM_DOC":
        for pattern in BAD_NORM_DOC_PATTERNS:
            if pattern.match(text):
                return True

    return False


def deduplicate_entities(entities):
    seen = set()
    result = []

    for ent in entities:
        key = (ent["label"], ent["start"], ent["end"], ent["text"])
        if key not in seen:
            seen.add(key)
            result.append(ent)

    return sorted(result, key=lambda x: (x["start"], x["end"], x["label"]))


def clean_predicted_entities(entities):
    cleaned = [ent for ent in entities if not predicted_entity_is_bad(ent)]
    return deduplicate_entities(cleaned)


def get_ner_pipeline(model_dir: Path | str = DEFAULT_MODEL_DIR):
    model_dir = str(model_dir)

    if model_dir in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_dir]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    tokenizer.model_max_length = 512

    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    ner_pipe = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        stride=128,
    )

    _PIPELINE_CACHE[model_dir] = ner_pipe
    return ner_pipe


def predict_entities(text: str, ner_pipe=None):
    if not text or not text.strip():
        return []

    if ner_pipe is None:
        ner_pipe = get_ner_pipeline()

    preds = ner_pipe(text)

    entities = []
    for pred in preds:
        entities.append({
            "text": pred["word"],
            "label": pred["entity_group"],
            "start": int(pred["start"]),
            "end": int(pred["end"]),
            "score": float(pred["score"]),
        })

    return clean_predicted_entities(entities)