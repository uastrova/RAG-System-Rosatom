from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "ntd"
PROCESSED_DIR = DATA_DIR / "processed"

EXTRACTED_TEXT_DIR = PROCESSED_DIR / "extracted_text"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
VECTORSTORE_DIR = PROCESSED_DIR / "vectorstore" / "faiss_ntd"

MODELS_DIR = PROJECT_ROOT / "models"
EMB_MODEL_DIR = MODELS_DIR / "emb" / "sbert_large_nlu_ru"

LLAMA_BASE_URL = "http://127.0.0.1:8080/v1"
LLAMA_API_KEY = "sk-no-key-required"

TOP_K = 4
TOP_K_RAW = 8
TOP_K_FINAL = 4