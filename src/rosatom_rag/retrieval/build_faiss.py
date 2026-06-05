from pathlib import Path
import json
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rosatom_rag.config import CHUNKS_DIR, VECTORSTORE_DIR, EMB_MODEL_DIR
from rosatom_rag.retrieval.embeddings import LocalSentenceTransformerEmbeddings
from rosatom_rag.utils import print_header


def load_chunk_records(chunks_path: Path):
    records = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def records_to_documents(records):
    documents = []
    for record in records:
        doc = Document(
            page_content=record["text"],
            metadata={
                "chunk_id": record["chunk_id"],
                "source_file": record["source_file"],
                "source_stem": record["source_stem"],
                "page_num": record["page_num"],
                "chunk_index_on_page": record["chunk_index_on_page"],
                "char_start": record["char_start"],
                "char_end": record["char_end"],
                "chunk_type": record["chunk_type"],
            }
        )
        documents.append(doc)
    return documents


def main():
    chunks_path = CHUNKS_DIR / "ntd_chunks.jsonl"
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    print_header("BUILD FAISS")
    print("CHUNKS_PATH:", chunks_path)
    print("VECTORSTORE_DIR:", VECTORSTORE_DIR)
    print("EMB_MODEL_DIR:", EMB_MODEL_DIR)

    records = load_chunk_records(chunks_path)
    print("Загружено chunk records:", len(records))

    documents = records_to_documents(records)
    print("Создано documents:", len(documents))

    embeddings = LocalSentenceTransformerEmbeddings(model_path=str(EMB_MODEL_DIR))
    print("Embedding-модель загружена")

    vectorstore = FAISS.from_documents(documents, embeddings)
    print("FAISS построен")

    vectorstore.save_local(str(VECTORSTORE_DIR))
    print("Индекс сохранён в:", VECTORSTORE_DIR)


if __name__ == "__main__":
    main()