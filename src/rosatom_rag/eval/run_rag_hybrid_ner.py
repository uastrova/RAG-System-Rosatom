from pathlib import Path
import json

from rosatom_rag.llm.llama_client import ask_llm_with_context, format_docs, collect_sources
from rosatom_rag.retrieval.hybrid_search_ner import similarity_search_hybrid_ner_reranked
from rosatom_rag.utils import print_header


OUTPUT_DIR = Path("data/eval/rag_hybrid_ner_v1")

QUESTIONS = [
    "Какие требования предъявляются к заземлению?",
    "Какие требования ПУЭ относятся к электроустановкам?",
    "Что указано в таблице 1.7.7?",
    "Какие требования предъявляются к защитному проводнику?",
    "Что сказано в ПУЭ про заземляющее устройство?",
]


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_one_question(question: str, top_k: int, out_path: Path):
    retrieved_docs = similarity_search_hybrid_ner_reranked(
        question,
        k=top_k,
        faiss_k=20,
        bm25_k=20,
        exact_k=10,
        entity_k=5,
        max_candidates=20,
        reranker_max_length=1024,
    )

    context = format_docs(retrieved_docs)
    answer = ask_llm_with_context(question, context)
    sources = collect_sources(retrieved_docs)

    result = {
        "question": question,
        "top_k": top_k,
        "answer": answer,
        "sources": sources,
    }

    save_json(result, out_path)
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_header("RAG HYBRID NER")
    print("OUTPUT_DIR:", OUTPUT_DIR)

    for i, question in enumerate(QUESTIONS, start=1):
        out_path = OUTPUT_DIR / f"test_{i:03d}.json"

        print("-" * 80)
        print("QUESTION:", question)

        result = run_one_question(question, top_k=3, out_path=out_path)

        print("SAVED:", out_path)
        print("ANSWER PREVIEW:")
        print(result["answer"][:1200])
        print()

    print("Готово.")


if __name__ == "__main__":
    main()