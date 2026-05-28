from pathlib import Path
import json

from rosatom_rag.retrieval.search import similarity_search
from rosatom_rag.llm.llama_client import format_docs, collect_sources, ask_llm_with_context
from rosatom_rag.utils import print_header


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "rag_baseline_v2"


TEST_QUESTIONS = [
    "Какие требования предъявляются к заземлению?",
    "Какие требования ПУЭ относятся к электроустановкам?",
    "Что указано в таблице 1.7.7?",
    "Какие требования предъявляются к защитному проводнику?",
    "Что сказано в ПУЭ про заземляющее устройство?",
]


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one_question(question: str, top_k: int, out_path: Path):
    retrieved_docs = similarity_search(question, k=top_k)
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

    print_header("RAG BASELINE EVAL")
    print("OUTPUT_DIR:", OUTPUT_DIR)

    for i, question in enumerate(TEST_QUESTIONS, start=1):
        out_path = OUTPUT_DIR / f"test_{i:03d}.json"
        print("-" * 80)
        print("QUESTION:", question)

        result = run_one_question(question, top_k=4, out_path=out_path)

        print("SAVED:", out_path)
        print("ANSWER PREVIEW:")
        print(result["answer"][:1200])
        print()

    print("Готово.")


if __name__ == "__main__":
    main()