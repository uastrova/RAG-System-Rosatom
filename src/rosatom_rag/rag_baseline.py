from pathlib import Path
import json
import argparse

from rosatom_rag.retrieval.search import similarity_search
from rosatom_rag.llm.llama_client import format_docs, ask_llm_with_context, collect_sources
from rosatom_rag.utils import print_header


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "rag_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--save_name", type=str, default="last_run.json")
    args = parser.parse_args()

    print_header("BASELINE RAG")
    print("QUESTION:", args.question)
    print("TOP_K:", args.k)

    retrieved_docs = similarity_search(args.question, k=args.k)

    print()
    print_header("RETRIEVED DOCS")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"[{i}] file={doc.metadata.get('source_file')} page={doc.metadata.get('page_num')} chunk={doc.metadata.get('chunk_id')}")
        print(doc.page_content[:800])
        print("-" * 80)

    context = format_docs(retrieved_docs)
    answer = ask_llm_with_context(args.question, context)
    sources = collect_sources(retrieved_docs)

    print()
    print_header("ANSWER")
    print(answer)

    result = {
        "question": args.question,
        "top_k": args.k,
        "answer": answer,
        "sources": sources,
    }

    output_path = OUTPUT_DIR / args.save_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print()
    print("Сохранено в:", output_path)


if __name__ == "__main__":
    main()