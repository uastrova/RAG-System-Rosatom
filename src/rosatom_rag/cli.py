import argparse
import json
import subprocess
import sys
from pathlib import Path
from rosatom_rag.config import EVAL_DIR
from rosatom_rag.utils import print_header


DEFAULT_K = 4
DEFAULT_FAISS_K = 200
DEFAULT_BM25_K = 200
DEFAULT_CANDIDATES_K = 200
DEFAULT_NER_WEIGHT = 0.05
DEFAULT_QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
DEFAULT_QRELS_PATH = EVAL_DIR / "qrels.jsonl"
DEFAULT_EVAL_OUTPUT_DIR = EVAL_DIR / "final_hybrid_ner_cli"


def print_sources(docs) -> None:
    print_header("RETRIEVED SOURCES")
    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        preview = doc.page_content[:500].replace("\n", " ")

        print(f"[{i}] chunk_id: {metadata.get('chunk_id')}")
        print(f"    source_file: {metadata.get('source_file')}")
        print(f"    page_num: {metadata.get('page_num')}")
        print(f"    faiss_rank: {metadata.get('faiss_rank')}")
        print(f"    bm25_rank: {metadata.get('bm25_rank')}")
        print(f"    hybrid_score: {metadata.get('hybrid_score')}")
        print(f"    ner_score: {metadata.get('ner_score')}")
        print(f"    ner_overlap: {metadata.get('ner_overlap')}")
        print(f"    text preview: {preview}")
        print()


def answer_question(args: argparse.Namespace) -> None:
    from rosatom_rag.llm.llama_client import ask_llm_with_context, collect_sources, format_docs
    from rosatom_rag.retrieval.hybrid_ner_pipelines import retrieve_hybrid_ner

    question = args.question
    if not question:
        question = input("Введите вопрос: ").strip()

    if not question:
        raise SystemExit("Вопрос пустой. Запуск остановлен.")

    print_header("QUESTION")
    print(question)

    docs = retrieve_hybrid_ner(
        question=question,
        k=args.k,
        faiss_k=args.faiss_k,
        bm25_k=args.bm25_k,
        candidates_k=args.candidates_k,
        ner_weight=args.ner_weight,
    )

    print_sources(docs)

    context = format_docs(docs)
    answer = ask_llm_with_context(question, context)

    print_header("RAG ANSWER")
    print(answer)

    print_header("SOURCES JSON")
    print(json.dumps(collect_sources(docs), ensure_ascii=False, indent=2))


def run_eval(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "rosatom_rag.eval.compare_hybrid_vs_ner",
        "--questions",
        str(args.questions),
        "--qrels",
        str(args.qrels),
        "--k",
        str(args.eval_k),
        "--faiss-k",
        str(args.faiss_k),
        "--bm25-k",
        str(args.bm25_k),
        "--candidates-k",
        str(args.candidates_k),
        "--ner-weight",
        str(args.ner_weight),
        "--output-dir",
        str(args.output_dir),
    ]

    print_header("RUN PIPELINE COMPARISON")
    print(" ".join(command))
    subprocess.run(command, check=True)

    metrics_path = Path(args.output_dir) / "metrics" / "hybrid_vs_hybrid_ner.csv"
    if metrics_path.exists():
        print_header("METRICS CSV")
        print(metrics_path.read_text(encoding="utf-8"))
    else:
        print(f"Метрики не найдены: {metrics_path}")


def interactive_menu(args: argparse.Namespace) -> None:
    while True:
        print_header("ROSATOM RAG MENU")
        print("1 - Получить ответ от RAG системы на свой вопрос")
        print("2 - Запустить сравнение пайплайнов без NER и с NER")
        print("0 - Выход")
        choice = input("Выберите пункт: ").strip()

        if choice == "1":
            question = input("Введите вопрос: ").strip()
            ask_args = argparse.Namespace(
                question=question,
                k=args.k,
                faiss_k=args.faiss_k,
                bm25_k=args.bm25_k,
                candidates_k=args.candidates_k,
                ner_weight=args.ner_weight,
            )
            answer_question(ask_args)
        elif choice == "2":
            eval_args = argparse.Namespace(
                questions=args.questions,
                qrels=args.qrels,
                output_dir=args.output_dir,
                eval_k=args.eval_k,
                faiss_k=args.faiss_k,
                bm25_k=args.bm25_k,
                candidates_k=args.candidates_k,
                ner_weight=args.ner_weight,
            )
            run_eval(eval_args)
        elif choice == "0":
            print("Выход.")
            return
        else:
            print("Неизвестный пункт меню. Выберите 1, 2 или 0.")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--faiss-k", type=int, default=DEFAULT_FAISS_K)
    parser.add_argument("--bm25-k", type=int, default=DEFAULT_BM25_K)
    parser.add_argument("--candidates-k", type=int, default=DEFAULT_CANDIDATES_K)
    parser.add_argument("--ner-weight", type=float, default=DEFAULT_NER_WEIGHT)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Удобный запуск Rosatom RAG из терминала.")
    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser("ask", help="Задать вопрос RAG системе.")
    ask_parser.add_argument("--question", type=str, help="Вопрос пользователя.")
    ask_parser.add_argument("--k", type=int, default=DEFAULT_K, help="Сколько чанков передать в LLM.")
    add_common_args(ask_parser)

    eval_parser = subparsers.add_parser("eval", help="Сравнить hybrid и hybrid_ner.")
    eval_parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    eval_parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS_PATH)
    eval_parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVAL_OUTPUT_DIR)
    eval_parser.add_argument("--eval-k", type=int, default=10)
    add_common_args(eval_parser)

    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVAL_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--eval-k", type=int, default=10)
    add_common_args(parser)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "ask":
        answer_question(args)
    elif args.command == "eval":
        run_eval(args)
    else:
        interactive_menu(args)


if __name__ == "__main__":
    main()
