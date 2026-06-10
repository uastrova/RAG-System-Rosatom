import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from rosatom_rag.config import EVAL_DIR
from rosatom_rag.eval.metrics import aggregate_metric_rows, compute_delta, evaluate_ranked_ids
from rosatom_rag.utils import print_header


PIPELINE_HYBRID = "hybrid"
PIPELINE_HYBRID_NER = "hybrid_ner"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            record["_line_num"] = line_num
            records.append(record)

    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_questions(path: Path) -> list[dict]:
    records = load_jsonl(path)
    questions = []

    for record in records:
        question_id = record.get("question_id") or record.get("id")
        question = record.get("question")

        if not question_id:
            raise ValueError(f"Missing question_id at {path}:{record['_line_num']}")
        if not question:
            raise ValueError(f"Missing question at {path}:{record['_line_num']}")

        questions.append({
            "question_id": str(question_id),
            "question": str(question),
        })

    return questions


def load_qrels(path: Path) -> dict[str, set[str]]:
    records = load_jsonl(path)
    qrels = {}

    for record in records:
        question_id = record.get("question_id") or record.get("id")
        relevant_chunk_ids = (
            record.get("relevant_chunk_ids")
            or record.get("relevant_chunks")
            or record.get("chunk_ids")
        )

        if not question_id:
            raise ValueError(f"Missing question_id at {path}:{record['_line_num']}")
        if not relevant_chunk_ids:
            raise ValueError(f"Missing relevant_chunk_ids at {path}:{record['_line_num']}")

        qrels[str(question_id)] = {str(chunk_id) for chunk_id in relevant_chunk_ids}

    return qrels


def doc_to_run_item(doc, rank: int) -> dict:
    metadata = doc.metadata
    score = (
        metadata.get("hybrid_ner_score")
        if metadata.get("hybrid_ner_score") is not None
        else metadata.get("hybrid_score")
    )

    return {
        "rank": rank,
        "chunk_id": metadata.get("chunk_id"),
        "source_file": metadata.get("source_file"),
        "page_num": metadata.get("page_num"),
        "chunk_type": metadata.get("chunk_type"),
        "score": score,
        "faiss_rank": metadata.get("faiss_rank"),
        "bm25_rank": metadata.get("bm25_rank"),
        "hybrid_score": metadata.get("hybrid_score"),
        "hybrid_ner_score": metadata.get("hybrid_ner_score"),
        "final_rank": metadata.get("final_rank", rank),
        "ner_score": metadata.get("ner_score"),
        "ner_overlap": metadata.get("ner_overlap"),
        "query_ner_entities": metadata.get("query_ner_entities"),
    }


def docs_to_run_record(question_id: str, question: str, pipeline: str, docs) -> dict:
    return {
        "question_id": question_id,
        "question": question,
        "pipeline": pipeline,
        "ranked_chunks": [
            doc_to_run_item(doc, rank)
            for rank, doc in enumerate(docs, start=1)
        ],
    }


def run_record_to_chunk_ids(run_record: dict) -> list[str]:
    return [
        str(item["chunk_id"])
        for item in run_record["ranked_chunks"]
        if item.get("chunk_id") is not None
    ]


def evaluate_pipeline_run(
    run_records: list[dict],
    qrels: dict[str, set[str]],
    k_values: list[int],
    mrr_k: int,
    ndcg_k: int,
) -> tuple[list[dict], dict[str, float]]:
    per_question = []
    metric_rows = []

    for run_record in run_records:
        question_id = run_record["question_id"]
        relevant_ids = qrels.get(question_id, set())
        predicted_ids = run_record_to_chunk_ids(run_record)
        metrics = evaluate_ranked_ids(
            predicted_ids=predicted_ids,
            relevant_ids=relevant_ids,
            k_values=k_values,
            mrr_k=mrr_k,
            ndcg_k=ndcg_k,
        )

        metric_rows.append(metrics)
        per_question.append({
            "question_id": question_id,
            "question": run_record["question"],
            "pipeline": run_record["pipeline"],
            "relevant_chunk_ids": sorted(relevant_ids),
            "top_chunk_ids": predicted_ids,
            "metrics": metrics,
        })

    return per_question, aggregate_metric_rows(metric_rows)


def save_metrics_csv(pipeline_metrics: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = sorted({
        metric_name
        for metrics in pipeline_metrics.values()
        for metric_name in metrics
    })

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pipeline", *metric_names])
        writer.writeheader()

        for pipeline, metrics in pipeline_metrics.items():
            row = {"pipeline": pipeline}
            row.update(metrics)
            writer.writerow(row)


def parse_k_values(raw_value: str) -> list[int]:
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if item:
            values.append(int(item))

    if not values:
        raise ValueError("At least one k value is required")

    return sorted(set(values))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare retrieval quality for FAISS+BM25 vs FAISS+BM25+NER."
    )
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--qrels", type=Path, default=EVAL_DIR / "qrels.jsonl")
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k-values", type=str, default="1,3,5,10")
    parser.add_argument("--faiss-k", type=int, default=100)
    parser.add_argument("--bm25-k", type=int, default=100)
    parser.add_argument("--candidates-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--ner-weight", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    k_values = parse_k_values(args.k_values)
    mrr_k = args.k
    ndcg_k = args.k

    print_header("COMPARE FAISS+BM25 VS FAISS+BM25+NER")
    print("QUESTIONS:", args.questions)
    print("QRELS    :", args.qrels)
    print("OUTPUT   :", args.output_dir)
    print("K        :", args.k)
    print("K_VALUES :", k_values)
    print("FAISS_K  :", args.faiss_k)
    print("BM25_K   :", args.bm25_k)
    print("CANDIDATES_K:", args.candidates_k)
    print("RRF_K    :", args.rrf_k)
    print("NER_WEIGHT:", args.ner_weight)

    from rosatom_rag.retrieval.hybrid_ner_pipelines import retrieve_hybrid, retrieve_hybrid_ner

    questions = load_questions(args.questions)
    qrels = load_qrels(args.qrels)

    hybrid_run_records = []
    hybrid_ner_run_records = []

    for idx, item in enumerate(questions, start=1):
        question_id = item["question_id"]
        question = item["question"]
        print(f"[{idx}/{len(questions)}] {question_id}: {question}")

        hybrid_docs = retrieve_hybrid(
            question=question,
            k=args.k,
            faiss_k=args.faiss_k,
            bm25_k=args.bm25_k,
            rrf_k=args.rrf_k,
        )
        hybrid_ner_docs = retrieve_hybrid_ner(
            question=question,
            k=args.k,
            candidates_k=args.candidates_k,
            faiss_k=args.faiss_k,
            bm25_k=args.bm25_k,
            rrf_k=args.rrf_k,
            ner_weight=args.ner_weight,
        )

        hybrid_run_records.append(docs_to_run_record(
            question_id=question_id,
            question=question,
            pipeline=PIPELINE_HYBRID,
            docs=hybrid_docs,
        ))
        hybrid_ner_run_records.append(docs_to_run_record(
            question_id=question_id,
            question=question,
            pipeline=PIPELINE_HYBRID_NER,
            docs=hybrid_ner_docs,
        ))

    runs_dir = args.output_dir / "runs"
    metrics_dir = args.output_dir / "metrics"

    save_jsonl(hybrid_run_records, runs_dir / "hybrid_run.jsonl")
    save_jsonl(hybrid_ner_run_records, runs_dir / "hybrid_ner_run.jsonl")

    hybrid_per_question, hybrid_metrics = evaluate_pipeline_run(
        run_records=hybrid_run_records,
        qrels=qrels,
        k_values=k_values,
        mrr_k=mrr_k,
        ndcg_k=ndcg_k,
    )
    hybrid_ner_per_question, hybrid_ner_metrics = evaluate_pipeline_run(
        run_records=hybrid_ner_run_records,
        qrels=qrels,
        k_values=k_values,
        mrr_k=mrr_k,
        ndcg_k=ndcg_k,
    )

    per_question_records = hybrid_per_question + hybrid_ner_per_question
    save_jsonl(per_question_records, metrics_dir / "per_question_hybrid_vs_hybrid_ner.jsonl")

    pipeline_metrics = {
        PIPELINE_HYBRID: hybrid_metrics,
        PIPELINE_HYBRID_NER: hybrid_ner_metrics,
    }
    summary = {
        "comparison": "hybrid_vs_hybrid_ner",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "questions_count": len(questions),
        "qrels_count": len(qrels),
        "params": {
            "k": args.k,
            "k_values": k_values,
            "faiss_k": args.faiss_k,
            "bm25_k": args.bm25_k,
            "candidates_k": args.candidates_k,
            "rrf_k": args.rrf_k,
            "ner_weight": args.ner_weight,
        },
        "pipelines": pipeline_metrics,
        "delta": compute_delta(hybrid_metrics, hybrid_ner_metrics),
        "outputs": {
            "hybrid_run": str(runs_dir / "hybrid_run.jsonl"),
            "hybrid_ner_run": str(runs_dir / "hybrid_ner_run.jsonl"),
            "per_question": str(metrics_dir / "per_question_hybrid_vs_hybrid_ner.jsonl"),
            "metrics_json": str(metrics_dir / "hybrid_vs_hybrid_ner.json"),
            "metrics_csv": str(metrics_dir / "hybrid_vs_hybrid_ner.csv"),
        },
    }

    save_json(summary, metrics_dir / "hybrid_vs_hybrid_ner.json")
    save_metrics_csv(pipeline_metrics, metrics_dir / "hybrid_vs_hybrid_ner.csv")

    print()
    print("Saved runs to:", runs_dir)
    print("Saved metrics to:", metrics_dir)
    print("Done.")


if __name__ == "__main__":
    main()
