from pathlib import Path
from functools import lru_cache
from copy import deepcopy
import argparse
import json
import re

from rosatom_rag.retrieval.hybrid_search import (
    similarity_search_hybrid_reranked,
    similarity_search_faiss_raw,
    exact_table_reference_search,
)
from rosatom_rag.retrieval.bm25_entity_search import similarity_search_bm25_entity
from rosatom_rag.retrieval.chunk_store import load_chunk_documents
from rosatom_rag.retrieval.reranker import rerank_documents
from rosatom_rag.utils import print_header


DEFAULT_QUESTIONS_PATH = Path("data/eval/eval_questions_v1.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/eval/retrieval_compare_hybrid_vs_entity_v1")

def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_compact(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-zа-я0-9]+", "", text)
    return text


def doc_hint_matches(source_file: str, source_doc_hint: str) -> bool:
    sf = normalize_text(source_file)
    hint = normalize_text(source_doc_hint)

    if hint in sf:
        return True

    sf_compact = normalize_compact(source_file)
    hint_compact = normalize_compact(source_doc_hint)

    return hint_compact in sf_compact


def normalize_ref_text(text: str) -> str:
    text = normalize_text(text)

    text = re.sub(r"\bтабл\.\s*", "таблица ", text)
    text = re.sub(r"\bтабл\s+", "таблица ", text)
    text = re.sub(r"\bтаблица\s+", "таблица ", text)

    text = re.sub(r"\bп\.\s*", "п. ", text)
    text = re.sub(r"\bпункт\s+", "п. ", text)

    text = re.sub(r"\bрис\.\s*", "рисунок ", text)
    text = re.sub(r"\bрисунок\s+", "рисунок ", text)

    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    text = re.sub(r"([а-яa-z])\s+(?=\d)", r"\1", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_ref_number(ref_hint: str):
    ref = normalize_ref_text(ref_hint)

    m = re.search(r"таблица\s+([а-яa-z]?\d+(?:\.\d+)*)", ref)
    if m:
        return "table", m.group(1)

    m = re.search(r"п\.\s*([а-яa-z]?\d+(?:\.\d+)*)", ref)
    if m:
        return "point", m.group(1)

    return "raw", ref


def ref_hint_matches_text(page_text: str, source_ref_hint: str) -> bool:
    text = normalize_ref_text(page_text)
    ref_type, ref_value = extract_ref_number(source_ref_hint)

    if ref_type == "table":
        return f"таблица {ref_value}" in text

    if ref_type == "point":
        pattern = rf"(?<!\d){re.escape(ref_value)}(?!\d)"
        return re.search(pattern, text) is not None

    return ref_value in text


# FAISS + BM25 + exact + reranker

def run_baseline_hybrid(question: str, top_k: int, args):
    return similarity_search_hybrid_reranked(
        question,
        k=top_k,
        faiss_k=args.faiss_k,
        bm25_k=args.bm25_k,
        exact_k=args.exact_k,
        max_candidates=args.max_candidates,
        reranker_max_length=args.reranker_max_length,
    )

# FAISS + exact + entity search + reranker

@lru_cache(maxsize=1)
def load_chunk_doc_map():
    docs = load_chunk_documents()
    return {doc.metadata["chunk_id"]: doc for doc in docs}


def clone_doc(doc):
    return deepcopy(doc)


def get_full_doc_by_chunk_id(chunk_id: str):
    doc_map = load_chunk_doc_map()
    doc = doc_map.get(chunk_id)
    if doc is None:
        return None
    return clone_doc(doc)


def collect_faiss_exact_entity_candidates(
    question: str,
    faiss_k: int = 20,
    exact_k: int = 10,
    entity_k: int = 10,
    rrf_k: int = 60,
    max_candidates: int = 20,
    entity_rrf_weight: float = 0.35,
):
    faiss_docs = similarity_search_faiss_raw(question, k=faiss_k)
    exact_docs = exact_table_reference_search(question, k=exact_k)
    entity_docs = similarity_search_bm25_entity(question, k=entity_k)

    merged = {}

    def ensure_item(chunk_id, full_doc):
        if chunk_id not in merged:
            merged[chunk_id] = {
                "doc": full_doc,
                "faiss_rank": None,
                "bm25_rank": None,
                "exact_rank": None,
                "entity_rank": None,
                "entity_bm25_score": None,
            }

    for rank, doc in enumerate(faiss_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        if not chunk_id:
            continue

        ensure_item(chunk_id, clone_doc(doc))
        merged[chunk_id]["faiss_rank"] = rank

    for rank, doc in enumerate(exact_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        if not chunk_id:
            continue

        ensure_item(chunk_id, clone_doc(doc))
        merged[chunk_id]["exact_rank"] = rank

        if "exact_ref_score" in doc.metadata:
            merged[chunk_id]["doc"].metadata["exact_ref_score"] = doc.metadata["exact_ref_score"]

    for rank, doc in enumerate(entity_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id")
        if not chunk_id:
            continue

        full_doc = get_full_doc_by_chunk_id(chunk_id)
        if full_doc is None:
            continue

        ensure_item(chunk_id, full_doc)
        merged[chunk_id]["entity_rank"] = rank
        merged[chunk_id]["entity_bm25_score"] = doc.metadata.get("entity_bm25_score")

        merged[chunk_id]["doc"].metadata["entity_texts"] = doc.metadata.get("entity_texts", [])
        merged[chunk_id]["doc"].metadata["entity_labels"] = doc.metadata.get("entity_labels", [])
        merged[chunk_id]["doc"].metadata["entity_bm25_score"] = doc.metadata.get("entity_bm25_score")

    scored_docs = []

    for item in merged.values():
        doc = item["doc"]

        faiss_rank = item["faiss_rank"]
        exact_rank = item["exact_rank"]
        entity_rank = item["entity_rank"]

        hybrid_score = 0.0

        if faiss_rank is not None:
            hybrid_score += 1.0 / (rrf_k + faiss_rank)

        if exact_rank is not None:
            # exact search специально даём сильный вклад,
            # потому что совпадение таблицы / пункта очень важно.
            hybrid_score += 1.0 / exact_rank

        if entity_rank is not None:
            hybrid_score += entity_rrf_weight * (1.0 / (rrf_k + entity_rank))

        doc.metadata["faiss_rank"] = faiss_rank
        doc.metadata["bm25_rank"] = None
        doc.metadata["exact_rank"] = exact_rank
        doc.metadata["entity_rank"] = entity_rank
        doc.metadata["hybrid_score"] = hybrid_score

        scored_docs.append(doc)

    scored_docs.sort(key=lambda d: d.metadata["hybrid_score"], reverse=True)
    return scored_docs[:max_candidates]


def similarity_search_faiss_exact_entity_reranked(
    question: str,
    k: int = 3,
    faiss_k: int = 20,
    exact_k: int = 10,
    entity_k: int = 10,
    rrf_k: int = 60,
    max_candidates: int = 20,
    entity_rrf_weight: float = 0.35,
    reranker_max_length: int = 1024,
):
    candidates = collect_faiss_exact_entity_candidates(
        question=question,
        faiss_k=faiss_k,
        exact_k=exact_k,
        entity_k=entity_k,
        rrf_k=rrf_k,
        max_candidates=max_candidates,
        entity_rrf_weight=entity_rrf_weight,
    )

    reranked = rerank_documents(
        question=question,
        docs=candidates,
        top_k=k,
        max_length=reranker_max_length,
    )

    return reranked


def run_faiss_exact_entity(question: str, top_k: int, args):
    return similarity_search_faiss_exact_entity_reranked(
        question,
        k=top_k,
        faiss_k=args.faiss_k,
        exact_k=args.exact_k,
        entity_k=args.entity_k,
        rrf_k=args.rrf_k,
        max_candidates=args.max_candidates,
        entity_rrf_weight=args.entity_rrf_weight,
        reranker_max_length=args.reranker_max_length,
    )


def evaluate_one_run(question_record: dict, docs):
    source_doc_hint = question_record["source_doc_hint"]
    source_ref_hint = question_record["source_ref_hint"]

    retrieved = []

    doc_hit = False
    ref_hit = False
    joint_hit = False

    first_doc_hit_rank = None
    first_ref_hit_rank = None
    first_joint_hit_rank = None

    for i, doc in enumerate(docs, start=1):
        source_file = doc.metadata.get("source_file", "")
        page_num = doc.metadata.get("page_num")
        chunk_id = doc.metadata.get("chunk_id", "")
        text = doc.page_content

        hit_doc = doc_hint_matches(source_file, source_doc_hint)
        hit_ref = ref_hint_matches_text(text, source_ref_hint)
        hit_joint = hit_doc and hit_ref

        if hit_doc and first_doc_hit_rank is None:
            first_doc_hit_rank = i

        if hit_ref and first_ref_hit_rank is None:
            first_ref_hit_rank = i

        if hit_joint and first_joint_hit_rank is None:
            first_joint_hit_rank = i

        doc_hit = doc_hit or hit_doc
        ref_hit = ref_hit or hit_ref
        joint_hit = joint_hit or hit_joint

        retrieved.append({
            "rank": i,
            "source_file": source_file,
            "page_num": page_num,
            "chunk_id": chunk_id,

            "doc_hit": hit_doc,
            "ref_hit": hit_ref,
            "joint_hit": hit_joint,

            "faiss_rank": doc.metadata.get("faiss_rank"),
            "bm25_rank": doc.metadata.get("bm25_rank"),
            "exact_rank": doc.metadata.get("exact_rank"),
            "entity_rank": doc.metadata.get("entity_rank"),

            "hybrid_score": doc.metadata.get("hybrid_score"),
            "reranker_score": doc.metadata.get("reranker_score"),
            "entity_bm25_score": doc.metadata.get("entity_bm25_score"),

            "preview": text[:400].replace("\n", "\\n"),
        })

    return {
        "doc_hit@k": doc_hit,
        "ref_hit@k": ref_hit,
        "joint_hit@k": joint_hit,

        "first_doc_hit_rank": first_doc_hit_rank,
        "first_ref_hit_rank": first_ref_hit_rank,
        "first_joint_hit_rank": first_joint_hit_rank,

        "retrieved": retrieved,
    }


def summarize(mode_to_results: dict):
    summary = {}

    for mode, results in mode_to_results.items():
        total = len(results)

        doc_hits = sum(1 for r in results if r["eval"]["doc_hit@k"])
        ref_hits = sum(1 for r in results if r["eval"]["ref_hit@k"])
        joint_hits = sum(1 for r in results if r["eval"]["joint_hit@k"])

        summary[mode] = {
            "total_questions": total,

            "doc_hit@k_count": doc_hits,
            "doc_hit@k_rate": round(doc_hits / total, 4) if total else 0.0,

            "ref_hit@k_count": ref_hits,
            "ref_hit@k_rate": round(ref_hits / total, 4) if total else 0.0,

            "joint_hit@k_count": joint_hits,
            "joint_hit@k_rate": round(joint_hits / total, 4) if total else 0.0,
        }

    if (
        "baseline_faiss_bm25_exact_reranker" in summary
        and "faiss_exact_entity_reranker" in summary
    ):
        base = summary["baseline_faiss_bm25_exact_reranker"]
        ent = summary["faiss_exact_entity_reranker"]

        summary["delta_entity_minus_baseline"] = {
            "doc_hit@k_rate_delta": round(
                ent["doc_hit@k_rate"] - base["doc_hit@k_rate"], 4
            ),
            "ref_hit@k_rate_delta": round(
                ent["ref_hit@k_rate"] - base["ref_hit@k_rate"], 4
            ),
            "joint_hit@k_rate_delta": round(
                ent["joint_hit@k_rate"] - base["joint_hit@k_rate"], 4
            ),
        }

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare retrieval pipelines: "
            "FAISS+BM25+exact+reranker vs FAISS+exact+entity+reranker"
        )
    )

    parser.add_argument(
        "--questions_file",
        type=str,
        default=str(DEFAULT_QUESTIONS_PATH),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--faiss_k", type=int, default=20)
    parser.add_argument("--bm25_k", type=int, default=20)
    parser.add_argument("--exact_k", type=int, default=10)
    parser.add_argument("--entity_k", type=int, default=10)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--max_candidates", type=int, default=20)
    parser.add_argument("--entity_rrf_weight", type=float, default=0.35)
    parser.add_argument("--reranker_max_length", type=int, default=1024)

    return parser.parse_args()


def main():
    args = parse_args()

    questions_path = Path(args.questions_file)
    output_dir = Path(args.output_dir)

    details_path = output_dir / "details.jsonl"
    summary_path = output_dir / "summary.json"

    print_header("COMPARE RETRIEVAL: HYBRID BASELINE VS ENTITY PIPELINE")

    print("QUESTIONS_PATH:", questions_path)
    print("OUTPUT_DIR    :", output_dir)
    print()
    print("PIPELINE 1: FAISS + BM25 + exact + reranker")
    print("PIPELINE 2: FAISS + exact + entity search + reranker")
    print()
    print("k                  :", args.k)
    print("faiss_k            :", args.faiss_k)
    print("bm25_k baseline    :", args.bm25_k)
    print("exact_k            :", args.exact_k)
    print("entity_k           :", args.entity_k)
    print("max_candidates     :", args.max_candidates)
    print("entity_rrf_weight  :", args.entity_rrf_weight)
    print("reranker_max_length:", args.reranker_max_length)
    print()

    questions = load_jsonl(questions_path)

    baseline_results = []
    entity_results = []
    details = []

    for qrec in questions:
        question = qrec["question"]

        print("-" * 80)
        print(f"ID {qrec.get('id')}: {question}")

        baseline_docs = run_baseline_hybrid(question, top_k=args.k, args=args)
        entity_docs = run_faiss_exact_entity(question, top_k=args.k, args=args)

        baseline_eval = evaluate_one_run(qrec, baseline_docs)
        entity_eval = evaluate_one_run(qrec, entity_docs)

        baseline_item = {
            "id": qrec.get("id"),
            "discipline": qrec.get("discipline"),
            "question": question,
            "source_doc_hint": qrec["source_doc_hint"],
            "source_ref_hint": qrec["source_ref_hint"],
            "eval": baseline_eval,
        }

        entity_item = {
            "id": qrec.get("id"),
            "discipline": qrec.get("discipline"),
            "question": question,
            "source_doc_hint": qrec["source_doc_hint"],
            "source_ref_hint": qrec["source_ref_hint"],
            "eval": entity_eval,
        }

        baseline_results.append(baseline_item)
        entity_results.append(entity_item)

        details.append({
            "id": qrec.get("id"),
            "discipline": qrec.get("discipline"),
            "question": question,
            "source_doc_hint": qrec["source_doc_hint"],
            "source_ref_hint": qrec["source_ref_hint"],

            "baseline_faiss_bm25_exact_reranker": baseline_eval,
            "faiss_exact_entity_reranker": entity_eval,
        })

        print(
            "BASELINE | doc_hit:",
            baseline_eval["doc_hit@k"],
            "| ref_hit:",
            baseline_eval["ref_hit@k"],
            "| joint_hit:",
            baseline_eval["joint_hit@k"],
        )

        print(
            "ENTITY   | doc_hit:",
            entity_eval["doc_hit@k"],
            "| ref_hit:",
            entity_eval["ref_hit@k"],
            "| joint_hit:",
            entity_eval["joint_hit@k"],
        )

    mode_to_results = {
        "baseline_faiss_bm25_exact_reranker": baseline_results,
        "faiss_exact_entity_reranker": entity_results,
    }

    summary = summarize(mode_to_results)

    save_jsonl(details, details_path)
    save_json(summary, summary_path)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("DETAILS_PATH:", details_path)
    print("SUMMARY_PATH:", summary_path)


if __name__ == "__main__":
    main()