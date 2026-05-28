from pathlib import Path
import json
import re

from rosatom_rag.retrieval.hybrid_search import similarity_search_hybrid_reranked
from rosatom_rag.retrieval.hybrid_search_ner import similarity_search_hybrid_ner_reranked
from rosatom_rag.utils import print_header


QUESTIONS_PATH = Path("data/eval/eval_questions_v1.jsonl")
OUTPUT_DIR = Path("data/eval/retrieval_compare_v1")
DETAILS_PATH = OUTPUT_DIR / "details.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


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
        # Ищем сам номер пункта как структурную ссылку
        pattern = rf"(?<!\d){re.escape(ref_value)}(?!\d)"
        return re.search(pattern, text) is not None

    return ref_value in text


def run_baseline(question: str, top_k: int = 3):
    return similarity_search_hybrid_reranked(
        question,
        k=top_k,
        faiss_k=20,
        bm25_k=20,
        exact_k=10,
        max_candidates=20,
        reranker_max_length=1024,
    )


def run_ner(question: str, top_k: int = 3):
    return similarity_search_hybrid_ner_reranked(
        question,
        k=top_k,
        faiss_k=20,
        bm25_k=20,
        exact_k=10,
        entity_k=5,
        max_candidates=20,
        reranker_max_length=1024,
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

    return summary


def main():
    print_header("COMPARE RETRIEVAL: BASELINE VS NER")
    print("QUESTIONS_PATH:", QUESTIONS_PATH)
    print("OUTPUT_DIR    :", OUTPUT_DIR)

    questions = load_jsonl(QUESTIONS_PATH)

    baseline_results = []
    ner_results = []
    details = []

    for qrec in questions:
        question = qrec["question"]

        print("-" * 80)
        print(f"ID {qrec['id']}: {question}")

        baseline_docs = run_baseline(question, top_k=3)
        ner_docs = run_ner(question, top_k=3)

        baseline_eval = evaluate_one_run(qrec, baseline_docs)
        ner_eval = evaluate_one_run(qrec, ner_docs)

        baseline_item = {
            "id": qrec["id"],
            "discipline": qrec["discipline"],
            "question": question,
            "source_doc_hint": qrec["source_doc_hint"],
            "source_ref_hint": qrec["source_ref_hint"],
            "eval": baseline_eval,
        }

        ner_item = {
            "id": qrec["id"],
            "discipline": qrec["discipline"],
            "question": question,
            "source_doc_hint": qrec["source_doc_hint"],
            "source_ref_hint": qrec["source_ref_hint"],
            "eval": ner_eval,
        }

        baseline_results.append(baseline_item)
        ner_results.append(ner_item)

        details.append({
            "id": qrec["id"],
            "discipline": qrec["discipline"],
            "question": question,
            "source_doc_hint": qrec["source_doc_hint"],
            "source_ref_hint": qrec["source_ref_hint"],
            "baseline": baseline_eval,
            "ner": ner_eval,
        })

        print("BASELINE  | doc_hit:", baseline_eval["doc_hit@k"],
              "| ref_hit:", baseline_eval["ref_hit@k"],
              "| joint_hit:", baseline_eval["joint_hit@k"])
        print("NER       | doc_hit:", ner_eval["doc_hit@k"],
              "| ref_hit:", ner_eval["ref_hit@k"],
              "| joint_hit:", ner_eval["joint_hit@k"])

    mode_to_results = {
        "baseline": baseline_results,
        "ner": ner_results,
    }

    summary = summarize(mode_to_results)

    save_jsonl(details, DETAILS_PATH)
    save_json(summary, SUMMARY_PATH)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("DETAILS_PATH:", DETAILS_PATH)
    print("SUMMARY_PATH:", SUMMARY_PATH)


if __name__ == "__main__":
    main()