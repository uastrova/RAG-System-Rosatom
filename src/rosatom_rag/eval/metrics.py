import math
from collections import defaultdict


# метрики для сравнения ранжированных списков chunk_id.


def _top_k(predicted_ids: list[str], k: int) -> list[str]:
    return predicted_ids[:k]


def hit_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return float(any(chunk_id in relevant_ids for chunk_id in _top_k(predicted_ids, k)))


def recall_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    found = sum(1 for chunk_id in _top_k(predicted_ids, k) if chunk_id in relevant_ids)
    return found / len(relevant_ids)


def precision_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    found = sum(1 for chunk_id in _top_k(predicted_ids, k) if chunk_id in relevant_ids)
    return found / k


def mrr_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0

    for rank, chunk_id in enumerate(_top_k(predicted_ids, k), start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0

    dcg = 0.0
    for rank, chunk_id in enumerate(_top_k(predicted_ids, k), start=1):
        if chunk_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_ranked_ids(
    predicted_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int],
    mrr_k: int,
    ndcg_k: int,
) -> dict[str, float]:
    result = {}

    for k in k_values:
        result[f"hit@{k}"] = hit_at_k(predicted_ids, relevant_ids, k)
        result[f"recall@{k}"] = recall_at_k(predicted_ids, relevant_ids, k)
        result[f"precision@{k}"] = precision_at_k(predicted_ids, relevant_ids, k)

    result[f"mrr@{mrr_k}"] = mrr_at_k(predicted_ids, relevant_ids, mrr_k)
    result[f"ndcg@{ndcg_k}"] = ndcg_at_k(predicted_ids, relevant_ids, ndcg_k)
    return result


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}

    sums = defaultdict(float)
    for row in rows:
        for metric_name, value in row.items():
            sums[metric_name] += float(value)

    return {
        metric_name: value / len(rows)
        for metric_name, value in sorted(sums.items())
    }


def compute_delta(base_metrics: dict[str, float], improved_metrics: dict[str, float]) -> dict[str, float]:
    metric_names = sorted(set(base_metrics) | set(improved_metrics))
    return {
        metric_name: improved_metrics.get(metric_name, 0.0) - base_metrics.get(metric_name, 0.0)
        for metric_name in metric_names
    }
