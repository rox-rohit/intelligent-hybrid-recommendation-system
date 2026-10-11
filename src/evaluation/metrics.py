"""Recommender-system evaluation metrics.

All functions share one convention:
  recommendations: (n_users, k) array of *ranked* item ids (best first)
  ground_truth:   {user_id: set(item_ids)} of held-out relevant items

Metrics (spec section 17): Precision@K, Recall@K, HitRate@K, NDCG@K,
MAP@K. Coverage@K (share of catalog reachable) measures popularity-bias
resistance — a cheap, honest addition.

These are recommender metrics, deliberately NOT accuracy: with 62k movies,
a 0.1% Precision@10 is already strong signal.
"""

from __future__ import annotations

import numpy as np


def precision_at_k(recs: np.ndarray, truth: dict[int, set[int]], k: int | None = None) -> float:
    """Fraction of recommended items that are relevant, averaged over users."""
    k = k or recs.shape[1]
    hits = [
        len(set(recs[i, :k]).intersection(truth.get(user_id, set())))
        for i, user_id in enumerate(_user_order(truth, recs))
    ]
    return float(np.mean([h / k for h in hits])) if hits else 0.0


def recall_at_k(recs: np.ndarray, truth: dict[int, set[int]], k: int | None = None) -> float:
    """Fraction of each user's relevant items that were recommended."""
    k = k or recs.shape[1]
    recalls = []
    for i, user_id in enumerate(_user_order(truth, recs)):
        t = truth.get(user_id, set())
        if not t:
            continue
        hits = len(set(recs[i, :k]).intersection(t))
        recalls.append(hits / len(t))
    return float(np.mean(recalls)) if recalls else 0.0


def hit_rate_at_k(recs: np.ndarray, truth: dict[int, set[int]], k: int | None = None) -> float:
    """Share of users with at least one relevant recommendation."""
    k = k or recs.shape[1]
    got_hit = []
    for i, user_id in enumerate(_user_order(truth, recs)):
        t = truth.get(user_id, set())
        got_hit.append(bool(set(recs[i, :k]).intersection(t)))
    return float(np.mean(got_hit)) if got_hit else 0.0


def ndcg_at_k(recs: np.ndarray, truth: dict[int, set[int]], k: int | None = None) -> float:
    """NDCG@K: ranking quality with logarithmic position discount.

    DCG  = sum over positions p of  rel(i_p) / log2(p + 1)
    IDCG = DCG of the ideal (all relevant items first, capped at k)
    """
    k = k or recs.shape[1]
    ndcgs = []
    for i, user_id in enumerate(_user_order(truth, recs)):
        t = truth.get(user_id, set())
        if not t:
            continue
        gains = np.array(
            [1.0 if item in t else 0.0 for item in recs[i, :k]]
        )
        discounts = np.log2(np.arange(len(gains)) + 2)
        dcg = float((gains / discounts).sum())
        ideal_hits = min(len(t), k)
        idcg = float((np.ones(ideal_hits) / np.log2(np.arange(ideal_hits) + 2)).sum())
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def map_at_k(recs: np.ndarray, truth: dict[int, set[int]], k: int | None = None) -> float:
    """Mean Average Precision@K over users."""
    k = k or recs.shape[1]
    aps = []
    for i, user_id in enumerate(_user_order(truth, recs)):
        t = truth.get(user_id, set())
        if not t:
            continue
        hits = 0
        precisions = []
        for rank, item in enumerate(recs[i, :k], start=1):
            if item in t:
                hits += 1
                precisions.append(hits / rank)
        aps.append(float(np.mean(precisions)) if precisions else 0.0)
    return float(np.mean(aps)) if aps else 0.0


def catalog_coverage(recs: np.ndarray, n_items: int) -> float:
    """Share of the catalog that ever appears in the recommendations."""
    unique = len(np.unique(recs))
    return float(unique / n_items) if n_items else 0.0


def _user_order(truth: dict[int, set[int]], recs: np.ndarray) -> list[int]:
    """User ids aligned with recs rows. Prefers truth's key order when it
    matches row count (callers build truth and recs in the same order);
    falls back to positional ids for synthetic tests."""
    keys = list(truth.keys())
    if len(keys) == recs.shape[0]:
        return keys
    return list(range(recs.shape[0]))


def evaluate_all(
    recs: np.ndarray, truth: dict[int, set[int]], k: int = 10, n_items: int | None = None
) -> dict[str, float]:
    """Compute the full metric suite — one call per model for comparison tables."""
    out = {
        f"precision@{k}": precision_at_k(recs, truth, k),
        f"recall@{k}": recall_at_k(recs, truth, k),
        f"hit_rate@{k}": hit_rate_at_k(recs, truth, k),
        f"ndcg@{k}": ndcg_at_k(recs, truth, k),
        f"map@{k}": map_at_k(recs, truth, k),
    }
    if n_items:
        out["coverage"] = catalog_coverage(recs, n_items)
    return out
