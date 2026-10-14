"""Weighted hybrid recommender (spec section 9).

score(u, i) = w_cf * cf_score(u, i)
            + w_content * content_score(u, i)
            + w_pop * pop_score(i)

Each component is min-max normalized per user *before* blending, so a
confident CF score of 137 and a popularity score of 0.42 contribute on a
comparable scale. Weights are tuned by grid search on validation NDCG@10
(scripts/train_hybrid.py), never hand-waved.

Design notes:
  - CF and content produce scores for *different* item sets; the hybrid
    scores the union, with missing components contributing 0 (this is
    where cold-start fallbacks plug in later).
  - Everything is computed over candidate sets, not the whole catalog —
    production hybrid rankers retrieve candidates cheaply, then blend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

log = get_logger(__name__)


def minmax(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalize a {item: score} dict to [0, 1]. Empty-safe."""
    if not scores:
        return {}
    vals = np.fromiter(scores.values(), dtype=np.float64)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


@dataclass
class HybridWeights:
    cf: float = 0.5
    content: float = 0.3
    popularity: float = 0.2

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.cf, self.content, self.popularity)

    def normalized(self) -> "HybridWeights":
        total = self.cf + self.content + self.popularity
        if total <= 0:
            return HybridWeights(1 / 3, 1 / 3, 1 / 3)
        return HybridWeights(self.cf / total, self.content / total, self.popularity / total)


class HybridRecommender:
    """Blends CF, content, and popularity scores into one ranked list.

    Components are injected as callables mapping user_id -> {item_id: raw
    score}; this keeps the hybrid model-agnostic (ItemKNN, ALS, or a
    future model slot in without code changes).
    """

    def __init__(
        self,
        cf_scorer=None,
        content_scorer=None,
        popularity_scorer=None,
        weights: HybridWeights | None = None,
    ):
        self.cf_scorer = cf_scorer
        self.content_scorer = content_scorer
        self.popularity_scorer = popularity_scorer
        self.weights = (weights or HybridWeights()).normalized()

    def score_items(
        self,
        user_id: int,
        candidate_items: list[int] | set[int] | None = None,
        seen_items: set[int] | None = None,
    ) -> dict[int, float]:
        """Blend raw component scores for one user.

        candidate_items: if given, only these items are scored (and each
        component may return scores for a subset of them). If None, the
        union of all components' item sets is scored.
        seen_items are excluded from the output.
        """
        w = self.weights

        cf_raw = self.cf_scorer(user_id) if self.cf_scorer else {}
        ct_raw = self.content_scorer(user_id) if self.content_scorer else {}
        pop_raw = self.popularity_scorer(user_id) if self.popularity_scorer else {}

        cf = minmax(cf_raw)
        ct = minmax(ct_raw)
        pop = minmax(pop_raw)

        candidates: set[int]
        if candidate_items is not None:
            candidates = set(candidate_items)
        else:
            candidates = set(cf) | set(ct) | set(pop)

        if seen_items:
            candidates -= set(seen_items)

        blended: dict[int, float] = {}
        for item in candidates:
            blended[item] = (
                w.cf * cf.get(item, 0.0)
                + w.content * ct.get(item, 0.0)
                + w.popularity * pop.get(item, 0.0)
            )
        return blended

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        candidate_items: list[int] | set[int] | None = None,
        seen_items: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Top-k blended recommendations, best first."""
        blended = self.score_items(user_id, candidate_items, seen_items)
        ranked = sorted(blended.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:k]

    def explain(self, user_id: int, item_id: int) -> dict[str, float | str]:
        """Per-component contribution breakdown for one recommendation.

        Powers the "why am I seeing this?" UI (spec section 10): the
        dominant component names the reason ("users like you", "similar
        to X", "popular right now").
        """
        w = self.weights
        cf_raw = self.cf_scorer(user_id) if self.cf_scorer else {}
        ct_raw = self.content_scorer(user_id) if self.content_scorer else {}
        pop_raw = self.popularity_scorer(user_id) if self.popularity_scorer else {}
        cf, ct, pop = minmax(cf_raw), minmax(ct_raw), minmax(pop_raw)

        contrib = {
            "collaborative": w.cf * cf.get(item_id, 0.0),
            "content": w.content * ct.get(item_id, 0.0),
            "popularity": w.popularity * pop.get(item_id, 0.0),
        }
        dominant = max(contrib, key=contrib.get) if contrib else "none"
        reasons = {
            "collaborative": "users with similar taste watched this",
            "content": "matches the genres and themes you like",
            "popularity": "highly rated and popular overall",
        }
        return {
            "contributions": contrib,
            "dominant": dominant,
            "reason": reasons.get(dominant, "recommended for you"),
        }
