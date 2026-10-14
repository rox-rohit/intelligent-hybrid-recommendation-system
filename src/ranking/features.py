"""Ranking-stage feature extraction (spec section 5, Milestone 5).

For each (user, candidate_item) pair in the candidate set, build a
feature vector that captures retrieval scores, user activity, item
popularity, content overlap, and temporal signals. The ranker learns
which combination of these signals predicts real engagement — a richer
decision surface than the fixed-weight hybrid blend.

Feature contract:
  FEATURE_COLUMNS lists every feature in a fixed order. Every builder
  method returns a DataFrame with exactly these columns (plus userId,
  movieId for joining), so the ranker always sees the same schema at
  train and inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

log = get_logger(__name__)

# Canonical feature order — never reorder; new features append at the end.
FEATURE_COLUMNS: list[str] = [
    "cf_als_score",
    "cf_knn_score",
    "content_score",
    "pop_score",
    "freshness",
    "user_n_ratings",
    "item_n_ratings",
    "item_avg_rating",
    "genre_overlap",
    "log_item_pop",
]


def _minmax_dict(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalize a {item: score} dict to [0, 1]. Empty-safe."""
    if not scores:
        return {}
    vals = np.fromiter(scores.values(), dtype=np.float64)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


@dataclass
class UserItemStats:
    """Pre-aggregated statistics needed for feature building.

    Computed once from the training split, then reused across all users.
    """

    # Per-item stats from training data
    item_n_ratings: dict[int, int] = field(default_factory=dict)
    item_avg_rating: dict[int, float] = field(default_factory=dict)

    # Per-user stats from training data
    user_n_ratings: dict[int, int] = field(default_factory=dict)

    # Per-user genre profile: {userId: {genre: fraction_of_user_ratings}}
    user_genre_profile: dict[int, dict[str, float]] = field(default_factory=dict)

    # Per-item genres: {movieId: set[str]}
    item_genres: dict[int, set[str]] = field(default_factory=dict)

    # Per-item freshness: {movieId: float in [0, 1]}
    item_freshness: dict[int, float] = field(default_factory=dict)

    # Per-item popularity: {movieId: float in [0, 1]}
    item_pop_score: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_train(
        cls,
        train: pd.DataFrame,
        movies: pd.DataFrame,
        pop_scores: pd.Series | dict,
        freshness: pd.Series | dict,
    ) -> "UserItemStats":
        """Build stats from training data.

        train: training ratings DataFrame (userId, movieId, rating).
        movies: movies DataFrame with movieId, genres_list.
        pop_scores: {movieId: float} Bayesian popularity in [0, 1].
        freshness: {movieId: float} release-year recency in [0, 1].
        """
        # Item stats
        item_stats = train.groupby("movieId")["rating"].agg(["count", "mean"])
        item_n = dict(zip(item_stats.index, item_stats["count"].astype(int)))
        item_avg = dict(zip(item_stats.index, item_stats["mean"].astype(float)))

        # User stats
        user_stats = train.groupby("userId")["rating"].count()
        user_n = dict(zip(user_stats.index, user_stats.values.astype(int)))

        # Item genres — genres_list arrives from parquet as a numpy array
        # (or None/NaN for missing), so truthiness checks are ambiguous.
        item_genres_map: dict[int, set[str]] = {}
        for mid, gl in zip(movies["movieId"], movies["genres_list"]):
            try:
                item_genres_map[int(mid)] = set(gl) if gl is not None and len(gl) > 0 else set()
            except TypeError:
                # NaN or unhashable content -> no genres
                item_genres_map[int(mid)] = set()

        # User genre profile: fraction of each genre in user's train history
        user_genre_profile: dict[int, dict[str, float]] = {}
        for uid, grp in train.groupby("userId"):
            genre_counts: dict[str, int] = {}
            total = 0
            for mid in grp["movieId"]:
                for g in item_genres_map.get(int(mid), set()):
                    genre_counts[g] = genre_counts.get(g, 0) + 1
                    total += 1
            if total > 0:
                user_genre_profile[int(uid)] = {
                    g: c / total for g, c in genre_counts.items()
                }
            else:
                user_genre_profile[int(uid)] = {}

        pop_dict = dict(pop_scores) if not isinstance(pop_scores, dict) else pop_scores
        fresh_dict = dict(freshness) if not isinstance(freshness, dict) else freshness

        return cls(
            item_n_ratings=item_n,
            item_avg_rating=item_avg,
            user_n_ratings=user_n,
            user_genre_profile=user_genre_profile,
            item_genres=item_genres_map,
            item_freshness=fresh_dict,
            item_pop_score=pop_dict,
        )


def build_features(
    user_ids: list[int],
    candidate_items: dict[int, list[int]],
    als_scores: dict[int, dict[int, float]],
    knn_scores: dict[int, dict[int, float]],
    content_scores: dict[int, dict[int, float]],
    stats: UserItemStats,
) -> pd.DataFrame:
    """Build the ranking feature matrix for a set of (user, candidate) pairs.

    Returns a DataFrame with columns [userId, movieId] + FEATURE_COLUMNS.
    One row per (user, candidate_item) pair. Missing component scores
    default to 0.0 — this is intentional: the ranker learns that a
    missing CF score for a tail item is a signal, not a bug.
    """
    rows: list[dict] = []

    for uid in user_ids:
        cands = candidate_items.get(uid, [])
        if not cands:
            continue

        # Normalize retrieval scores per user (same scale as the hybrid)
        als_norm = _minmax_dict(als_scores.get(uid, {}))
        knn_norm = _minmax_dict(knn_scores.get(uid, {}))
        ct_norm = _minmax_dict(content_scores.get(uid, {}))
        u_genres = stats.user_genre_profile.get(uid, {})
        u_n = stats.user_n_ratings.get(uid, 0)

        for item in cands:
            i_genres = stats.item_genres.get(item, set())
            # Genre overlap: fraction of the item's genres in the user's profile
            if i_genres and u_genres:
                overlap = sum(1 for g in i_genres if g in u_genres) / len(i_genres)
            else:
                overlap = 0.0

            i_n = stats.item_n_ratings.get(item, 0)

            rows.append({
                "userId": uid,
                "movieId": item,
                "cf_als_score": als_norm.get(item, 0.0),
                "cf_knn_score": knn_norm.get(item, 0.0),
                "content_score": ct_norm.get(item, 0.0),
                "pop_score": stats.item_pop_score.get(item, 0.0),
                "freshness": stats.item_freshness.get(item, 0.0),
                "user_n_ratings": u_n,
                "item_n_ratings": i_n,
                "item_avg_rating": stats.item_avg_rating.get(item, 0.0),
                "genre_overlap": overlap,
                "log_item_pop": float(np.log1p(i_n)),
            })

    if not rows:
        return pd.DataFrame(columns=["userId", "movieId"] + FEATURE_COLUMNS)

    df = pd.DataFrame(rows)
    # Guarantee column order and presence
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[["userId", "movieId"] + FEATURE_COLUMNS]


def label_candidates(
    features_df: pd.DataFrame,
    ground_truth: dict[int, set[int]],
    relevance_threshold: float = 3.5,
) -> np.ndarray:
    """Binary labels for the feature matrix: 1 if the (user, item) pair
    appears in the ground truth, 0 otherwise.

    ground_truth: {userId: set(movieIds)} of held-out relevant items
                  (items with rating >= relevance_threshold in the
                  held-out split).
    """
    labels = np.zeros(len(features_df), dtype=np.int32)
    for i, (uid, mid) in enumerate(
        zip(features_df["userId"], features_df["movieId"])
    ):
        if mid in ground_truth.get(uid, set()):
            labels[i] = 1
    return labels
