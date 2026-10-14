"""Real-time content scoring using updated user profiles (Milestone 6).

Instead of precomputed centroids from training data, this scores candidates
against the user's CURRENT genre affinity profile + recent interaction history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.features.content_features import build_movie_feature_matrix, cosine_top_k
from src.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class RealtimeContentScorer:
    """Scores candidates using the user's live genre affinity profile.

    Builds a query vector from:
      - User's genre affinity (normalized weights per genre)
      - Recent interaction genres (recency-weighted)
      - Combines with genome tags for full content scoring
    """

    content_matrix: sp.csr_matrix      # L2-normalized movie vectors (n_movies, n_features)
    movie_index: pd.DataFrame          # movieId, row
    feature_names: dict[str, int]      # feature_name -> column index
    genre_weight: float = 0.5

    # Cached genre column indices
    _genre_cols: dict[str, int] = None
    _tag_cols: dict[int, int] = None

    def __post_init__(self):
        self._genre_cols = {
            name.removeprefix("genre:"): col
            for name, col in self.feature_names.items()
            if name.startswith("genre:")
        }
        self._tag_cols = {
            int(name.removeprefix("tag:")): col
            for name, col in self.feature_names.items()
            if name.startswith("tag:")
        }

    def _build_user_query_vector(
        self,
        genre_affinity: dict[str, float],
        recent_interaction_genres: list[set[str]] | None = None,
        recency_weights: list[float] | None = None,
    ) -> np.ndarray:
        """Build L2-normalized query vector from user profile."""
        n_features = self.content_matrix.shape[1]
        query = np.zeros(n_features, dtype=np.float32)

        # 1. Genre affinity contribution
        for genre, weight in genre_affinity.items():
            if genre in self._genre_cols:
                query[self._genre_cols[genre]] += weight * self.genre_weight

        # 2. Recent interaction genres (recency-weighted)
        if recent_interaction_genres:
            if recency_weights is None:
                # Exponential decay: most recent = 1.0, older decay
                n = len(recent_interaction_genres)
                recency_weights = [0.9 ** (n - 1 - i) for i in range(n)]

            for genres, w in zip(recent_interaction_genres, recency_weights):
                for genre in genres:
                    if genre in self._genre_cols:
                        query[self._genre_cols[genre]] += w * self.genre_weight

        # 3. Normalize
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        return query

    def score_candidates(
        self,
        genre_affinity: dict[str, float],
        candidate_ids: list[int],
        recent_interaction_genres: list[set[str]] | None = None,
        recency_weights: list[float] | None = None,
    ) -> dict[int, float]:
        """Score candidate movies against user's live profile.

        Returns {movie_id: cosine_similarity} for each candidate.
        """
        if not candidate_ids:
            return {}

        # Build query vector
        query_vec = self._build_user_query_vector(
            genre_affinity, recent_interaction_genres, recency_weights
        )

        if query_vec.sum() == 0:
            return {mid: 0.0 for mid in candidate_ids}

        # Get candidate rows
        row_map = dict(zip(self.movie_index["movieId"], self.movie_index["row"]))
        candidate_rows = []
        valid_candidates = []

        for mid in candidate_ids:
            if mid in row_map:
                candidate_rows.append(row_map[mid])
                valid_candidates.append(mid)

        if not candidate_rows:
            return {mid: 0.0 for mid in candidate_ids}

        # Batch cosine similarity
        cand_matrix = self.content_matrix[candidate_rows]  # (n_candidates, n_features)
        scores = cand_matrix @ query_vec  # (n_candidates,)

        result = {mid: float(score) for mid, score in zip(valid_candidates, scores)}

        # Fill missing with 0
        for mid in candidate_ids:
            if mid not in result:
                result[mid] = 0.0

        return result

    def get_top_k(
        self,
        genre_affinity: dict[str, float],
        candidate_ids: list[int],
        k: int = 200,
        recent_interaction_genres: list[set[str]] | None = None,
        recency_weights: list[float] | None = None,
    ) -> list[tuple[int, float]]:
        """Get top-k candidates by content score."""
        scores = self.score_candidates(
            genre_affinity, candidate_ids, recent_interaction_genres, recency_weights
        )
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_items[:k]


def create_realtime_scorer_from_data(
    genome: pd.DataFrame,
    movies: pd.DataFrame,
    genome_tags: pd.DataFrame,
    genre_weight: float = 0.5,
) -> RealtimeContentScorer:
    """Factory to create scorer from processed data tables."""
    matrix, movie_index, feature_names = build_movie_feature_matrix(
        genome, movies, genre_weight=genre_weight
    )
    return RealtimeContentScorer(
        content_matrix=matrix,
        movie_index=movie_index,
        feature_names=feature_names,
        genre_weight=genre_weight,
    )