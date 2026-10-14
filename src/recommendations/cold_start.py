"""Cold-start handling for new users and new movies (Milestone 7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.recommendations.realtime_content import RealtimeContentScorer, create_realtime_scorer_from_data
from src.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class ColdStartConfig:
    min_ratings_for_cf: int = 10        # Min ratings before using CF
    min_ratings_for_hybrid: int = 5     # Min ratings before using hybrid
    onboarding_genres_required: int = 3 # Min genres to select in onboarding
    onboarding_movies_required: int = 5 # Min movies to rate in onboarding


class ColdStartHandler:
    """Handles recommendations for users/items with insufficient history."""

    def __init__(
        self,
        content_scorer: RealtimeContentScorer,
        popular_movies: list[int],      # Pre-ranked popular movie IDs
        movie_genres: dict[int, set[str]],
        config: ColdStartConfig | None = None,
    ):
        self.content_scorer = content_scorer
        self.popular_movies = popular_movies
        self.movie_genres = movie_genres
        self.config = config or ColdStartConfig()

    def get_recommendations(
        self,
        user_id: int,
        n_interactions: int,
        genre_affinity: dict[str, float],
        k: int = 10,
        seen_items: Optional[set[int]] = None,
        candidate_pool: Optional[list[int]] = None,
    ) -> list[tuple[int, float]]:
        """Get recommendations based on user's interaction count."""

        if seen_items is None:
            seen_items = set()

        # Determine strategy
        if n_interactions == 0:
            return self._cold_start_new_user(genre_affinity, k, seen_items)
        elif n_interactions < self.config.min_ratings_for_cf:
            return self._warm_start_content_only(
                genre_affinity, k, seen_items, candidate_pool
            )
        else:
            # Sufficient history for full hybrid — caller handles this
            return []

    def _cold_start_new_user(
        self,
        genre_affinity: dict[str, float],
        k: int,
        seen_items: set[int],
    ) -> list[tuple[int, float]]:
        """Pure content-based from onboarding genre preferences."""
        if not genre_affinity:
            # No genre affinity — fall back to popular
            return self._popular_fallback(k, seen_items)

        # Use content scorer with genre affinity only
        all_movies = list(self.movie_genres.keys())
        candidates = [m for m in all_movies if m not in seen_items]

        scores = self.content_scorer.score_candidates(
            genre_affinity=genre_affinity,
            candidate_ids=candidates,
        )

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:k]

    def _warm_start_content_only(
        self,
        genre_affinity: dict[str, float],
        k: int,
        seen_items: set[int],
        candidate_pool: Optional[list[int]],
    ) -> list[tuple[int, float]]:
        """Content + popularity blend for users with some history but not enough for CF."""
        if candidate_pool is None:
            # Use popular movies as candidate pool
            candidate_pool = self.popular_movies

        candidates = [m for m in candidate_pool if m not in seen_items]

        # Content scores
        content_scores = self.content_scorer.score_candidates(
            genre_affinity=genre_affinity,
            candidate_ids=candidates,
        )

        # Popularity scores (normalized)
        max_pop_rank = len(self.popular_movies)
        pop_scores = {}
        for i, mid in enumerate(self.popular_movies):
            if mid in candidates:
                pop_scores[mid] = 1.0 - (i / max_pop_rank)

        # Blend: 70% content, 30% popularity
        blended = {}
        for mid in candidates:
            blended[mid] = 0.7 * content_scores.get(mid, 0) + 0.3 * pop_scores.get(mid, 0)

        ranked = sorted(blended.items(), key=lambda x: -x[1])
        return ranked[:k]

    def _popular_fallback(self, k: int, seen_items: set[int]) -> list[tuple[int, float]]:
        """Fallback to popularity when nothing else works."""
        result = []
        for i, mid in enumerate(self.popular_movies):
            if mid not in seen_items:
                result.append((mid, 1.0 - i * 0.01))
            if len(result) >= k:
                break
        return result

    def get_new_movie_score(
        self,
        movie_id: int,
        user_genre_affinity: dict[str, float],
    ) -> float:
        """Score a brand-new movie (no ratings) for a user."""
        if movie_id not in self.movie_genres:
            return 0.0

        # Pure content match
        genres = self.movie_genres[movie_id]
        if not genres or not user_genre_affinity:
            return 0.0

        # Jaccard-like overlap weighted by affinity
        score = sum(user_genre_affinity.get(g, 0) for g in genres) / len(genres)
        return float(score)


def create_onboarding_profile(
    selected_genres: list[str],
    rated_movies: dict[int, float],  # movie_id -> rating (1-5)
    movie_genres: dict[int, set[str]],
) -> dict[str, float]:
    """Build initial genre affinity from onboarding data."""
    genre_weights = {}

    # 1. Explicit genre selections (strong signal)
    for genre in selected_genres:
        genre_weights[genre] = genre_weights.get(genre, 0) + 2.0

    # 2. Rated movies' genres (weighted by rating)
    for movie_id, rating in rated_movies.items():
        genres = movie_genres.get(movie_id, set())
        for genre in genres:
            # Normalize rating to [0, 1] weight
            weight = max(0.0, min(1.0, (rating - 1) / 4.0))  # 1->0, 5->1
            genre_weights[genre] = genre_weights.get(genre, 0) + weight

    # Normalize
    total = sum(genre_weights.values())
    if total > 0:
        genre_weights = {g: w / total for g, w in genre_weights.items()}

    return genre_weights
