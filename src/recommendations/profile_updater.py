"""Dynamic user profile updates on every interaction (Milestone 6).

Implements incremental profile updates with exponential time decay.
Profiles track:
  - Genre affinities (normalized, decayed over time)
  - Average rating (running average)
  - Interaction count
  - Last update timestamp
  - Recency-weighted centroid for content scoring
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.features.interactions import INTERACTION_WEIGHTS
from src.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class ProfileUpdateConfig:
    """Configuration for profile update behavior."""
    decay_half_life_days: float = 30.0      # Affinity halves every 30 days
    min_interactions_for_cf: int = 10       # Min interactions before CF kicks in
    max_genre_affinities: int = 20          # Cap on tracked genres
    recency_weight: float = 0.7             # Weight for recent interactions in centroid


class ProfileUpdater:
    """Incremental user profile updates with time decay.

    On each interaction:
      1. Decay existing genre affinities by elapsed time
      2. Add new evidence from the interaction
      3. Renormalize affinities to sum to 1.0
      4. Update running statistics
    """

    def __init__(self, config: ProfileUpdateConfig | None = None):
        self.config = config or ProfileUpdateConfig()
        self._daily_decay = np.exp(-np.log(2) / self.config.decay_half_life_days)

    def update_on_interaction(
        self,
        genre_affinity: dict[str, float],
        movie_genres: set[str],
        interaction_type: str,
        rating: Optional[float],
        timestamp: datetime,
        last_updated: Optional[datetime] = None,
        n_interactions: int = 0,
        avg_rating: Optional[float] = None,
    ) -> dict:
        """Apply one interaction to the profile.

        Returns updated profile dict with:
          - genre_affinity: dict[str, float] (normalized)
          - n_interactions: int
          - avg_rating: float
          - last_updated: datetime
        """
        # 1. Time decay
        if last_updated:
            days = (timestamp - last_updated).days
            if days > 0:
                decay_factor = self._daily_decay ** days
                genre_affinity = {g: v * decay_factor for g, v in genre_affinity.items()}

        # 2. Add positive evidence or reduce affinity for explicit negatives.
        weight = INTERACTION_WEIGHTS.get(interaction_type, 1.0)
        negative = interaction_type in {"dislike", "skip"} or (rating is not None and rating < 2.5)
        if negative:
            for genre in movie_genres:
                genre_affinity[genre] = max(0.0, genre_affinity.get(genre, 0.0) * 0.5)
        else:
            if rating is not None:
                weight *= max(0.25, rating / 3.0)
            for genre in movie_genres:
                genre_affinity[genre] = genre_affinity.get(genre, 0.0) + weight

        genre_affinity = {g: v for g, v in genre_affinity.items() if v > 0}

        # 3. Cap and renormalize
        if len(genre_affinity) > self.config.max_genre_affinities:
            # Keep top genres by affinity
            top_genres = sorted(genre_affinity.items(), key=lambda x: -x[1])[:self.config.max_genre_affinities]
            genre_affinity = dict(top_genres)

        total = sum(genre_affinity.values())
        if total > 0:
            genre_affinity = {g: v / total for g, v in genre_affinity.items()}

        # 4. Update stats
        n_interactions += 1

        if rating is not None:
            if avg_rating is None:
                avg_rating = rating
            else:
                # Running average: new_avg = (old_avg * n + new) / (n + 1)
                # But n_interactions already incremented, so use n_interactions - 1
                n = n_interactions - 1
                avg_rating = (avg_rating * n + rating) / n_interactions

        return {
            "genre_affinity": genre_affinity,
            "n_interactions": n_interactions,
            "avg_rating": avg_rating,
            "last_updated": timestamp,
        }

    def rebuild_from_interactions(
        self,
        interactions: list[dict],
        movie_genres_map: dict[int, set[str]],
    ) -> dict:
        """Full rebuild from interaction history (for periodic sync).

        interactions: list of {movie_id, interaction_type, rating, timestamp}
        movie_genres_map: {movie_id: set(genres)}
        """
        # Sort by timestamp
        interactions = sorted(interactions, key=lambda x: x["timestamp"])

        genre_affinity: dict[str, float] = {}
        n_interactions = 0
        avg_rating: Optional[float] = None
        last_updated = None

        for inter in interactions:
            movie_id = inter["movie_id"]
            genres = movie_genres_map.get(movie_id, set())

            result = self.update_on_interaction(
                genre_affinity=genre_affinity,
                movie_genres=genres,
                interaction_type=inter["interaction_type"],
                rating=inter.get("rating"),
                timestamp=inter["timestamp"],
                last_updated=last_updated,
                n_interactions=n_interactions,
                avg_rating=avg_rating,
            )
            genre_affinity = result["genre_affinity"]
            n_interactions = result["n_interactions"]
            avg_rating = result["avg_rating"]
            last_updated = result["last_updated"]

        return {
            "genre_affinity": genre_affinity,
            "n_interactions": n_interactions,
            "avg_rating": avg_rating,
            "last_updated": last_updated,
        }


def running_average(old_avg: float, new_value: float, n: int) -> float:
    """Compute running average: (old_avg * (n-1) + new_value) / n"""
    return (old_avg * (n - 1) + new_value) / n
