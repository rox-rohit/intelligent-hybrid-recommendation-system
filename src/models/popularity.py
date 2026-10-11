"""Popularity and freshness signals.

Popularity: Bayesian-smoothed mean rating (IMDb weighted rating formula).
  WR = (v / (v + m)) * R + (m / (v + m)) * C
  where R = movie mean, v = rating count, m = minimum votes (prior
  strength), C = global mean. A movie with 3 ratings of 5.0 must not
  outrank a classic with 50,000 ratings of 4.2.

Freshness: recency decay over release year, normalized to catalog span.
Documented tradeoff: pure popularity is engagement-biased toward old
classics; pure freshness is a list of new releases. The hybrid blends
both with tuned weights rather than trusting either alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

log = get_logger(__name__)

DEFAULT_MIN_VOTES = 100  # prior strength: below this, ratings barely move the score


class PopularityModel:
    """Bayesian-smoothed popularity + normalized freshness per movie."""

    def __init__(self, movie_stats: pd.DataFrame, min_votes: int = DEFAULT_MIN_VOTES):
        """movie_stats: DataFrame indexed by movieId with columns
        [mean_rating, n_ratings, release_year]."""
        missing = {"mean_rating", "n_ratings", "release_year"} - set(movie_stats.columns)
        if missing:
            raise ValueError(f"movie_stats missing columns: {missing}")
        self.stats = movie_stats
        self.min_votes = min_votes
        # Prior C = the mean of ALL ratings (vote-weighted), not the mean of
        # per-movie means — the latter is inflated by tiny-sample movies.
        total_votes = float(movie_stats["n_ratings"].sum())
        self.global_mean = float(
            (movie_stats["mean_rating"] * movie_stats["n_ratings"]).sum() / total_votes
        )

        v = movie_stats["n_ratings"].astype(float)
        m = float(min_votes)
        r = movie_stats["mean_rating"].astype(float)
        self.bayesian_score = (v / (v + m)) * r + (m / (v + m)) * self.global_mean

        years = movie_stats["release_year"]
        self._year_min = float(years.min())
        self._year_max = float(years.max())

    # ------------------------------------------------------------------ #
    @property
    def popularity(self) -> pd.Series:
        """Min-max normalized Bayesian popularity score in [0, 1]."""
        s = self.bayesian_score
        return (s - s.min()) / (s.max() - s.min() + 1e-12)

    @property
    def freshness(self) -> pd.Series:
        """Release-year recency normalized to [0, 1] over the catalog span.

        Missing years get 0 (treated as oldest) — documented, not hidden.
        """
        years = self.stats["release_year"]
        span = self._year_max - self._year_min
        if span <= 0:
            return pd.Series(0.5, index=self.stats.index)
        return ((years - self._year_min) / span).fillna(0.0).clip(0, 1)

    def top_popular(self, k: int = 10) -> pd.DataFrame:
        """The k most popular movies by Bayesian score (the popularity baseline
        for evaluation)."""
        out = self.stats.assign(
            popularity_score=self.popularity, bayesian=self.bayesian_score
        )
        return out.nlargest(k, "bayesian")[["bayesian", "popularity_score"]]
