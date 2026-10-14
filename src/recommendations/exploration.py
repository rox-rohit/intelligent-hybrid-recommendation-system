"""Exploration policies: ε-greedy + novelty boosting (Milestone 7)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class ExplorationConfig:
    epsilon: float = 0.1              # ε-greedy exploration rate
    novelty_weight: float = 0.3       # Boost for long-tail items
    min_popularity_for_novelty: float = 0.1  # Below this = novel
    explore_mode_boost: float = 2.0   # Multiplier in explore mode


class ExplorationPolicy:
    """Applies exploration to ranked recommendations.

    Two modes:
      - personalized: ε-greedy (mostly exploit, sometimes explore)
      - explore: Boost novelty/long-tail items
    """

    def __init__(self, config: ExplorationConfig | None = None):
        self.config = config or ExplorationConfig()

    def apply(
        self,
        ranked_items: list[tuple[int, float]],   # (movie_id, score)
        item_popularity: dict[int, float],       # movie_id -> popularity [0, 1]
        mode: str = "personalized",              # "personalized" | "explore"
        rng: Optional[random.Random] = None,
    ) -> list[tuple[int, float]]:
        """Apply exploration policy to ranked items.

        Returns re-ranked list with exploration applied.
        """
        if not ranked_items:
            return []

        if rng is None:
            rng = random

        if mode == "explore":
            return self._apply_explore_mode(ranked_items, item_popularity)

        # Personalized mode: ε-greedy
        return self._apply_epsilon_greedy(ranked_items, rng)

    def _apply_epsilon_greedy(
        self,
        ranked_items: list[tuple[int, float]],
        rng: random.Random,
    ) -> list[tuple[int, float]]:
        """With probability ε, shuffle top candidates."""
        if rng.random() >= self.config.epsilon:
            return ranked_items

        # Shuffle top portion
        n_shuffle = min(20, len(ranked_items))
        top = ranked_items[:n_shuffle]
        rest = ranked_items[n_shuffle:]

        shuffled = top.copy()
        rng.shuffle(shuffled)

        return shuffled + rest

    def _apply_explore_mode(
        self,
        ranked_items: list[tuple[int, float]],
        item_popularity: dict[int, float],
    ) -> list[tuple[int, float]]:
        """Boost novel/long-tail items for exploration."""
        boosted = []

        for mid, score in ranked_items:
            pop = item_popularity.get(mid, 0.5)

            if pop < self.config.min_popularity_for_novelty:
                # Novel item: boost score
                novelty_factor = (1 - pop) / self.config.min_popularity_for_novelty
                boost = self.config.explore_mode_boost * novelty_factor
                new_score = score * (1 + boost)
            else:
                new_score = score

            boosted.append((mid, new_score))

        # Re-sort by boosted score
        boosted.sort(key=lambda x: -x[1])
        return boosted


def compute_novelty_score(
    item_popularity: float,
    config: ExplorationConfig | None = None,
) -> float:
    """Compute novelty multiplier for an item.

    Returns multiplier >= 1.0 (1.0 = no boost).
    """
    if config is None:
        config = ExplorationConfig()

    if item_popularity < config.min_popularity_for_novelty:
        return 1.0 + config.novelty_weight * (
            1.0 - item_popularity / config.min_popularity_for_novelty
        )
    return 1.0


def get_exploration_explanation(
    mode: str,
    was_explored: bool,
    item_popularity: float,
) -> Optional[str]:
    """Generate human-readable explanation for exploration decision."""
    if mode == "explore":
        if item_popularity < 0.1:
            return "Recommended for exploration — a hidden gem!"
        return "Explore mode: showing variety beyond your usual picks"

    if was_explored:
        return "Surprise pick — trying something different"

    return None