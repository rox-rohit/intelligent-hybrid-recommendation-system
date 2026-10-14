"""Implicit feedback: convert interaction events into training confidence.

Methodology (spec section 11 — principled, documented, no fake 'real' data):

MovieLens 25M contains ONLY explicit ratings. Real platforms log implicit
events (views, clicks, skips, watchlist adds). To build and demonstrate
the implicit-feedback pipeline honestly, we derive interaction *types*
from the observed rating behavior with a fixed, published mapping —
and we document that these are DERIVED signals, not fabricated telemetry.

Derivation rules (deterministic, from rating value + platform patterns
reported in the literature):
  rating >= 4.5        -> 'complete'  (finished and loved it)
  rating in [3.5, 4.5) -> 'watch'     (watched, liked)
  rating in [2.5, 3.5) -> 'view'      (watched, lukewarm)
  rating < 2.5         -> 'skip'      (negative signal)
  (user tags on a movie are treated as a 'search' interest signal)

Confidence weights follow Hu, Koren & Volinsky (2008): positive events
get confidence >= 1 (more certain), negatives near 0. Watchlist adds
(when they exist in the live app via POST /watchlists) get the highest
explicit-intent weight.

This module owns the single source of truth for interaction -> weight;
the API, the CF trainers, and the profile builder all import it, so
behavior stays consistent across the system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Single source of truth: every interaction type and its confidence weight.
INTERACTION_WEIGHTS: dict[str, float] = {
    "complete": 5.0,   # high rating: strong positive
    "watch": 3.0,      # positive watch
    "view": 1.0,       # neutral interaction
    "click": 0.5,      # weak interest
    "search": 1.0,    # tag applied / searched
    "watchlist": 6.0,  # explicit intent to watch (live-app signal)
    "like": 4.0,       # explicit like (live-app signal)
    "rate": 1.0,       # explicit rating, weighted by rating value at inference
    "remove_watchlist": 0.0,  # explicit removal, no positive preference evidence
    "dislike": 0.05,   # explicit negative (near-zero confidence, never 0:
                       # ALS treats 0 as 'unobserved', so negatives must be
                       # distinguishable from 'never seen')
    "skip": 0.05,      # negative implicit signal, same reasoning
    "repeat_view": 2.0,  # watched more than once (derived from duplicates)
}

# Rating -> interaction type derivation (documented simulation, see above).
RATING_TO_TYPE: list[tuple[float, str]] = [
    (4.5, "complete"),
    (3.5, "watch"),
    (2.5, "view"),
    (0.0, "skip"),
]


@dataclass(frozen=True)
class InteractionSchema:
    """Column contract for interaction tables across the system."""

    USER = "userId"
    ITEM = "movieId"
    TYPE = "interaction_type"
    TS = "timestamp"
    RATING = "rating"


def rating_to_interaction_type(rating: float) -> str:
    """Map an explicit rating to a simulated interaction type (documented)."""
    for threshold, itype in RATING_TO_TYPE:
        if rating >= threshold:
            return itype
    return "skip"


def derive_interactions(ratings: pd.DataFrame) -> pd.DataFrame:
    """Turn the ratings table into a typed interactions table.

    Output columns: userId, movieId, interaction_type, rating, timestamp.
    The interaction_type column is DERIVED (see module docstring) — this
    is the documented simulation, and every consumer labels it as such.
    """
    out = ratings[["userId", "movieId", "rating", "timestamp"]].copy()
    out["interaction_type"] = out["rating"].map(rating_to_interaction_type)
    return out


def weight_of(interaction_type: str) -> float:
    """Confidence weight for an interaction type (KeyError on unknown)."""
    try:
        return INTERACTION_WEIGHTS[interaction_type]
    except KeyError:
        raise ValueError(
            f"Unknown interaction type '{interaction_type}'. "
            f"Known: {sorted(INTERACTION_WEIGHTS)}"
        ) from None


def build_confidence_matrix(
    interactions: pd.DataFrame,
    n_users: int,
    n_items: int,
    user_index: pd.Series,   # userId -> row
    item_index: pd.Series,   # movieId -> column
) -> "scipy.sparse.csr_matrix":
    """Sparse (n_users, n_items) matrix of implicit confidence weights.

    This is the training input for ItemKNN and ImplicitALS, and the
    lookup structure for live-app interactions. If a user/item pair
    appears with several types (possible in the live app), the maximum
    confidence wins — strongest evidence.
    """
    import scipy.sparse as sp

    rows = interactions["userId"].map(user_index)
    cols = interactions["movieId"].map(item_index)
    if rows.isna().any() or cols.isna().any():
        unknown_users = interactions.loc[rows.isna(), "userId"].unique()[:5]
        unknown_items = interactions.loc[cols.isna(), "movieId"].unique()[:5]
        raise ValueError(
            f"Interactions reference unknown users/items "
            f"(e.g. users {list(unknown_users)}, items {list(unknown_items)}). "
            "Build the index from the same interaction set."
        )

    r = rows.to_numpy(np.int32)
    c = cols.to_numpy(np.int32)
    w = interactions["interaction_type"].map(weight_of).to_numpy(np.float32)

    # csr_matrix would SUM duplicate (user, item) pairs; we want MAX.
    # Sort by (user, item, weight) and keep only the last entry per pair.
    order = np.lexsort((w, r, c))
    r, c, w = r[order], c[order], w[order]
    keep = np.ones(len(r), dtype=bool)
    if len(r) > 1:
        same_pair = (r[1:] == r[:-1]) & (c[1:] == c[:-1])
        keep[:-1][same_pair] = False  # drop earlier (lower-weight) duplicates

    return sp.csr_matrix(
        (w[keep], (r[keep], c[keep])), shape=(n_users, n_items), dtype=np.float32
    )
