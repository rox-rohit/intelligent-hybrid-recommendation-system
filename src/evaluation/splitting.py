"""Time-aware splitting of user interactions.

Rule (spec section 12: never leak future information):
For each user, interactions are sorted by timestamp and split by
chronology, NOT by random sampling:
  train:    earliest ~80% of each user's interactions
  val:      next ~10% (immediately after train)
  test:     latest ~10%

This mirrors production: models see the past, are scored on what the user
actually did next. Random splits leak the future into training and
inflate offline metrics — a classic recommender evaluation bug.

Users with too few interactions (< min_per_split) are kept whole in
train: they are realistic cold-start training signal, and cannot be
evaluated offline anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def n_users_evaluated(self) -> int:
        """Users present in validation or test (users we can score offline)."""
        return len(
            set(self.validation["userId"].unique()) | set(self.test["userId"].unique())
        )


def time_aware_split(
    ratings: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    min_per_split: int = 2,
) -> TimeSplit:
    """Chronological per-user split. ratings must contain userId, movieId,
    timestamp (and is expected pre-sorted per user by the pipeline)."""
    if not {"userId", "movieId", "timestamp"}.issubset(ratings.columns):
        raise ValueError("ratings must have userId, movieId, timestamp columns")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    parts_val: list[pd.DataFrame] = []
    parts_test: list[pd.DataFrame] = []
    parts_train: list[pd.DataFrame] = []  # evaluated users' earliest slice

    df = ratings.sort_values(["userId", "timestamp"], kind="stable")

    for uid, grp in df.groupby("userId", sort=False):
        n = len(grp)
        if n < min_per_split:
            parts_train.append(grp)  # whole history stays in train (cold)
            continue
        n_val = max(1, int(round(n * val_frac)))
        n_test = max(1, int(round(n * test_frac)))
        # guard: at least one interaction stays in train
        while n_val + n_test >= n:
            (n_val, n_test) = (n_val - 1, n_test) if n_val >= n_test else (n_val, n_test - 1)
        parts_train.append(grp.iloc[: n - n_val - n_test])
        parts_val.append(grp.iloc[n - n_val - n_test : n - n_test])
        parts_test.append(grp.iloc[n - n_test :])

    train = pd.concat(parts_train, ignore_index=True) if parts_train else ratings.iloc[0:0]
    validation = pd.concat(parts_val, ignore_index=True) if parts_val else ratings.iloc[0:0]
    test = pd.concat(parts_test, ignore_index=True) if parts_test else ratings.iloc[0:0]
    return TimeSplit(train=train, validation=validation, test=test)
