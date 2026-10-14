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

Implementation: a vectorized groupby over ~25M rows — no per-user
Python loop, no concat of 162k small frames (the naive version OOM'd
at exactly this step on the real dataset).
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


def _is_sorted_by_user_time(ratings: pd.DataFrame) -> bool:
    """Cheap vectorized check: are rows already grouped by userId with
    non-decreasing timestamps inside each user block? If so we can skip
    the full-frame sort copy (a ~700MB allocation on 25M rows)."""
    if len(ratings) < 2:
        return True
    uid = ratings["userId"].to_numpy()
    ts = ratings["timestamp"].to_numpy()
    same_user = uid[1:] == uid[:-1]
    ts_ok = ts[1:] >= ts[:-1]
    # steps within a user block must be non-decreasing in time
    return bool(np.all(ts_ok | ~same_user))


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

    if _is_sorted_by_user_time(ratings):
        df = ratings  # already sorted — avoid the sort's full-frame copy
    else:
        df = ratings.sort_values(["userId", "timestamp"], kind="stable")
    n = len(df)
    # position of each row within its user's chronology (0-based)
    grp_sizes = df.groupby("userId", sort=False)["userId"].transform("size").to_numpy()
    grp_ranks = df.groupby("userId", sort=False).cumcount().to_numpy()

    # Per-user split sizes (computed for every row's user at once). Identical
    # arithmetic to the scalar version, including the "at least one train
    # interaction" guard, which removes from the larger side first (ties
    # from val). The guard only bites for tiny users, so the vectorized loop
    # converges in a few full-array passes.
    n_val = np.maximum(1, np.rint(grp_sizes * val_frac)).astype(np.int64)
    n_test = np.maximum(1, np.rint(grp_sizes * test_frac)).astype(np.int64)
    while True:
        over = (n_val + n_test) >= grp_sizes
        if not over.any():
            break
        take_val = over & (n_val >= n_test)
        take_test = over & (n_val < n_test)
        n_val = np.where(take_val, n_val - 1, n_val)
        n_test = np.where(take_test, n_test - 1, n_test)

    cold = grp_sizes < min_per_split
    cut_val = grp_sizes - n_val - n_test   # first val rank (per row's user)
    cut_test = grp_sizes - n_test         # first test rank
    is_train = cold | (grp_ranks < cut_val)
    is_val = (~cold) & (grp_ranks >= cut_val) & (grp_ranks < cut_test)
    is_test = (~cold) & (grp_ranks >= cut_test)
    # free intermediates before the row takes — on 25M rows these arrays
    # plus the split copies were enough to OOM on a 16GB machine
    del grp_sizes, grp_ranks, n_val, n_test, cut_val, cut_test, cold, over

    # Boolean masks over the sorted frame; slicing keeps the stable-sorted
    # row order and leaks nothing. Free each mask right after its take.
    train = df[is_train].reset_index(drop=True)
    del is_train
    validation = df[is_val].reset_index(drop=True)
    del is_val
    test = df[is_test].reset_index(drop=True)
    del is_test
    assert len(train) + len(validation) + len(test) == n
    return TimeSplit(train=train, validation=validation, test=test)
