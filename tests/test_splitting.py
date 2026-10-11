"""Unit tests for the time-aware splitter (leakage prevention)."""

import pandas as pd
import pytest

from src.evaluation.splitting import time_aware_split


def make_ratings():
    """User 1: 10 interactions, user 2: 3 (stays whole in train),
    user 3: 1 (cold, whole in train). Timestamps strictly increasing."""
    rows = []
    for uid, n in [(1, 10), (2, 3), (3, 1)]:
        for i in range(n):
            rows.append({"userId": uid, "movieId": 100 + i, "timestamp": 1000 * uid + i})
    return pd.DataFrame(rows)


class TestChronology:
    def test_no_future_leakage(self):
        """Every validation/test timestamp must be >= all train timestamps
        for the same user."""
        split = time_aware_split(make_ratings())
        for part_name in ("validation", "test"):
            part = getattr(split, part_name)
            merged = part.merge(
                split.train.groupby("userId")["timestamp"].max().rename("max_train_ts"),
                on="userId",
            )
            assert (merged["timestamp"] > merged["max_train_ts"]).all(), part_name

    def test_val_before_test(self):
        split = time_aware_split(make_ratings())
        merged = split.test.merge(
            split.validation.groupby("userId")["timestamp"].max().rename("max_val_ts"),
            on="userId",
        )
        assert (merged["timestamp"] > merged["max_val_ts"]).all()

    def test_split_sizes(self):
        split = time_aware_split(make_ratings())
        # user 1: 10 interactions -> ~1 val, ~1 test, 8 train
        assert len(split.validation[split.validation.userId == 1]) == 1
        assert len(split.test[split.test.userId == 1]) == 1
        assert len(split.train[split.train.userId == 1]) == 8

    def test_row_conservation(self):
        ratings = make_ratings()
        split = time_aware_split(ratings)
        assert (
            len(split.train) + len(split.validation) + len(split.test)
            == len(ratings)
        )


class TestColdUsers:
    def test_tiny_user_stays_whole_in_train(self):
        # user 3 has 1 interaction (< min_per_split=2) -> whole in train
        split = time_aware_split(make_ratings())
        assert len(split.validation[split.validation.userId == 3]) == 0
        assert len(split.test[split.test.userId == 3]) == 0
        assert len(split.train[split.train.userId == 3]) == 1

    def test_three_interaction_user_evaluated(self):
        # user 2 has 3 interactions (>= min_per_split) -> 1 val, 1 test, 1 train
        split = time_aware_split(make_ratings())
        assert len(split.validation[split.validation.userId == 2]) == 1
        assert len(split.test[split.test.userId == 2]) == 1
        assert len(split.train[split.train.userId == 2]) == 1

    def test_min_per_split_default(self):
        split = time_aware_split(make_ratings())
        # only user 1 has >= 2 interactions... user 2 has 3, so also evaluated
        assert split.n_users_evaluated == 2


class TestValidation:
    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="userId"):
            time_aware_split(pd.DataFrame({"movieId": [1], "timestamp": [1]}))

    def test_bad_fractions_raise(self):
        ratings = make_ratings()
        with pytest.raises(ValueError, match="val_frac"):
            time_aware_split(ratings, val_frac=0.6, test_frac=0.6)

    def test_tiny_users_guard_train_nonempty(self):
        # every evaluated user keeps >= 1 train interaction even at
        # extreme fractions
        split = time_aware_split(make_ratings(), val_frac=0.45, test_frac=0.45)
        assert (split.train.groupby("userId").size() >= 1).all()
