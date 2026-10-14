"""Unit tests for the ranking stage (features + ranker) on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.ranking.features import (
    FEATURE_COLUMNS,
    UserItemStats,
    build_features,
    label_candidates,
    _minmax_dict,
)
from src.ranking.ranker import PointwiseRanker


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_stats():
    """UserItemStats over a 5-item, 2-user toy catalog."""
    return UserItemStats(
        item_n_ratings={1: 100, 2: 50, 3: 200, 4: 10, 5: 5},
        item_avg_rating={1: 4.2, 2: 3.5, 3: 4.8, 4: 2.0, 5: 3.0},
        user_n_ratings={0: 20, 1: 5},
        user_genre_profile={
            0: {"Action": 0.5, "Comedy": 0.3, "Drama": 0.2},
            1: {"Horror": 0.8, "Comedy": 0.2},
        },
        item_genres={
            1: {"Action", "Drama"},
            2: {"Comedy"},
            3: {"Action", "Horror"},
            4: {"Romance"},
            5: {"Comedy", "Drama"},
        },
        item_freshness={1: 0.8, 2: 0.5, 3: 0.9, 4: 0.2, 5: 0.6},
        item_pop_score={1: 0.7, 2: 0.4, 3: 0.95, 4: 0.1, 5: 0.05},
    )


@pytest.fixture
def als_scores():
    return {
        0: {1: 5.0, 2: 3.0, 3: 1.0, 4: 0.5, 5: 0.1},
        1: {3: 4.0, 4: 2.0, 5: 1.0},
    }


@pytest.fixture
def knn_scores():
    return {
        0: {1: 0.9, 2: 0.7, 3: 0.5},
        1: {3: 0.6, 5: 0.3},
    }


@pytest.fixture
def content_scores():
    return {
        0: {1: 0.8, 2: 0.6, 4: 0.3},
        1: {3: 0.9, 4: 0.4},
    }


# ── Feature builder tests ────────────────────────────────────────────────

class TestMinmaxDict:
    def test_basic(self):
        out = _minmax_dict({1: 2.0, 2: 4.0, 3: 6.0})
        assert out == {1: 0.0, 2: 0.5, 3: 1.0}

    def test_empty(self):
        assert _minmax_dict({}) == {}

    def test_constant(self):
        assert _minmax_dict({1: 3.0, 2: 3.0}) == {1: 0.0, 2: 0.0}


class TestBuildFeatures:
    def test_correct_columns(self, tiny_stats, als_scores, knn_scores, content_scores):
        df = build_features(
            [0, 1], {0: [1, 2, 3], 1: [3, 4, 5]},
            als_scores, knn_scores, content_scores, tiny_stats,
        )
        expected_cols = {"userId", "movieId"} | set(FEATURE_COLUMNS)
        assert set(df.columns) == expected_cols

    def test_correct_row_count(self, tiny_stats, als_scores, knn_scores, content_scores):
        df = build_features(
            [0, 1], {0: [1, 2, 3], 1: [3, 4, 5]},
            als_scores, knn_scores, content_scores, tiny_stats,
        )
        assert len(df) == 6  # 3 candidates per user

    def test_missing_scores_default_zero(self, tiny_stats, als_scores, knn_scores, content_scores):
        """Items with no ALS/KNN/content score should get 0.0, not NaN."""
        df = build_features(
            [0], {0: [5]},  # item 5 has near-zero ALS, no KNN, no content
            als_scores, knn_scores, content_scores, tiny_stats,
        )
        row = df.iloc[0]
        assert row["cf_knn_score"] == 0.0
        assert row["content_score"] == 0.0

    def test_empty_candidates(self, tiny_stats):
        df = build_features([0], {0: []}, {}, {}, {}, tiny_stats)
        assert len(df) == 0
        assert set(df.columns) == {"userId", "movieId"} | set(FEATURE_COLUMNS)

    def test_genre_overlap_correct(self, tiny_stats, als_scores, knn_scores, content_scores):
        """User 0 likes Action/Comedy/Drama; item 1 is Action+Drama → overlap = 1.0."""
        df = build_features(
            [0], {0: [1]},
            als_scores, knn_scores, content_scores, tiny_stats,
        )
        assert df.iloc[0]["genre_overlap"] == 1.0  # both Action, Drama in profile

    def test_genre_overlap_partial(self, tiny_stats, als_scores, knn_scores, content_scores):
        """User 0 likes Action/Comedy/Drama; item 3 is Action+Horror → overlap = 0.5."""
        df = build_features(
            [0], {0: [3]},
            als_scores, knn_scores, content_scores, tiny_stats,
        )
        assert df.iloc[0]["genre_overlap"] == 0.5

    def test_genre_overlap_no_match(self, tiny_stats, als_scores, knn_scores, content_scores):
        """User 0 likes Action/Comedy/Drama; item 4 is Romance → overlap = 0.0."""
        df = build_features(
            [0], {0: [4]},
            als_scores, knn_scores, content_scores, tiny_stats,
        )
        assert df.iloc[0]["genre_overlap"] == 0.0

    def test_log_item_pop(self, tiny_stats, als_scores, knn_scores, content_scores):
        df = build_features(
            [0], {0: [1]},
            als_scores, knn_scores, content_scores, tiny_stats,
        )
        assert abs(df.iloc[0]["log_item_pop"] - np.log1p(100)) < 1e-6

    def test_cf_scores_normalized_per_user(self, tiny_stats, als_scores, knn_scores, content_scores):
        """ALS scores for user 0 are [5, 3, 1, 0.5, 0.1] → item 1 gets 1.0."""
        df = build_features(
            [0], {0: [1, 2, 3, 4, 5]},
            als_scores, knn_scores, content_scores, tiny_stats,
        )
        item1 = df[df["movieId"] == 1].iloc[0]
        item5 = df[df["movieId"] == 5].iloc[0]
        assert abs(item1["cf_als_score"] - 1.0) < 1e-6
        assert abs(item5["cf_als_score"] - 0.0) < 1e-6


class TestLabelCandidates:
    def test_labels_match_ground_truth(self, tiny_stats, als_scores, knn_scores, content_scores):
        df = build_features(
            [0], {0: [1, 2, 3]},
            als_scores, knn_scores, content_scores, tiny_stats,
        )
        truth = {0: {1, 3}}  # user 0's held-out relevant items
        labels = label_candidates(df, truth)
        assert list(labels) == [1, 0, 1]

    def test_no_positives(self, tiny_stats, als_scores, knn_scores, content_scores):
        df = build_features(
            [0], {0: [2, 4]},
            als_scores, knn_scores, content_scores, tiny_stats,
        )
        labels = label_candidates(df, {0: {99}})
        assert labels.sum() == 0

    def test_empty_truth(self, tiny_stats, als_scores, knn_scores, content_scores):
        df = build_features(
            [0], {0: [1]},
            als_scores, knn_scores, content_scores, tiny_stats,
        )
        labels = label_candidates(df, {})
        assert labels.sum() == 0


class TestUserItemStatsFromTrain:
    def test_from_train(self):
        train = pd.DataFrame({
            "userId": [0, 0, 0, 1, 1],
            "movieId": [1, 2, 3, 1, 4],
            "rating": [4.5, 3.0, 5.0, 2.0, 4.0],
        })
        movies = pd.DataFrame({
            "movieId": [1, 2, 3, 4],
            "genres_list": [["Action"], ["Comedy"], ["Action", "Drama"], ["Comedy"]],
        })
        pop = {1: 0.7, 2: 0.3, 3: 0.9, 4: 0.1}
        fresh = {1: 0.8, 2: 0.5, 3: 0.9, 4: 0.2}
        stats = UserItemStats.from_train(train, movies, pop, fresh)

        assert stats.user_n_ratings[0] == 3
        assert stats.user_n_ratings[1] == 2
        assert stats.item_n_ratings[1] == 2  # rated by user 0 and 1
        assert stats.item_n_ratings[4] == 1
        assert abs(stats.item_avg_rating[1] - 3.25) < 1e-6  # (4.5 + 2.0) / 2
        assert "Action" in stats.user_genre_profile[0]
        assert stats.item_genres[3] == {"Action", "Drama"}


# ── Ranker tests ──────────────────────────────────────────────────────────

class TestPointwiseRanker:
    @pytest.fixture
    def synthetic_data(self):
        """200 labeled samples with 10 features, ~20% positive rate."""
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame(
            rng.standard_normal((n, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )
        # Make labels correlated with first feature so the model can learn
        prob = 1 / (1 + np.exp(-2 * X["cf_als_score"]))
        y = rng.binomial(1, prob).astype(np.int32)
        X["userId"] = np.repeat([0, 1], n // 2)
        X["movieId"] = np.tile(np.arange(n // 2), 2)
        return X, y

    def test_fit_predict_cycle(self, synthetic_data):
        X, y = synthetic_data
        ranker = PointwiseRanker(max_iter=50)
        ranker.fit(X, y)
        scores = ranker.predict_scores(X)
        assert scores.shape == (len(X),)
        assert all(0 <= s <= 1 for s in scores)

    def test_not_fitted_raises(self):
        ranker = PointwiseRanker()
        with pytest.raises(RuntimeError, match="not fitted"):
            ranker.predict_scores(np.zeros((5, len(FEATURE_COLUMNS))))

    def test_rerank_ordering(self, synthetic_data):
        X, y = synthetic_data
        ranker = PointwiseRanker(max_iter=50)
        ranker.fit(X, y)
        user0 = X[X["userId"] == 0].copy().reset_index(drop=True)
        recs = ranker.rerank(user0, k=5)
        assert len(recs) == 5
        # Scores must be in descending order
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_feature_importances(self, synthetic_data):
        X, y = synthetic_data
        ranker = PointwiseRanker(max_iter=50)
        ranker.fit(X, y)
        imps = ranker.feature_importances()
        assert set(imps.keys()) == set(FEATURE_COLUMNS)
        assert all(v >= 0 for v in imps.values())

    def test_feature_importances_not_fitted(self):
        ranker = PointwiseRanker()
        with pytest.raises(RuntimeError, match="not fitted"):
            ranker.feature_importances()

    def test_save_load_roundtrip(self, synthetic_data, tmp_path):
        X, y = synthetic_data
        ranker = PointwiseRanker(max_iter=50)
        ranker.fit(X, y)
        scores_before = ranker.predict_scores(X)

        path = tmp_path / "ranker.joblib"
        ranker.save(path)
        loaded = PointwiseRanker.load(path)
        scores_after = loaded.predict_scores(X)

        np.testing.assert_array_almost_equal(scores_before, scores_after)

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PointwiseRanker.load(tmp_path / "nonexistent.joblib")

    def test_rerank_respects_k(self, synthetic_data):
        X, y = synthetic_data
        ranker = PointwiseRanker(max_iter=50)
        ranker.fit(X, y)
        user0 = X[X["userId"] == 0].copy().reset_index(drop=True)
        for k in [1, 3, 10]:
            recs = ranker.rerank(user0, k=k)
            assert len(recs) == k
