"""Unit tests for implicit-feedback weighting."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.features.interactions import (
    INTERACTION_WEIGHTS,
    build_confidence_matrix,
    derive_interactions,
    rating_to_interaction_type,
    weight_of,
)


class TestRatingToType:
    def test_high_rating_completes(self):
        assert rating_to_interaction_type(5.0) == "complete"
        assert rating_to_interaction_type(4.5) == "complete"

    def test_mid_ratings(self):
        assert rating_to_interaction_type(4.0) == "watch"
        assert rating_to_interaction_type(3.0) == "view"
        assert rating_to_interaction_type(2.0) == "skip"
        assert rating_to_interaction_type(0.5) == "skip"

    def test_all_types_covered(self):
        # every valid MovieLens rating maps to a known weight
        for r in np.arange(0.5, 5.01, 0.5):
            assert weight_of(rating_to_interaction_type(r)) > 0


class TestDeriveInteractions:
    def test_columns_and_types(self):
        ratings = pd.DataFrame(
            {
                "userId": [1, 1],
                "movieId": [10, 20],
                "rating": [5.0, 1.0],
                "timestamp": [1, 2],
            }
        )
        out = derive_interactions(ratings)
        assert list(out.columns) == [
            "userId", "movieId", "rating", "timestamp", "interaction_type",
        ]
        assert out.iloc[0]["interaction_type"] == "complete"
        assert out.iloc[1]["interaction_type"] == "skip"

    def test_input_not_mutated(self):
        ratings = pd.DataFrame(
            {"userId": [1], "movieId": [10], "rating": [4.0], "timestamp": [1]}
        )
        snap = ratings.copy()
        derive_interactions(ratings)
        pd.testing.assert_frame_equal(ratings, snap)


class TestWeights:
    def test_positive_intent_orders_sensibly(self):
        assert weight_of("watchlist") > weight_of("complete")
        assert weight_of("complete") > weight_of("watch") > weight_of("view")

    def test_negatives_distinguishable_from_unobserved(self):
        # ALS treats 0 as 'unobserved'; negatives must be > 0 but tiny
        assert 0 < weight_of("dislike") < weight_of("view")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown interaction type"):
            weight_of("teleport")

    def test_schema_covers_all_types(self):
        # every type referenced in the live-app API must have a weight
        for t in ["like", "dislike", "watchlist", "click", "search"]:
            assert t in INTERACTION_WEIGHTS


class TestConfidenceMatrix:
    @pytest.fixture
    def setup(self):
        interactions = pd.DataFrame(
            {
                "userId": [1, 1, 2, 2],
                "movieId": [10, 20, 10, 10],
                "interaction_type": ["watch", "complete", "view", "complete"],
            }
        )
        user_index = pd.Series({1: 0, 2: 1})
        item_index = pd.Series({10: 0, 20: 1})
        return interactions, user_index, item_index

    def test_shape_and_values(self, setup):
        interactions, ui, ii = setup
        mat = build_confidence_matrix(interactions, 2, 2, ui, ii)
        assert mat.shape == (2, 2)
        assert mat[0, 0] == weight_of("watch")
        assert mat[0, 1] == weight_of("complete")
        assert mat[1, 0] == weight_of("complete")  # duplicate pair -> max

    def test_duplicates_take_max(self, setup):
        interactions, ui, ii = setup
        mat = build_confidence_matrix(interactions, 2, 2, ui, ii)
        # user 2 x item 10 appears as view (1.0) then complete (5.0) -> 5.0
        assert mat[1, 0] == 5.0

    def test_sparse_output(self, setup):
        interactions, ui, ii = setup
        mat = build_confidence_matrix(interactions, 2, 2, ui, ii)
        assert sp.issparse(mat)
        assert mat.nnz == 3  # (0,0), (0,1), (1,0) — no phantom entries

    def test_unknown_user_raises(self, setup):
        interactions, ui, ii = setup
        bad = pd.concat(
            [interactions, pd.DataFrame(
                [{"userId": 99, "movieId": 10, "interaction_type": "watch"}]
            )],
            ignore_index=True,
        )
        with pytest.raises(ValueError, match="unknown users"):
            build_confidence_matrix(bad, 2, 2, ui, ii)
