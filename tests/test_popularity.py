"""Unit tests for the popularity/freshness model."""

import pandas as pd
import pytest

from src.models.popularity import PopularityModel


@pytest.fixture
def movie_stats():
    # classic: many ratings, solid mean; new hit: few ratings, perfect mean;
    # obscure: 2 ratings of 5.0 (should NOT top the ranking)
    return pd.DataFrame(
        {
            "mean_rating": [4.2, 5.0, 5.0, 3.0],
            "n_ratings": [50_000, 5, 2, 1_000],
            "release_year": [1999, 2023, 2021, 1990],
        },
        index=pd.Index([1, 2, 3, 4], name="movieId"),
    )


class TestPopularity:
    def test_bayesian_smoothing_penalizes_few_votes(self, movie_stats):
        model = PopularityModel(movie_stats, min_votes=100)
        # movie 1 (50k ratings @ 4.2) must outrank movie 3 (2 ratings @ 5.0)
        assert model.bayesian_score[1] > model.bayesian_score[3]

    def test_perfect_mean_with_few_votes_still_low(self, movie_stats):
        model = PopularityModel(movie_stats, min_votes=100)
        # movie 3's score should be pulled hard toward the global mean
        assert model.bayesian_score[3] < 4.2

    def test_popularity_normalized(self, movie_stats):
        pop = PopularityModel(movie_stats).popularity
        assert pop.between(0, 1).all()
        assert abs(pop.max() - 1.0) < 1e-9
        assert abs(pop.min()) < 1e-9

    def test_top_popular_ordering(self, movie_stats):
        # strong prior (m=1000): the 50k-rating classic outranks the
        # 5-rating perfect movie — smoothing dominates small samples
        top = PopularityModel(movie_stats, min_votes=1000).top_popular(k=2)
        assert top.index[0] == 1
        assert len(top) == 2


class TestFreshness:
    def test_freshness_newest_is_one(self, movie_stats):
        fresh = PopularityModel(movie_stats).freshness
        assert abs(fresh.loc[2] - 1.0) < 1e-9  # 2023

    def test_freshness_oldest_is_zero(self, movie_stats):
        fresh = PopularityModel(movie_stats).freshness
        assert abs(fresh.loc[4]) < 1e-9  # 1990

    def test_freshness_missing_year_treated_as_oldest(self):
        # two real years + one missing: span > 0 so the fillna path is exercised
        stats = pd.DataFrame(
            {
                "mean_rating": [4.0, 4.0, 4.0],
                "n_ratings": [100, 100, 100],
                "release_year": [2000, 2010, None],
            },
            index=pd.Index([1, 2, 3], name="movieId"),
        )
        fresh = PopularityModel(stats).freshness
        assert fresh.loc[3] == 0.0   # missing -> oldest
        assert fresh.loc[2] == 1.0   # newest
        assert fresh.loc[1] == 0.0   # oldest real year


class TestValidation:
    def test_missing_columns_raises(self):
        with pytest.raises(ValueError, match="missing columns"):
            PopularityModel(pd.DataFrame({"mean_rating": [1.0]}))
