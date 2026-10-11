"""Unit tests for data cleaning rules, using small synthetic fixtures."""

import pandas as pd
import pytest

from src.data import cleaning


@pytest.fixture
def raw_movies():
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": ["Toy Story (1995)", "Unknown Movie", "Heat (1995)", "Old Film (1893)"],
            "genres": ["Animation|Comedy", "(no genres listed)", "Crime|Drama", "Documentary"],
        }
    )


@pytest.fixture
def raw_ratings():
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 2, 3],
            "movieId": [10, 10, 10, 20, 20, 10],
            "rating": [4.0, 3.5, 5.0, 0.0, 4.5, 5.5],  # dup, dup, out-of-range 5.5
            "timestamp": [100, 50, 80, 30, 20, 60],
        }
    )


@pytest.fixture
def raw_tags():
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 10, 10, 20],
            "tag": ["  Sci-Fi  ", "sci-fi", "MIND-BENDING  movie", ""],
            "timestamp": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def raw_genome():
    scores = pd.DataFrame(
        {
            "movieId": [10, 10, 20],
            "tagId": [1, 2, 1],
            "relevance": [0.9, 1.4, 0.7],  # 1.4 invalid
        }
    )
    tags = pd.DataFrame({"tagId": [1, 2], "tag": ["sci-fi", "action"]})
    return scores, tags


class TestCleanMovies:
    def test_year_extraction(self, raw_movies):
        out, _ = cleaning.clean_movies(raw_movies)
        assert out.loc[0, "release_year"] == 1995
        assert pd.isna(out.loc[1, "release_year"])  # malformed title kept, year NaN

    def test_no_genres_listed_becomes_empty(self, raw_movies):
        out, _ = cleaning.clean_movies(raw_movies)
        assert out.loc[1, "genres_list"] == []
        assert out.loc[1, "n_genres"] == 0
        assert out.loc[0, "genres_list"] == ["Animation", "Comedy"]

    def test_title_year_stripped(self, raw_movies):
        out, _ = cleaning.clean_movies(raw_movies)
        assert out.loc[0, "title_clean"] == "Toy Story"

    def test_input_not_mutated(self, raw_movies):
        snapshot = raw_movies.copy()
        cleaning.clean_movies(raw_movies)
        pd.testing.assert_frame_equal(raw_movies, snapshot)


class TestCleanRatings:
    def test_duplicates_removed(self, raw_ratings):
        out, rep = cleaning.clean_ratings(raw_ratings)
        # after range filtering, only (1, 10) remains duplicated
        assert rep.duplicates_removed == 1
        assert len(out) == 3

    def test_out_of_range_removed(self, raw_ratings):
        out, rep = cleaning.clean_ratings(raw_ratings)
        # 0.0 and 5.5 both fall outside [0.5, 5.0]
        assert out["rating"].between(0.5, 5.0).all()
        assert any("outside" in i for i in rep.issues)

    def test_sorted_by_user_then_time(self, raw_ratings):
        out, _ = cleaning.clean_ratings(raw_ratings)
        for uid, grp in out.groupby("userId"):
            assert grp["timestamp"].is_monotonic_increasing

    def test_keeps_first_occurrence_on_dedupe(self, raw_ratings):
        out, _ = cleaning.clean_ratings(raw_ratings)
        # (1, 10): first timestamp is 50 -> rating 3.5 kept
        row = out[(out["userId"] == 1) & (out["movieId"] == 10)]
        assert row["rating"].iloc[0] == 3.5


class TestCleanTags:
    def test_normalization(self, raw_tags):
        out, _ = cleaning.clean_tags(raw_tags)
        # '  Sci-Fi  ' and 'sci-fi' collapse to 'sci-fi' after dedupe
        assert sorted(out["tag"]) == ["mind-bending movie", "sci-fi"]

    def test_empty_tag_dropped(self, raw_tags):
        out, rep = cleaning.clean_tags(raw_tags)
        assert (out["tag"] != "").all()
        assert any("empty" in i for i in rep.issues)


class TestCleanGenome:
    def test_invalid_relevance_removed(self, raw_genome):
        scores, tags = raw_genome
        out, rep = cleaning.clean_genome(scores, tags)
        assert len(out) == 2
        assert out["relevance"].between(0.0, 1.0).all()

    def test_tag_names_joined(self, raw_genome):
        scores, tags = raw_genome
        out, _ = cleaning.clean_genome(scores, tags)
        tag_by_id = dict(zip(out["tagId"], out["tag"]))
        assert tag_by_id[1] == "sci-fi"  # tagId 2 row was invalid-relevance, dropped
        assert 2 not in tag_by_id


class TestOutliers:
    def test_outlier_summary_shape(self, raw_ratings):
        summary = cleaning.summarize_outliers(raw_ratings)
        assert set(summary) == {
            "p25", "p75", "iqr_upper_bound",
            "share_users_above_bound", "max_ratings_by_user",
        }
        assert summary["max_ratings_by_user"] >= 1
