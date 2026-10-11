"""Unit tests for the content-based recommender (synthetic fixtures)."""

import pandas as pd
import pytest

from src.models.content_based import ContentBasedRecommender


@pytest.fixture
def recommender():
    genome = pd.DataFrame(
        {
            "movieId": [1, 1, 2, 2, 4],
            "tagId": [10, 11, 10, 12, 10],
            "relevance": [0.9, 0.8, 0.85, 0.7, 0.6],
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title_clean": ["Star Quest", "Galaxy Hour", "Empty Film", "Laugh Time"],
            "genres_list": [["Sci-Fi"], ["Sci-Fi", "Drama"], [], ["Comedy"]],
        }
    )
    genome_tags = pd.DataFrame(
        {"tagId": [10, 11, 12], "tag": ["sci-fi", "space travel", "drama"]}
    )
    return ContentBasedRecommender.from_processed_data(genome, movies, genome_tags)


class TestSimilarMovies:
    def test_returns_k_results(self, recommender):
        recs = recommender.recommend_similar_movies(1, k=2)
        assert len(recs) == 2

    def test_self_not_included(self, recommender):
        recs = recommender.recommend_similar_movies(1, k=3)
        assert all(r.movie_id != 1 for r in recs)

    def test_most_similar_first(self, recommender):
        recs = recommender.recommend_similar_movies(1, k=3)
        sims = [r.similarity for r in recs]
        assert sims == sorted(sims, reverse=True)
        # movie 2 shares sci-fi tag + Sci-Fi genre -> strongest match
        assert recs[0].movie_id == 2

    def test_scores_in_unit_range(self, recommender):
        for r in recommender.recommend_similar_movies(1, k=3):
            assert 0 <= r.similarity <= 1.0001

    def test_explanation_names_real_features(self, recommender):
        recs = recommender.recommend_similar_movies(1, k=3)
        top = recs[0]  # movie 2
        assert "sci-fi" in top.shared_features
        assert all(f in {"sci-fi", "space travel", "Sci-Fi", "Drama"} for f in top.shared_features)

    def test_empty_movie_has_no_fake_explanation(self, recommender):
        # movie 3 has no content; whoever gets zero-sim gets no explanation
        recs = recommender.recommend_similar_movies(3, k=3)
        for r in recs:
            if r.similarity == 0:
                assert r.shared_features == []

    def test_unknown_movie_raises(self, recommender):
        with pytest.raises(KeyError):
            recommender.recommend_similar_movies(999, k=5)


class TestExplanationText:
    def test_text_includes_features_and_score(self, recommender):
        recs = recommender.recommend_similar_movies(1, k=1)
        text = recommender.explain_text(recs[0])
        assert "sci-fi" in text
        assert "%" in text

    def test_text_fallback_without_features(self, recommender):
        from src.models.content_based import SimilarMovie

        fallback = recommender.explain_text(
            SimilarMovie(3, None, 0.0, [])
        )
        assert "content similarity" in fallback
