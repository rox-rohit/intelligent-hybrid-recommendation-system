"""Unit tests for content feature engineering (synthetic fixtures)."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.features.content_features import build_movie_feature_matrix, cosine_top_k


@pytest.fixture
def genome():
    # 3 movies, 3 tags. Movie 3 has no genome data (cold-start-ish).
    return pd.DataFrame(
        {
            "movieId": [1, 1, 2, 2, 4],
            "tagId": [10, 11, 10, 12, 10],
            "tag": ["sci-fi", "space", "sci-fi", "drama", "sci-fi"],
            "relevance": [0.9, 0.8, 0.85, 0.7, 0.6],
        }
    )


@pytest.fixture
def movies():
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "genres_list": [["Sci-Fi"], ["Sci-Fi", "Drama"], [], ["Comedy"]],
        }
    )


class TestBuildMatrix:
    def test_shape_and_lookup(self, genome, movies):
        matrix, movie_index, feats = build_movie_feature_matrix(genome, movies)
        assert matrix.shape[0] == 4  # 4 unique movies
        assert sp.issparse(matrix)
        assert set(movie_index["movieId"]) == {1, 2, 3, 4}
        assert movie_index["row"].is_monotonic_increasing

    def test_rows_l2_normalized(self, genome, movies):
        matrix, movie_index, _ = build_movie_feature_matrix(genome, movies)
        norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
        has_content = matrix.getnnz(axis=1) > 0
        assert np.allclose(norms[has_content], 1.0, atol=1e-5)
        # movies with no content features at all keep a zero vector
        assert np.allclose(norms[~has_content], 0.0)

    def test_empty_movie_has_zero_vector(self, genome, movies):
        # movie 3 has neither genome nor genre data
        matrix, movie_index, _ = build_movie_feature_matrix(genome, movies)
        row = movie_index.loc[movie_index["movieId"] == 3, "row"].iloc[0]
        assert matrix[row].nnz == 0

    def test_feature_names_cover_genres(self, genome, movies):
        _, _, feats = build_movie_feature_matrix(genome, movies)
        assert "genre:Sci-Fi" in feats
        assert "genre:Comedy" in feats
        # genome tag columns come before genre columns
        assert feats["genre:Sci-Fi"] > feats.get("tag:10", -1)


class TestCosineTopK:
    def test_most_similar_pair(self, genome, movies):
        matrix, movie_index, _ = build_movie_feature_matrix(genome, movies)
        rows = movie_index.set_index("movieId")["row"]
        q = np.array([rows[1]])
        n = matrix.shape[0]
        mask = sp.csr_matrix(([True], (np.zeros(1, dtype=int), [rows[1]])), shape=(1, n))
        sims, idxs = cosine_top_k(matrix, q, k=3, exclude=mask)
        # movies 1 and 2 share tag 10 (sci-fi) + genre Sci-Fi -> top match
        best = movie_index.iloc[idxs[0, 0]]["movieId"]
        assert best == 2
        assert 0 < sims[0, 0] <= 1.0001

    def test_self_excluded_when_masked(self, genome, movies):
        matrix, movie_index, _ = build_movie_feature_matrix(genome, movies)
        rows = movie_index.set_index("movieId")["row"]
        q = np.array([rows[1]])
        n = matrix.shape[0]
        mask = sp.csr_matrix(([True], (q, [rows[1]])), shape=(1, n))
        sims, idxs = cosine_top_k(matrix, q, k=3, exclude=mask)
        assert movie_index.iloc[idxs[0, 0]]["movieId"] != 1

    def test_k_capped_by_catalog(self, genome, movies):
        matrix, _, _ = build_movie_feature_matrix(genome, movies)
        sims, idxs = cosine_top_k(matrix, np.array([0]), k=100)
        assert idxs.shape[1] == matrix.shape[0] - 1

    def test_identical_movies_similar_to_1(self):
        matrix = sp.csr_matrix(np.eye(2, dtype=np.float32))
        sims, idxs = cosine_top_k(matrix, np.array([0]), k=1)
        assert np.isclose(sims[0, 0], 1.0, atol=1e-5)
