"""Unit tests for collaborative filtering models (synthetic fixtures)."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.models.collaborative import ItemKNN, ImplicitALSModel


@pytest.fixture
def conf_matrix():
    """4 users x 5 items. Users 0/1 share taste (items 0,1); users 2/3 share another."""
    X = sp.csr_matrix(
        np.array(
            [
                [1.0, 1.0, 0, 0, 0],
                [1.0, 1.0, 0, 0, 0.5],
                [0, 0, 1.0, 1.0, 0],
                [0, 0, 1.0, 1.0, 0.5],
            ],
            dtype=np.float32,
        )
    )
    return X


class TestItemKNN:
    def test_similar_items_shared_taste(self, conf_matrix):
        knn = ItemKNN(conf_matrix)
        sim = knn.similar_items(0, k=2)
        # item 1 co-occurs with item 0 across two users -> top match
        assert sim[0][0] == 1
        assert 0 < sim[0][1] <= 1.0

    def test_similar_excludes_self(self, conf_matrix):
        knn = ItemKNN(conf_matrix)
        ids = [i for i, _ in knn.similar_items(0, k=4)]
        assert 0 not in ids

    def test_recommend_for_user(self, conf_matrix):
        knn = ItemKNN(conf_matrix)
        # 5 items: indices 0-4. user 0 has seen 0,1 -> next should be 4 (user 1's minor)
        item_index = pd.Series({10: 0, 20: 1, 30: 2, 40: 3, 50: 4})  # dummy mapping
        recs = knn.recommend_for_user(0, k=3, seen_items={10, 20}, item_index=item_index)
        assert recs[0][0] == 4
        assert len(recs) == 3

    def test_cold_user_gets_empty(self, conf_matrix):
        knn = ItemKNN(conf_matrix)
        # grow the matrix with an all-zero user row
        X = sp.vstack([conf_matrix, sp.csr_matrix((1, conf_matrix.shape[1]))]).tocsr()
        knn = ItemKNN(X)
        assert knn.recommend_for_user(4, k=3) == []

    def test_shrinkage_demotes_sparse_cooccurrence(self, conf_matrix):
        plain = ItemKNN(conf_matrix).similarities
        shrunk = ItemKNN(conf_matrix, shrinkage=10).similarities
        # every non-zero pair has small support here, so all sims shrink
        nz_plain = plain.data[plain.data > 0]
        nz_shrunk = shrunk.data[shrunk.data > 0]
        assert len(nz_shrunk) > 0
        assert nz_shrunk.sum() < nz_plain.sum()

    def test_similarity_symmetric(self, conf_matrix):
        knn = ItemKNN(conf_matrix)
        s = knn.similarities.toarray()
        np.testing.assert_allclose(s, s.T, atol=1e-6)

    def test_partial_final_chunk_regression(self, conf_matrix):
        """Real-data regression: 59k items hit a partial final chunk and
        crashed on a global-vs-local diagonal index. Exercise chunking
        with a matrix bigger than one chunk (chunk_size=2, 5 items -> 3
        chunks, last one partial)."""
        knn = ItemKNN(conf_matrix, chunk_size=2, top_k_similar=3)
        # self-similarity must be masked everywhere, not just chunk-aligned
        s = knn.similarities.toarray()
        np.testing.assert_allclose(np.diag(s), 0.0, atol=1e-6)
        # and recommendations still work
        recs = knn.recommend_for_user(0, k=3, seen_items={0, 1})
        assert len(recs) == 3


class TestImplicitALS:
    @pytest.fixture(scope="class")
    def fitted(self):
        """Fit once (ALS is slower than kNN); reuse across tests."""
        X = sp.csr_matrix(
            np.array(
                [
                    [1.0, 1.0, 0, 0, 0],
                    [1.0, 1.0, 0, 0, 0.5],
                    [0, 0, 1.0, 1.0, 0],
                    [0, 0, 1.0, 1.0, 0.5],
                ],
                dtype=np.float32,
            )
        )
        model = ImplicitALSModel(factors=4, iterations=10, random_state=42)
        model.fit(X)
        # return both model and its training matrix for recommend()
        return model, X

    def test_fit_produces_factors(self, fitted):
        model, X = fitted
        assert model.model is not None
        # item_factors shape: (n_items, factors)
        assert model.model.item_factors.shape == (X.shape[1], 4)

    def test_recommend_for_user(self, fitted):
        model, X = fitted
        recs = model.recommend_for_user(0, k=3, user_items=X.tocsr()[:1])
        assert len(recs) == 3
        ids = [i for i, _ in recs]
        # user 0 already liked items 0,1 -> filtered out
        assert 0 not in ids and 1 not in ids

    def test_recommend_scores_finite(self, fitted):
        model, X = fitted
        recs = model.recommend_for_user(0, k=3, user_items=X.tocsr()[:1])
        for _, score in recs:
            assert np.isfinite(score)

    def test_similar_items(self, fitted):
        model, _ = fitted
        sim = model.similar_items(0, k=2)
        assert len(sim) == 2
        assert 0 not in [i for i, _ in sim]

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            ImplicitALSModel().recommend_for_user(0, k=2, user_items=None)
