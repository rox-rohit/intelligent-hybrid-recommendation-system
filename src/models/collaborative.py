"""Collaborative filtering: item-based kNN and implicit-feedback ALS.

Two complementary algorithms (spec section 8 — never rely on one):

1. ItemKNN — cosine similarity between item column vectors of the
   user-item confidence matrix. Recommends movies liked by users who
   liked the target movie. Memory-based, transparent, strong on
   popular items; no training loop.

2. ImplicitALS — Hu, Koren & Volinsky (2008) confidence-weighted
   matrix factorization. Explicit ratings are converted to implicit
   confidence (rating, timestamp recency) — see features/interactions.py
   (M6). Learns latent user/item factors; handles sparse data and
   tail items better than kNN.

Both consume a CSR user-item matrix of *implicit confidence weights*
(>= 0), not raw star ratings.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.utils.logging_config import get_logger

log = get_logger(__name__)


class ItemKNN:
    """Item-based collaborative filtering with truncated cosine similarities.

    Memory-bounded design: instead of materializing the full (n_items x
    n_items) similarity matrix (~900M nonzeros for MovieLens 25M — several
    GiB), similarities are computed chunk-wise and only the top
    `top_k_similar` neighbors per item are kept. This is the standard
    production approach; a full dense kNN graph is neither needed nor
    affordable.
    """

    def __init__(
        self,
        user_item: sp.csr_matrix,
        shrinkage: int = 0,
        top_k_similar: int = 200,
        chunk_size: int = 512,
    ):
        """user_item: CSR (n_users, n_items) of confidence weights >= 0."""
        if user_item.format != "csr":
            user_item = user_item.tocsr()
        self.X = user_item.astype(np.float32)
        self.n_items = self.X.shape[1]
        self.top_k = min(top_k_similar, self.n_items - 1)

        # Column-normalize once: dot products are then cosine similarities.
        col_norms = np.sqrt(self.X.multiply(self.X).sum(axis=0)).A.ravel()
        col_norms[col_norms == 0] = 1.0
        Xn = (self.X @ sp.diags(1.0 / col_norms, dtype=np.float32)).tocsr()

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        data: list[np.ndarray] = []

        # Chunk over query items: peak memory = chunk x n_items dense block
        for start in range(0, self.n_items, chunk_size):
            end = min(start + chunk_size, self.n_items)
            block = (Xn.T[start:end] @ Xn).toarray()  # (chunk, n_items)
            diag = np.arange(end - start)  # local indices within the chunk
            block[diag, diag + start] = -np.inf  # self-similarity masked

            k = min(self.top_k, self.n_items - 1)
            top = np.argpartition(-block, kth=k - 1, axis=1)[:, :k]
            scores = np.take_along_axis(block, top, axis=1)
            order = np.argsort(-scores, axis=1)
            top = np.take_along_axis(top, order, axis=1)
            scores = np.take_along_axis(scores, order, axis=1)

            keep = scores > 0  # drop zero/negative similarity neighbors
            nz_r, nz_c = np.nonzero(keep)
            rows.append(nz_r + start)  # offset back to global item ids
            cols.append(top[keep])
            data.append(scores[keep].astype(np.float32))

        sims = sp.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_items, self.n_items),
            dtype=np.float32,
        )

        if shrinkage:
            # Shrink similarities by interaction-count evidence
            # (sim * n / (n + shrinkage)) to demote co-occurrence on tiny
            # support. Sparse-safe: arithmetic on .data directly.
            co_counts = (self.X.astype(bool).T @ self.X.astype(bool)).tocsr()
            denom = co_counts.copy()
            denom.data = denom.data + shrinkage
            inv = co_counts.copy()
            inv.data = 1.0 / denom.data
            sims = sims.multiply(co_counts).multiply(inv).tocsr()

        self.similarities = sims  # truncated (n_items, n_items) CSR

    def similar_items(self, item_id: int, k: int = 10) -> list[tuple[int, float]]:
        """k most similar items to `item_id` (excluding itself)."""
        row = self.similarities.getrow(item_id).toarray().ravel()
        row[item_id] = -np.inf  # never recommend the item itself
        k = min(k, row.size - 1)
        top = np.argpartition(-row, kth=k - 1)[:k]
        top = top[np.argsort(-row[top])]
        return [(int(i), float(row[i])) for i in top]

    def recommend_for_user(
        self,
        user_id: int,
        k: int = 10,
        seen_items: set[int] | None = None,
        item_index: pd.Series | dict | None = None,
    ) -> list[tuple[int, float]]:
        """Score items by similarity-weighted sum of the user's history.

        score(u, i) = sum_j conf(u, j) * sim(i, j)  over the user's items j.

        `seen_items` should be a set of EXTERNAL movieIds (the same IDs
        used in the original data). If provided, `item_index` must be a
        mapping from external movieId -> internal column id (as returned
        by the confidence-matrix builder).
        """
        user_row = self.X.getrow(user_id)
        if user_row.nnz == 0:
            return []  # cold user — caller falls back to content/popularity
        hist_items = user_row.indices
        hist_conf = user_row.data

        scores = np.asarray(
            self.similarities[hist_items].multiply(hist_conf[:, None]).sum(axis=0)
        ).ravel()

        if seen_items and item_index is not None:
            seen_internal = {item_index[m] for m in seen_items if m in item_index}
            if seen_internal:
                scores[list(seen_internal)] = -np.inf

        k = min(k, scores.size - 1)
        top = np.argpartition(-scores, kth=k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


class ImplicitALSModel:
    """Confidence-weighted ALS via the `implicit` library (iALS)."""

    def __init__(self, factors: int = 64, iterations: int = 20,
                 regularization: float = 0.05, random_state: int = 42):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.random_state = random_state
        self.model = None

    def fit(self, user_item: sp.csr_matrix) -> "ImplicitALSModel":
        """Fit on a CSR matrix of implicit confidence weights (>= 0)."""
        import implicit  # deferred: heavy import stays out of module load

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.random_state,
        )
        # implicit >= 0.7: fit() expects the (users, items) CSR matrix
        self.model.fit(user_item.tocsr())
        return self

    def recommend_for_user(
        self,
        user_id: int,
        k: int = 10,
        user_items: sp.csr_matrix | None = None,
        filter_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Top-k items for a user. user_items: the fit matrix row source."""
        if self.model is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        ids, scores = self.model.recommend(
            user_id,
            user_items,
            N=k,
            filter_already_liked_items=filter_seen,
        )
        return [(int(i), float(s)) for i, s in zip(ids, scores)]

    def similar_items(self, item_id: int, k: int = 10) -> list[tuple[int, float]]:
        """Top-k most similar items by factor cosine (self excluded)."""
        if self.model is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        ids, scores = self.model.similar_items(item_id, N=k + 1)
        pairs = [(int(i), float(s)) for i, s in zip(ids, scores) if int(i) != item_id]
        return pairs[:k]
