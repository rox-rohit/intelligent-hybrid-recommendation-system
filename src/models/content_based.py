"""Content-based recommender over genome-tag movie vectors.

The public API is recommend_similar_movies(movie_id, k) which returns
records with movie, similarity score, and a *real* explanation naming the
shared genome tags / genres that drove the score (spec section 7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.features.content_features import build_movie_feature_matrix, cosine_top_k
from src.utils.logging_config import get_logger

log = get_logger(__name__)

MAX_EXPLANATION_FEATURES = 3  # keep explanations human-sized


@dataclass
class SimilarMovie:
    movie_id: int
    title: str | None
    similarity: float
    shared_features: list[str]  # names of the features driving the score


class ContentBasedRecommender:
    """KNN-style content recommender over the movie feature matrix."""

    def __init__(
        self,
        matrix: sp.csr_matrix,
        movie_index: pd.DataFrame,
        feature_names: dict[str, int],
        tag_id_to_name: pd.Series | dict | None = None,
        titles: pd.Series | dict | None = None,
    ):
        self.matrix = matrix
        self.movie_index = movie_index
        self._row_by_id = dict(zip(movie_index["movieId"], movie_index["row"]))
        self.feature_names = feature_names
        self._col_by_name = {v: k for k, v in feature_names.items()}
        self._tag_names = (
            tag_id_to_name if tag_id_to_name is not None else {}
        )
        self._titles = titles if titles is not None else {}

    @classmethod
    def from_processed_data(
        cls, genome: pd.DataFrame, movies: pd.DataFrame, genome_tags: pd.DataFrame
    ) -> "ContentBasedRecommender":
        """Build from the processed tables produced by the pipeline."""
        matrix, movie_index, feature_names = build_movie_feature_matrix(genome, movies)
        tag_names = dict(zip(genome_tags["tagId"], genome_tags["tag"]))
        # Resolve 'tag:{tagId}' feature names to readable tag strings once.
        readable: dict[str, int] = {}
        for name, col in feature_names.items():
            if name.startswith("tag:"):
                readable[str(tag_names.get(int(name[4:]), name))] = col
            else:
                readable[name.removeprefix("genre:")] = col
        titles = dict(zip(movies["movieId"], movies["title_clean"]))
        return cls(matrix, movie_index, readable, tag_names, titles)

    # ------------------------------------------------------------------ #
    def _explain(
        self, query_row: int, other_row: int, min_contrib: float = 0.05
    ) -> list[str]:
        """Name the top shared features between two movie vectors.

        contribution(j) = query[j] * other[j]; the largest products are
        exactly the features that drove the cosine score.
        """
        q = self.matrix[query_row].toarray().ravel()
        o = self.matrix[other_row].toarray().ravel()
        contrib = q * o
        top = np.argsort(-contrib)[: MAX_EXPLANATION_FEATURES + 3]
        shared = []
        for j in top:
            if contrib[j] < min_contrib:
                break
            name = self._col_by_name.get(int(j))
            if name and name not in shared:
                shared.append(name)
            if len(shared) >= MAX_EXPLANATION_FEATURES:
                break
        return shared

    def recommend_similar_movies(
        self, movie_id: int, k: int = 10
    ) -> list[SimilarMovie]:
        """Return the k most content-similar movies with real explanations."""
        if movie_id not in self._row_by_id:
            raise KeyError(f"Unknown movie_id: {movie_id}")
        query_row = self._row_by_id[movie_id]

        n = self.matrix.shape[0]
        exclude = sp.csr_matrix(
            ([True], ([0], [query_row])), shape=(1, n)
        )
        sims, idxs = cosine_top_k(self.matrix, np.array([query_row]), k=k, exclude=exclude)

        results: list[SimilarMovie] = []
        for sim, row in zip(sims[0], idxs[0]):
            mid = int(self.movie_index.iloc[row]["movieId"])
            if self.matrix[row].nnz == 0:
                shared = []  # no content signal; do not fake an explanation
            else:
                shared = self._explain(query_row, int(row))
            results.append(
                SimilarMovie(
                    movie_id=mid,
                    title=self._titles.get(mid),
                    similarity=float(sim),
                    shared_features=shared,
                )
            )
        return results

    def explain_text(self, rec: SimilarMovie) -> str:
        """Render a SimilarMovie explanation as a user-facing sentence."""
        if not rec.shared_features:
            return "Recommended by overall content similarity."
        feats = ", ".join(rec.shared_features)
        return (
            f"Shares {feats} with your selected movie "
            f"({rec.similarity:.0%} content match)."
        )
