"""Content feature engineering: movie content vectors from genome tags.

Representation (M2 decision):
- Each movie is a sparse vector over 1,128 genome tags, values = relevance.
- L2-normalized so cosine similarity is a simple dot product.
- Genres are appended as separate binary dimensions with a configurable
  weight (genres are the most human-interpretable signal, so they deserve
  explicit presence in explanations even though the genome already
  encodes them implicitly).

The matrix is precomputed once and cached to models/ — never recomputed
per request (spec section 28).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils.logging_config import get_logger

log = get_logger(__name__)

GENRE_WEIGHT = 0.5  # genre dimensions vs genome dimensions, both L2-normalized


def build_movie_feature_matrix(
    genome: pd.DataFrame,
    movies: pd.DataFrame,
    genre_weight: float = GENRE_WEIGHT,
) -> tuple[sp.csr_matrix, pd.DataFrame, dict[str, int]]:
    """Build the movie x feature sparse matrix.

    genome: processed genome table [movieId, tagId, tag, relevance]
            (already thresholded at 0.5 by the pipeline).
    movies: processed movies table with movieId, genres_list.

    Returns:
      matrix: (n_movies, n_tags + n_genres) CSR, L2-normalized rows
      movie_index: DataFrame [movieId, row] for id -> row lookup
      feature_names: {feature_name: column_index} for explanations
    """
    # Stable, reproducible row/col orderings
    movie_ids = np.sort(movies["movieId"].unique())
    row_pos = {mid: i for i, mid in enumerate(movie_ids)}

    # --- Genome tag block ---
    # Keep only movies present in the movies table (referential integrity)
    genome = genome[genome["movieId"].isin(row_pos)]
    tag_ids = np.sort(genome["tagId"].unique())
    col_pos = {tid: j for j, tid in enumerate(tag_ids)}

    rows = genome["movieId"].map(row_pos).to_numpy()
    cols = genome["tagId"].map(col_pos).to_numpy()
    vals = genome["relevance"].to_numpy(dtype=np.float32)

    tag_block = sp.csr_matrix(
        (vals, (rows, cols)), shape=(len(movie_ids), len(tag_ids)), dtype=np.float32
    )

    # --- Genre block (binary, weighted) ---
    all_genres = sorted(
        {g for gs in movies["genres_list"] for g in gs if g}
    )
    genre_pos = {g: j for j, g in enumerate(all_genres)}
    feature_names: dict[str, int] = {}
    for tid, j in col_pos.items():
        feature_names[f"tag:{tid}"] = j  # tag names resolved by caller via genome_tags

    genre_rows: list[int] = []
    genre_cols: list[int] = []
    for mid, gs in zip(movies["movieId"], movies["genres_list"]):
        for g in gs:
            if g in genre_pos:
                genre_rows.append(row_pos[mid])
                genre_cols.append(genre_pos[g])  # local index; offset applied on hstack

    genre_block = sp.csr_matrix(
        (
            np.full(len(genre_rows), genre_weight, dtype=np.float32),
            (genre_rows, genre_cols),
        ),
        shape=(len(movie_ids), len(all_genres)),
        dtype=np.float32,
    )
    for g, j in genre_pos.items():
        feature_names[f"genre:{g}"] = len(tag_ids) + j

    matrix = sp.hstack([tag_block, genre_block]).tocsr()

    # --- L2 normalize rows (guard zero rows: movies with no content) ---
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
    zero_rows = norms == 0
    if zero_rows.any():
        log.warning("%d movies have empty content vectors (no genome/genre data).", zero_rows.sum())
        norms[zero_rows] = 1.0
    inv = sp.diags(1.0 / norms, dtype=np.float32)
    matrix = inv @ matrix

    movie_index = pd.DataFrame({"movieId": movie_ids, "row": np.arange(len(movie_ids))})
    return matrix.tocsr(), movie_index, feature_names


def cosine_top_k(
    matrix: sp.csr_matrix,
    query_rows: np.ndarray,
    k: int,
    exclude: sp.csr_matrix | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k most similar rows for each query row, via normalized dot product.

    matrix: L2-normalized CSR (rows = movies)
    query_rows: row indices to query
    exclude: optional boolean-sparse (n_queries, n_movies) mask of positions
             to drop (e.g., the query movie itself)

    Returns (sims, indices) of shape (n_queries, k).
    """
    sims = matrix[query_rows] @ matrix.T  # (n_queries, n_movies)
    sims = sims.toarray()
    if exclude is not None:
        sims[exclude.toarray().astype(bool)] = -np.inf
    k = min(k, sims.shape[1] - 1)
    top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    # sort each row by score desc
    row_sims = np.take_along_axis(sims, top_idx, axis=1)
    order = np.argsort(-row_sims, axis=1)
    return np.take_along_axis(row_sims, order, axis=1), np.take_along_axis(top_idx, order, axis=1)
