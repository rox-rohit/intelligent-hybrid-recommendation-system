"""MMR (Maximal Marginal Relevance) for diversity (Milestone 7).

MMR balances relevance vs diversity:
  score(i) = λ * relevance(i) - (1-λ) * max_{j in selected} similarity(i, j)

λ=1.0 → pure relevance
λ=0.5 → balanced
λ=0.0 → pure diversity
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from src.utils.logging_config import get_logger

log = get_logger(__name__)


def mmr_rerank(
    candidates: list[tuple[int, float]],           # (movie_id, relevance_score)
    content_matrix: sp.csr_matrix,                  # L2-normalized movie vectors
    movie_index: pd.DataFrame,                      # movieId -> row
    lambda_param: float = 0.5,
    k: int = 10,
    exclude: Optional[set[int]] = None,
) -> list[tuple[int, float]]:
    """Re-rank candidates using MMR for diversity.

    Args:
        candidates: List of (movie_id, relevance_score) from hybrid/ranker
        content_matrix: L2-normalized movie feature matrix
        movie_index: DataFrame with movieId, row columns
        lambda_param: Relevance vs diversity tradeoff [0, 1]
        k: Number of items to return
        exclude: Set of movie_ids to exclude from selection

    Returns:
        List of (movie_id, mmr_score) of length up to k
    """
    if not candidates:
        return []

    if exclude is None:
        exclude = set()

    # Filter excluded
    candidates = [(mid, score) for mid, score in candidates if mid not in exclude]
    if not candidates:
        return []

    # Map movie_id -> matrix row
    row_map = dict(zip(movie_index["movieId"], movie_index["row"]))

    # Build candidate matrix
    valid_candidates = []
    valid_rows = []
    valid_scores = []

    for mid, score in candidates:
        if mid in row_map:
            valid_candidates.append(mid)
            valid_rows.append(row_map[mid])
            valid_scores.append(score)

    if not valid_candidates:
        return []

    cand_matrix = content_matrix[valid_rows]  # (n_candidates, n_features)
    n = len(valid_candidates)

    # Precompute all pairwise similarities between candidates
    # Since matrix is L2-normalized, cosine = dot product
    sim_matrix = (cand_matrix @ cand_matrix.T).toarray()  # (n, n)
    np.fill_diagonal(sim_matrix, -1.0)  # Exclude self

    # Relevance scores
    relevance = np.array(valid_scores, dtype=np.float32)

    # MMR selection
    selected_idx = []
    remaining_idx = set(range(n))

    # First item: highest relevance
    first_idx = int(np.argmax(relevance))
    selected_idx.append(first_idx)
    remaining_idx.remove(first_idx)

    while len(selected_idx) < min(k, n) and remaining_idx:
        best_idx = None
        best_mmr = -np.inf

        for idx in remaining_idx:
            # Max similarity to already selected
            max_sim = max(sim_matrix[idx, sel] for sel in selected_idx)

            mmr_score = lambda_param * relevance[idx] - (1 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

    return [(valid_candidates[i], float(relevance[i])) for i in selected_idx]


def mmr_rerank_with_genre(
    candidates: list[tuple[int, float]],
    content_matrix: sp.csr_matrix,
    movie_index: pd.DataFrame,
    movie_genres: dict[int, set[str]],
    lambda_param: float = 0.5,
    k: int = 10,
    genre_penalty: float = 0.3,
) -> list[tuple[int, float]]:
    """MMR with explicit genre diversity penalty.

    Adds penalty for genre overlap with already selected items.
    """
    if not candidates:
        return []

    row_map = dict(zip(movie_index["movieId"], movie_index["row"]))

    valid_candidates = []
    valid_rows = []
    valid_scores = []
    valid_genres = []

    for mid, score in candidates:
        if mid in row_map:
            valid_candidates.append(mid)
            valid_rows.append(row_map[mid])
            valid_scores.append(score)
            valid_genres.append(movie_genres.get(mid, set()))

    if not valid_candidates:
        return []

    cand_matrix = content_matrix[valid_rows]
    n = len(valid_candidates)

    # Content similarity matrix
    sim_matrix = (cand_matrix @ cand_matrix.T).toarray()
    np.fill_diagonal(sim_matrix, -1.0)

    # Genre overlap matrix
    genre_overlap = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            gi = valid_genres[i]
            gj = valid_genres[j]
            if gi and gj:
                overlap = len(gi & gj) / len(gi | gj)  # Jaccard
            else:
                overlap = 0.0
            genre_overlap[i, j] = overlap
            genre_overlap[j, i] = overlap

    relevance = np.array(valid_scores, dtype=np.float32)

    selected_idx = []
    remaining_idx = set(range(n))

    first_idx = int(np.argmax(relevance))
    selected_idx.append(first_idx)
    remaining_idx.remove(first_idx)

    while len(selected_idx) < min(k, n) and remaining_idx:
        best_idx = None
        best_mmr = -np.inf

        for idx in remaining_idx:
            max_content_sim = max(sim_matrix[idx, sel] for sel in selected_idx)
            max_genre_overlap = max(genre_overlap[idx, sel] for sel in selected_idx)

            mmr_score = (
                lambda_param * relevance[idx]
                - (1 - lambda_param) * max_content_sim
                - genre_penalty * max_genre_overlap
            )

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

    return [(valid_candidates[i], float(relevance[i])) for i in selected_idx]


# Need pandas import
import pandas as pd