"""Tune hybrid weights by grid search on validation NDCG@10 (spec section 9).

Pipeline:
  1. Time-aware split; train ItemKNN + ALS on TRAIN; popularity from TRAIN.
  2. For each sampled VALIDATION user, precompute each component's raw
     scores over a shared candidate pool (ALS top-200 ∪ ItemKNN top-200 ∪
     content-profile top-200 ∪ popular top-200), excluding seen items.
  3. Grid-search weight mixes on validation NDCG@10.
  4. Refit nothing (weights only); report test metrics for the tuned mix
     alongside each component alone — the honest "did the hybrid help?"
     table. Test items are touched exactly once, at the end.

Writes docs/results/hybrid_tuning.json.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_processed
from src.evaluation.metrics import evaluate_all
from src.evaluation.splitting import time_aware_split
from src.features.content_features import build_movie_feature_matrix
from src.features.interactions import build_confidence_matrix, derive_interactions
from src.models.collaborative import ImplicitALSModel, ItemKNN
from src.models.hybrid import HybridRecommender, HybridWeights
from src.models.popularity import PopularityModel
from src.utils.logging_config import configure_logging, get_logger
from src.utils.paths import DOCS_DIR

log = get_logger(__name__)

K = 10
EVAL_USERS = 3_000        # validation users sampled for weight tuning
RELEVANCE_THRESHOLD = 3.5
RANDOM_STATE = 42
CANDIDATE_POOL = 200      # per-component candidate retrieval depth
CONTENT_WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4]
POP_WEIGHT_GRID = [0.0, 0.05, 0.1, 0.2]


def _pack(recs_by_user, user_order, k):
    out = np.full((len(user_order), k), -1, dtype=np.int64)
    for i, uid in enumerate(user_order):
        for j, (item, _) in enumerate(recs_by_user[uid][:k]):
            out[i, j] = item
    return out


def main() -> int:
    configure_logging("INFO")
    t_start = time.perf_counter()

    # ---- 1. Load, split, train ------------------------------------------
    ratings = load_processed("ratings")
    movies = load_processed("movies")
    split = time_aware_split(ratings)
    log.info("Split sizes: train=%s val=%s test=%s",
             len(split.train), len(split.validation), len(split.test))
    del ratings

    train_inter = derive_interactions(split.train)
    user_index = pd.Series(
        np.arange(train_inter["userId"].nunique()),
        index=np.sort(train_inter["userId"].unique()),
    )
    item_index = pd.Series(
        np.arange(train_inter["movieId"].nunique()),
        index=np.sort(train_inter["movieId"].unique()),
    )
    conf = build_confidence_matrix(
        train_inter, len(user_index), len(item_index), user_index, item_index
    )
    item_by_col = {int(c): int(mid) for mid, c in item_index.items()}
    col_by_item = {int(mid): int(c) for mid, c in item_index.items()}

    log.info("Training ItemKNN ...")
    knn = ItemKNN(conf)

    log.info("Training ALS ...")
    als = ImplicitALSModel(factors=64, iterations=15, random_state=RANDOM_STATE)
    als.fit(conf)

    movie_stats = split.train.groupby("movieId").agg(
        mean_rating=("rating", "mean"), n_ratings=("rating", "size")
    )
    movies_idx = movies.set_index("movieId")
    movie_stats["release_year"] = movies_idx["release_year"]
    pop = PopularityModel(movie_stats)
    pop_scores_norm = pop.popularity  # movieId -> [0, 1]

    # content: user profile = weighted centroid of genome vectors of liked
    # train items (rating >= 3.5); score = cosine to candidate vectors.
    from src.features.content_features import build_movie_feature_matrix
    genome = load_processed("genome_scores")
    genome_tags = load_processed("genome_tags")
    content_matrix, content_index, _ = build_movie_feature_matrix(genome, movies)
    content_row_by_id = dict(zip(content_index["movieId"], content_index["row"]))

    # ---- 2. Validation users + precomputed component scores --------------
    rng = np.random.default_rng(RANDOM_STATE)
    val_users_all = sorted(split.validation["userId"].unique())
    n_sample = min(EVAL_USERS, len(val_users_all))
    val_users = sorted(rng.choice(val_users_all, n_sample, replace=False).tolist())

    val_rel = split.validation[split.validation["rating"] >= RELEVANCE_THRESHOLD]
    truth_val: dict[int, set[int]] = {}
    for uid, grp in val_rel.groupby("userId"):
        truth_val[int(uid)] = set(grp["movieId"])
    val_users = [u for u in val_users if truth_val.get(u)]

    wanted = set(val_users)
    seen_by_user = {
        int(uid): set(grp["movieId"])
        for uid, grp in split.train.groupby("userId")
        if int(uid) in wanted
    }

    log.info("Precomputing component scores for %d validation users ...", len(val_users))
    cf_scores: dict[int, dict[int, float]] = {}    # ALS
    knn_scores: dict[int, dict[int, float]] = {}
    ct_scores: dict[int, dict[int, float]] = {}
    pop_scores: dict[int, dict[int, float]] = {}

    for uid in val_users:
        urow = int(user_index.loc[uid])
        seen = seen_by_user.get(uid, set())

        # ALS: top-200 raw scores (internal ids -> external)
        ids, scores = als.model.recommend(
            urow, conf[urow], N=CANDIDATE_POOL,
            filter_already_liked_items=True,
        )
        cf_scores[uid] = {
            item_by_col.get(int(i), int(i)): float(s)
            for i, s in zip(ids, scores)
            if item_by_col.get(int(i), int(i)) not in seen
        }

        # ItemKNN: top-200
        knn_scores[uid] = dict(
            knn.recommend_for_user(urow, k=CANDIDATE_POOL, seen_items=seen,
                                   item_index=item_index)
        )

        # Content: centroid of liked train items vs candidate pool = ALS ∪ KNN ∪ popular
        liked = [
            content_row_by_id[m]
            for m, r in split.train[split.train["userId"] == uid][["movieId", "rating"]].itertuples(index=False)
            if r >= RELEVANCE_THRESHOLD and m in content_row_by_id
        ]
        if liked:
            prof = np.asarray(content_matrix[liked].sum(axis=0)).ravel()
            norm = np.linalg.norm(prof)
            if norm > 0:
                prof = prof / norm
                pool = set(cf_scores[uid]) | set(knn_scores[uid])
                pool_cols = [col_by_item[m] for m in pool if m in col_by_item]
                if pool_cols:
                    cand_vecs = content_matrix[[content_row_by_id.get(m, -1) for m in pool]].toarray() \
                        if all(m in content_row_by_id for m in pool) else None
                # score pool items by cosine to profile
                ct = {}
                for m in pool:
                    row = content_row_by_id.get(m)
                    if row is None:
                        continue
                    v = content_matrix[row].toarray().ravel()
                    vn = np.linalg.norm(v)
                    if vn > 0:
                        ct[m] = float(prof @ v / vn)
                ct_scores[uid] = ct

        # Popularity over the same pool (constant per item)
        pop_scores[uid] = {m: float(pop_scores_norm.get(m, 0.0))
                           for m in set(cf_scores[uid]) | set(knn_scores[uid]) | set(ct_scores.get(uid, {}))}

    # ---- 3. Grid search on validation NDCG@10 ----------------------------
    log.info("Grid-searching hybrid weights ...")
    grid_results = []
    best = None
    for w_ct, w_pop in product(CONTENT_WEIGHT_GRID, POP_WEIGHT_GRID):
        w_cf = 1.0 - w_ct - w_pop
        if w_cf < 0:
            continue
        weights = HybridWeights(cf=w_cf, content=w_ct, popularity=w_pop)
        recs_by_user = {}
        for uid in val_users:
            hy = HybridRecommender(
                cf_scorer=lambda u, _cf=cf_scores: _cf.get(u, {}),
                content_scorer=lambda u, _ct=ct_scores: _ct.get(u, {}),
                popularity_scorer=lambda u, _p=pop_scores: _p.get(u, {}),
                weights=weights,
            )
            recs_by_user[uid] = hy.recommend(uid, k=K)
        recs_arr = _pack(recs_by_user, val_users, K)
        m = evaluate_all(recs_arr, {u: truth_val[u] for u in val_users}, k=K)
        row = {"weights": weights.as_tuple(), **m}
        grid_results.append(row)
        if best is None or m[f"ndcg@{K}"] > best[1]:
            best = (weights.as_tuple(), m[f"ndcg@{K}"])
        log.info("weights=%s -> val NDCG@10=%.4f", weights.as_tuple(), m[f"ndcg@{K}"])

    log.info("Best validation mix: %s (NDCG@10=%.4f)", best[0], best[1])

    # ---- 4. Test evaluation of tuned mix + components ---------------------
    log.info("Evaluating tuned hybrid on TEST ...")
    test_rel = split.test[split.test["rating"] >= RELEVANCE_THRESHOLD]
    test_users_all = sorted(split.test["userId"].unique())
    n_test = min(EVAL_USERS, len(test_users_all))
    test_users = sorted(rng.choice(test_users_all, n_test, replace=False).tolist())
    truth_test: dict[int, set[int]] = {}
    for uid, grp in test_rel.groupby("userId"):
        truth_test[int(uid)] = set(grp["movieId"])
    test_users = [u for u in test_users if truth_test.get(u)]

    # same precompute for test users (models unchanged — no retraining,
    # so this is a legitimate final test pass)
    cf_t, knn_t, ct_t, pop_t = {}, {}, {}, {}
    wanted = set(test_users)
    seen_test = {
        int(uid): set(grp["movieId"])
        for uid, grp in pd.concat([split.train, split.validation]).groupby("userId")
        if int(uid) in wanted
    }
    for uid in test_users:
        urow = int(user_index.loc[uid])
        seen = seen_test.get(uid, set())
        ids, scores = als.model.recommend(
            urow, conf[urow], N=CANDIDATE_POOL, filter_already_liked_items=True
        )
        cf_t[uid] = {item_by_col.get(int(i), int(i)): float(s)
                     for i, s in zip(ids, scores)
                     if item_by_col.get(int(i), int(i)) not in seen}
        knn_t[uid] = dict(knn.recommend_for_user(urow, k=CANDIDATE_POOL,
                                                 seen_items=seen, item_index=item_index))
        liked = [
            content_row_by_id[m]
            for m, r in split.train[split.train["userId"] == uid][["movieId", "rating"]].itertuples(index=False)
            if r >= RELEVANCE_THRESHOLD and m in content_row_by_id
        ]
        if liked:
            prof = np.asarray(content_matrix[liked].sum(axis=0)).ravel()
            norm = np.linalg.norm(prof)
            if norm > 0:
                prof = prof / norm
                pool = set(cf_t[uid]) | set(knn_t[uid])
                ct = {}
                for m in pool:
                    row = content_row_by_id.get(m)
                    if row is None:
                        continue
                    v = content_matrix[row].toarray().ravel()
                    vn = np.linalg.norm(v)
                    if vn > 0:
                        ct[m] = float(prof @ v / vn)
                ct_t[uid] = ct
        pop_t[uid] = {m: float(pop_scores_norm.get(m, 0.0))
                      for m in set(cf_t[uid]) | set(knn_t[uid]) | set(ct_t.get(uid, {}))}

    results: dict[str, dict] = {}

    def eval_set(name, recs_by_user, users):
        recs_arr = _pack(recs_by_user, users, K)
        results[name] = evaluate_all(recs_arr, {u: truth_test[u] for u in users}, k=K)

    tuned = HybridWeights(*best[0])
    hy = HybridRecommender(
        cf_scorer=lambda u: cf_t.get(u, {}),
        content_scorer=lambda u: ct_t.get(u, {}),
        popularity_scorer=lambda u: pop_t.get(u, {}),
        weights=tuned,
    )
    eval_set("hybrid_tuned", {u: hy.recommend(u, k=K) for u in test_users}, test_users)
    eval_set("als_only", {u: sorted(cf_t[u].items(), key=lambda x: -x[1])[:K]
                          if cf_t.get(u) else [] for u in test_users}, test_users)
    eval_set("itemknn_only", {u: sorted(knn_t[u].items(), key=lambda x: -x[1])[:K]
                              if knn_t.get(u) else [] for u in test_users}, test_users)
    eval_set("content_only", {u: sorted(ct_t[u].items(), key=lambda x: -x[1])[:K]
                              if ct_t.get(u) else [] for u in test_users}, test_users)

    # ---- 5. Persist ------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    payload = {
        "experiment": "hybrid_weight_tuning",
        "k": K,
        "n_val_users": len(val_users),
        "n_test_users": len(test_users),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "best_weights": {"cf": best[0][0], "content": best[0][1], "popularity": best[0][2]},
        "best_val_ndcg@10": best[1],
        "grid": grid_results,
        "test_results": results,
        "elapsed_seconds": round(elapsed, 1),
    }
    out_dir = DOCS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hybrid_tuning.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Hybrid tuning (k=10) ===")
    print(f"Best weights (cf, content, popularity): {best[0]}  val NDCG@10={best[1]:.4f}")
    print(f"\n{'model':<16} {'P@10':>8} {'R@10':>8} {'HR@10':>8} {'NDCG@10':>9} {'MAP@10':>8}")
    for name, m in results.items():
        print(f"{name:<16} {m[f'precision@{K}']:8.4f} {m[f'recall@{K}']:8.4f} "
              f"{m[f'hit_rate@{K}']:8.4f} {m[f'ndcg@{K}']:9.4f} {m[f'map@{K}']:8.4f}")
    print(f"\nSaved -> {out_path} (total {elapsed:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
