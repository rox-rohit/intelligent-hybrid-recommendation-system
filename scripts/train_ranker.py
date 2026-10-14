"""Train the pointwise re-ranker on hybrid candidates (Milestone 5).

Pipeline:
  1. Time-aware split; train retrieval models (ALS, ItemKNN, content,
     popularity) on TRAIN — same as train_hybrid.py.
  2. For sampled VALIDATION users, generate top-200 hybrid candidates,
     extract ranking features, label positives (val items rated >= 3.5),
     and train the HistGBM ranker.
  3. For sampled TEST users, generate candidates, extract features,
     rerank, and score against held-out test items.
  4. Comparison table: hybrid-only vs. hybrid+ranker vs. ALS-only.

Writes:
  models/ranker.joblib           — the fitted ranker
  docs/results/ranker_comparison.json — side-by-side metrics
"""

from __future__ import annotations

import json
import sys
import time
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
from src.models.hybrid import HybridRecommender, HybridWeights, minmax
from src.models.popularity import PopularityModel
from src.ranking.features import (
    FEATURE_COLUMNS,
    UserItemStats,
    build_features,
    label_candidates,
)
from src.ranking.ranker import PointwiseRanker
from src.utils.logging_config import configure_logging, get_logger
from src.utils.paths import DOCS_DIR, MODELS_DIR

log = get_logger(__name__)

K = 10
CANDIDATE_POOL = 200      # per-component candidate retrieval depth
EVAL_USERS = 3_000        # sampled users for train/eval of ranker
RELEVANCE_THRESHOLD = 3.5
RANDOM_STATE = 42

# Default hybrid weights (can be overridden by loading tuning results)
DEFAULT_WEIGHTS = HybridWeights(cf=0.5, content=0.3, popularity=0.2)


def _pack(recs_by_user: dict, user_order: list[int], k: int) -> np.ndarray:
    """Pack per-user [(item, score)] lists into a (n_users, k) array."""
    out = np.full((len(user_order), k), -1, dtype=np.int64)
    for i, uid in enumerate(user_order):
        for j, (item, _) in enumerate(recs_by_user.get(uid, [])[:k]):
            out[i, j] = item
    return out


def _precompute_scores(
    users: list[int],
    user_index: pd.Series,
    item_index: pd.Series,
    conf,
    als: ImplicitALSModel,
    knn: ItemKNN,
    content_matrix,
    content_row_by_id: dict,
    train: pd.DataFrame,
    seen_by_user: dict[int, set[int]],
    pop_scores_norm,
    pool_size: int = CANDIDATE_POOL,
) -> tuple[dict, dict, dict, dict, dict]:
    """Precompute per-user component scores + candidate sets.

    Returns (als_scores, knn_scores, ct_scores, pop_scores, candidate_items)
    all keyed by userId.
    """
    item_by_col = {int(c): int(mid) for mid, c in item_index.items()}

    als_scores: dict[int, dict[int, float]] = {}
    knn_scores: dict[int, dict[int, float]] = {}
    ct_scores: dict[int, dict[int, float]] = {}
    pop_scores: dict[int, dict[int, float]] = {}
    candidate_items: dict[int, list[int]] = {}

    for uid in users:
        if uid not in user_index.index:
            continue
        urow = int(user_index.loc[uid])
        seen = seen_by_user.get(uid, set())

        # ALS: top candidates (internal ids -> external)
        ids, scores = als.model.recommend(
            urow, conf[urow], N=pool_size,
            filter_already_liked_items=True,
        )
        als_scores[uid] = {
            item_by_col.get(int(i), int(i)): float(s)
            for i, s in zip(ids, scores)
            if item_by_col.get(int(i), int(i)) not in seen
        }

        # ItemKNN: top candidates
        knn_recs = knn.recommend_for_user(
            urow, k=pool_size, seen_items=seen, item_index=item_index,
        )
        knn_scores[uid] = dict(knn_recs)

        # Content: centroid of liked train items vs candidate pool
        user_train = train[train["userId"] == uid]
        liked = [
            content_row_by_id[int(m)]
            for m, r in zip(user_train["movieId"], user_train["rating"])
            if r >= RELEVANCE_THRESHOLD and int(m) in content_row_by_id
        ]
        if liked:
            prof = np.asarray(content_matrix[liked].sum(axis=0)).ravel()
            norm = np.linalg.norm(prof)
            if norm > 0:
                prof = prof / norm
                pool = set(als_scores[uid]) | set(knn_scores[uid])
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

        # Popularity over the union of all component candidates
        pool_all = (
            set(als_scores.get(uid, {}))
            | set(knn_scores.get(uid, {}))
            | set(ct_scores.get(uid, {}))
        )
        pop_scores[uid] = {
            m: float(pop_scores_norm.get(m, 0.0)) for m in pool_all
        }

        candidate_items[uid] = sorted(pool_all)

    return als_scores, knn_scores, ct_scores, pop_scores, candidate_items


def main() -> int:
    configure_logging("INFO")
    t_start = time.perf_counter()

    # ── 1. Load, split, train retrieval models ────────────────────────────
    log.info("Loading data ...")
    ratings = load_processed("ratings")
    movies = load_processed("movies")
    genome = load_processed("genome_scores")

    log.info("Time-aware split ...")
    split = time_aware_split(ratings)
    log.info(
        "Split sizes: train=%s val=%s test=%s",
        len(split.train), len(split.validation), len(split.test),
    )
    del ratings

    # Confidence matrix from TRAIN
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
        train_inter, len(user_index), len(item_index), user_index, item_index,
    )

    log.info("Training ItemKNN ...")
    knn = ItemKNN(conf)

    log.info("Training ALS ...")
    als = ImplicitALSModel(factors=64, iterations=15, random_state=RANDOM_STATE)
    als.fit(conf)

    # Popularity model
    movie_stats = split.train.groupby("movieId").agg(
        mean_rating=("rating", "mean"), n_ratings=("rating", "size"),
    )
    movies_idx = movies.set_index("movieId")
    movie_stats["release_year"] = movies_idx["release_year"]
    pop = PopularityModel(movie_stats)
    pop_scores_norm = pop.popularity
    freshness = pop.freshness

    # Content matrix
    content_matrix, content_index, _ = build_movie_feature_matrix(genome, movies)
    content_row_by_id = dict(zip(content_index["movieId"], content_index["row"]))

    # Pre-aggregate stats for ranking features
    stats = UserItemStats.from_train(
        split.train, movies, dict(pop_scores_norm), dict(freshness),
    )

    # ── 2. Validation: generate candidates, build features, train ranker ──
    rng = np.random.default_rng(RANDOM_STATE)

    val_users_all = sorted(split.validation["userId"].unique())
    n_sample = min(EVAL_USERS, len(val_users_all))
    val_users = sorted(rng.choice(val_users_all, n_sample, replace=False).tolist())

    val_rel = split.validation[split.validation["rating"] >= RELEVANCE_THRESHOLD]
    truth_val: dict[int, set[int]] = {}
    for uid, grp in val_rel.groupby("userId"):
        truth_val[int(uid)] = set(grp["movieId"])
    val_users = [u for u in val_users if truth_val.get(u)]

    wanted_val = set(val_users)
    seen_val = {
        int(uid): set(grp["movieId"])
        for uid, grp in split.train.groupby("userId")
        if int(uid) in wanted_val
    }

    log.info("Precomputing component scores for %d validation users ...", len(val_users))
    als_v, knn_v, ct_v, pop_v, cands_v = _precompute_scores(
        val_users, user_index, item_index, conf, als, knn,
        content_matrix, content_row_by_id, split.train, seen_val, pop_scores_norm,
    )

    log.info("Building ranking features (validation) ...")
    feat_val = build_features(val_users, cands_v, als_v, knn_v, ct_v, stats)
    y_val = label_candidates(feat_val, truth_val)
    pos_rate = y_val.mean()
    log.info(
        "Validation features: %d rows, %d positives (%.1f%%)",
        len(feat_val), y_val.sum(), pos_rate * 100,
    )

    log.info("Training PointwiseRanker ...")
    ranker = PointwiseRanker(max_iter=200, max_depth=5, random_state=RANDOM_STATE)
    ranker.fit(feat_val, y_val)

    imps = ranker.feature_importances()
    log.info("Feature importances: %s", {k: round(v, 3) for k, v in
             sorted(imps.items(), key=lambda x: -x[1])})

    # ── 3. Test: generate candidates, rerank, evaluate ────────────────────
    log.info("Evaluating on TEST ...")

    test_users_all = sorted(split.test["userId"].unique())
    n_test = min(EVAL_USERS, len(test_users_all))
    test_users = sorted(rng.choice(test_users_all, n_test, replace=False).tolist())

    test_rel = split.test[split.test["rating"] >= RELEVANCE_THRESHOLD]
    truth_test: dict[int, set[int]] = {}
    for uid, grp in test_rel.groupby("userId"):
        truth_test[int(uid)] = set(grp["movieId"])
    test_users = [u for u in test_users if truth_test.get(u)]

    wanted_test = set(test_users)
    seen_test = {
        int(uid): set(grp["movieId"])
        for uid, grp in pd.concat([split.train, split.validation]).groupby("userId")
        if int(uid) in wanted_test
    }

    log.info("Precomputing component scores for %d test users ...", len(test_users))
    als_t, knn_t, ct_t, pop_t, cands_t = _precompute_scores(
        test_users, user_index, item_index, conf, als, knn,
        content_matrix, content_row_by_id, split.train, seen_test, pop_scores_norm,
    )

    log.info("Building ranking features (test) ...")
    feat_test = build_features(test_users, cands_t, als_t, knn_t, ct_t, stats)

    # Hybrid+ranker: rerank candidates
    log.info("Reranking with PointwiseRanker ...")
    reranked_by_user: dict[int, list[tuple[int, float]]] = {}
    for uid in test_users:
        user_feats = feat_test[feat_test["userId"] == uid].reset_index(drop=True)
        if len(user_feats) == 0:
            reranked_by_user[uid] = []
            continue
        reranked_by_user[uid] = ranker.rerank(user_feats, k=K)

    # Hybrid-only: weighted blend without reranking
    # Try to load tuned weights from prior experiment
    tuning_path = DOCS_DIR / "results" / "hybrid_tuning.json"
    if tuning_path.exists():
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        bw = tuning["best_weights"]
        weights = HybridWeights(cf=bw["cf"], content=bw["content"], popularity=bw["popularity"])
        log.info("Loaded tuned weights: %s", weights.as_tuple())
    else:
        weights = DEFAULT_WEIGHTS
        log.info("No tuning results found; using default weights: %s", weights.as_tuple())

    hybrid_by_user: dict[int, list[tuple[int, float]]] = {}
    for uid in test_users:
        hy = HybridRecommender(
            cf_scorer=lambda u, _cf=als_t: _cf.get(u, {}),
            content_scorer=lambda u, _ct=ct_t: _ct.get(u, {}),
            popularity_scorer=lambda u, _p=pop_t: _p.get(u, {}),
            weights=weights,
        )
        hybrid_by_user[uid] = hy.recommend(uid, k=K)

    # ALS-only baseline
    als_by_user: dict[int, list[tuple[int, float]]] = {}
    for uid in test_users:
        items = als_t.get(uid, {})
        als_by_user[uid] = sorted(items.items(), key=lambda x: -x[1])[:K] if items else []

    results: dict[str, dict] = {}
    results["hybrid+ranker"] = evaluate_all(
        _pack(reranked_by_user, test_users, K),
        {u: truth_test[u] for u in test_users}, k=K,
    )
    results["hybrid_only"] = evaluate_all(
        _pack(hybrid_by_user, test_users, K),
        {u: truth_test[u] for u in test_users}, k=K,
    )
    results["als_only"] = evaluate_all(
        _pack(als_by_user, test_users, K),
        {u: truth_test[u] for u in test_users}, k=K,
    )

    # ── 4. Save model + results ───────────────────────────────────────────
    ranker_path = MODELS_DIR / "ranker.joblib"
    ranker.save(ranker_path)

    elapsed = time.perf_counter() - t_start
    payload = {
        "experiment": "ranking_stage",
        "k": K,
        "n_val_users": len(val_users),
        "n_test_users": len(test_users),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "candidate_pool": CANDIDATE_POOL,
        "val_positive_rate": round(float(pos_rate), 4),
        "feature_importances": imps,
        "hybrid_weights": weights.as_tuple(),
        "test_results": results,
        "elapsed_seconds": round(elapsed, 1),
    }
    out_dir = DOCS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ranker_comparison.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== Ranking stage comparison (test, k={K}, n={len(test_users)} users) ===")
    print(f"{'model':<18} {'P@10':>8} {'R@10':>8} {'HR@10':>8} {'NDCG@10':>9} {'MAP@10':>8}")
    for name, m in results.items():
        print(
            f"{name:<18} {m[f'precision@{K}']:8.4f} {m[f'recall@{K}']:8.4f} "
            f"{m[f'hit_rate@{K}']:8.4f} {m[f'ndcg@{K}']:9.4f} {m[f'map@{K}']:8.4f}"
        )
    print(f"\nFeature importances:")
    for feat, imp in sorted(imps.items(), key=lambda x: -x[1]):
        print(f"  {feat:<20} {imp:.3f}")
    print(f"\nRanker saved → {ranker_path}")
    print(f"Results saved → {out_path} (total {elapsed:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
