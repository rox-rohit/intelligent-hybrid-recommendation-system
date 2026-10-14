"""Train CF models on real data with the time-aware split; first metrics.

Pipeline (spec sections 8, 12, 17):
  1. Load processed ratings; time-aware split (80/10/10 per user)
  2. Build implicit confidence matrix from TRAIN interactions only
  3. Train ItemKNN and ImplicitALS on it
  4. Evaluate on TEST: for each user, recommend k=10 unseen items,
     score against the user's held-out test items (relevance = test rating
     >= 3.5, i.e. what they actually watched and liked next)
  5. Also evaluate the popularity baseline for the comparison table

Writes results to docs/results/model_comparison.json (first rows).
Evaluation scale: to fit memory/time on this machine, we evaluate a
random sample of test users (default 5,000, seeded) — a standard,
documented practice; CI on NDCG@10 with n=5000 is tight enough to
rank models.
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
from src.features.interactions import build_confidence_matrix, derive_interactions
from src.models.collaborative import ImplicitALSModel, ItemKNN
from src.models.popularity import PopularityModel
from src.utils.logging_config import configure_logging, get_logger
from src.utils.paths import DOCS_DIR

log = get_logger(__name__)

K = 10
EVAL_USERS = 5_000     # sampled test users for offline metrics
RELEVANCE_THRESHOLD = 3.5  # held-out rating counts as relevant if >= this
RANDOM_STATE = 42


def recs_matrix_to_arrays(recs_by_user: dict[int, list[tuple[int, float]]],
                          user_order: list[int], k: int) -> np.ndarray:
    """Pack per-user [(item, score)] lists into a (n_users, k) id array."""
    out = np.full((len(user_order), k), -1, dtype=np.int64)
    for i, uid in enumerate(user_order):
        for j, (item, _) in enumerate(recs_by_user[uid][:k]):
            out[i, j] = item
    return out


def main() -> int:
    configure_logging("INFO")
    t_start = time.perf_counter()

    # ---- 1. Load + split ------------------------------------------------
    log.info("Loading ratings ...")
    ratings = load_processed("ratings")
    log.info("Time-aware split (this walks 25M rows once) ...")
    split = time_aware_split(ratings)
    log.info("Split sizes: train=%s val=%s test=%s",
             len(split.train), len(split.validation), len(split.test))

    # ---- 2. Confidence matrix from TRAIN only ---------------------------
    train_inter = derive_interactions(split.train)
    user_index = pd.Series(
        np.arange(train_inter["userId"].nunique()),
        index=np.sort(train_inter["userId"].unique()),
    )
    item_index = pd.Series(
        np.arange(train_inter["movieId"].nunique()),
        index=np.sort(train_inter["movieId"].unique()),
    )
    t0 = time.perf_counter()
    conf = build_confidence_matrix(
        train_inter, len(user_index), len(item_index), user_index, item_index
    )
    log.info("Confidence matrix: %s in %.1fs", conf.shape, time.perf_counter() - t0)

    # ---- 3. Evaluation sample + ground truth ----------------------------
    rng = np.random.default_rng(RANDOM_STATE)
    test_users = sorted(split.test["userId"].unique())
    if len(test_users) > EVAL_USERS:
        sample_users = sorted(rng.choice(test_users, EVAL_USERS, replace=False).tolist())
    else:
        sample_users = test_users
    log.info("Evaluating on %d sampled test users (of %d).", len(sample_users), len(test_users))

    truth: dict[int, set[int]] = {}
    test_rel = split.test[split.test["rating"] >= RELEVANCE_THRESHOLD]
    for uid, grp in test_rel.groupby("userId"):
        truth[int(uid)] = set(grp["movieId"])
    # keep only sampled users that have at least one relevant test item
    sample_users = [u for u in sample_users if truth.get(u)]
    log.info("Sampled users with relevant test items: %d", len(sample_users))

    # seen items per user, but ONLY for the sampled users we score —
    # building sets for all 162k users (20M memberships) is a needless
    # multi-GiB allocation that contributed to OOMs.
    wanted = set(sample_users)
    seen_by_user: dict[int, set[int]] = {
        int(uid): set(grp["movieId"])
        for uid, grp in split.train.groupby("userId")
        if int(uid) in wanted
    }

    # item id -> internal column id
    col_by_item = {int(mid): int(c) for mid, c in item_index.items()}
    item_by_col = {int(c): int(mid) for mid, c in item_index.items()}
    n_items_all = len(item_index)

    # ---- 4. Train models -------------------------------------------------
    results: dict[str, dict] = {}

    # 4a. Popularity baseline (Bayesian top-k for everyone)
    log.info("Training popularity baseline ...")
    movie_stats = split.train.groupby("movieId").agg(
        mean_rating=("rating", "mean"), n_ratings=("rating", "size")
    )
    movies_df = load_processed("movies").set_index("movieId")
    movie_stats["release_year"] = movies_df["release_year"]
    pop = PopularityModel(movie_stats)
    top_pop = pop.top_popular(k=K)
    pop_items = list(top_pop.index)
    pop_by_user = {u: [(m, 0.0) for m in pop_items] for u in sample_users}
    recs_pop = recs_matrix_to_arrays(pop_by_user, sample_users, K)
    # exclude already-seen items the way every model must
    for i, uid in enumerate(sample_users):
        seen = seen_by_user.get(uid, set())
        recs_pop[i] = np.array([m for m in recs_pop[i] if m != -1 and m not in seen]
                                + [-1] * (K - len([m for m in recs_pop[i] if m != -1 and m not in seen])),
                                dtype=np.int64)
    results["popularity"] = evaluate_all(recs_pop, {u: truth[u] for u in sample_users},
                                        k=K, n_items=n_items_all)

    # 4b. ItemKNN
    log.info("Training ItemKNN (full %s similarity matrix) ...", conf.shape[1])
    t0 = time.perf_counter()
    knn = ItemKNN(conf)
    log.info("ItemKNN trained in %.1fs", time.perf_counter() - t0)
    knn_by_user = {}
    for uid in sample_users:
        urow = int(user_index.loc[uid])
        knn_by_user[uid] = knn.recommend_for_user(
            urow, k=K, seen_items=seen_by_user[uid], item_index=item_index
        )
    recs_knn = recs_matrix_to_arrays(knn_by_user, sample_users, K)
    results["item_knn"] = evaluate_all(recs_knn, {u: truth[u] for u in sample_users},
                                       k=K, n_items=n_items_all)

    # 4c. ImplicitALS
    log.info("Training ImplicitALS (factors=64) ...")
    t0 = time.perf_counter()
    als = ImplicitALSModel(factors=64, iterations=15, random_state=RANDOM_STATE)
    als.fit(conf)
    log.info("ALS trained in %.1fs", time.perf_counter() - t0)
    als_by_user = {}
    conf_rows = conf  # recommend() needs the same user-item matrix
    for uid in sample_users:
        urow = int(user_index.loc[uid])
        recs = als.recommend_for_user(
            urow, k=K + 50, user_items=conf_rows[urow],
        )
        # filter seen items, keep top K
        seen = seen_by_user.get(uid, set())
        recs = [(i, s) for i, s in recs if item_by_col.get(i, i) not in seen][:K]
        als_by_user[uid] = recs
    recs_als = recs_matrix_to_arrays(als_by_user, sample_users, K)
    results["implicit_als"] = evaluate_all(recs_als, {u: truth[u] for u in sample_users},
                                           k=K, n_items=n_items_all)

    # ---- 5. Persist ------------------------------------------------------
    # Save trained models to disk
    models_dir = DOCS_DIR.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    knn.save(models_dir / "itemknn.joblib")
    als.save(models_dir / "als.joblib")

    # Also save the index mappings needed for inference
    import joblib
    index_data = {
        'user_index': user_index,
        'item_index': item_index,
        'col_by_item': col_by_item,
        'item_by_col': item_by_col,
    }
    joblib.dump(index_data, models_dir / "cf_indices.joblib")
    log.info("Saved CF indices → %s", models_dir / "cf_indices.joblib")

    elapsed = time.perf_counter() - t_start
    payload = {
        "experiment": "cf_baselines",
        "k": K,
        "n_eval_users": len(sample_users),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }
    out_dir = DOCS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_comparison.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Model comparison (test, k=10, n=%d users) ===" % len(sample_users))
    print(f"{'model':<14} {'P@10':>8} {'R@10':>8} {'HR@10':>8} {'NDCG@10':>9} {'MAP@10':>8}")
    for name, m in results.items():
        print(f"{name:<14} {m['precision@10']:8.4f} {m['recall@10']:8.4f} "
              f"{m['hit_rate@10']:8.4f} {m['ndcg@10']:9.4f} {m['map@10']:8.4f}")
    print(f"\nSaved -> {out_path} (total {elapsed:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
