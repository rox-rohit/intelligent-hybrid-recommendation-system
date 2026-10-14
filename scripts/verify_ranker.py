"""Quick verification script for Milestone 5 implementation.

Run this to verify the ranking stage is working correctly before running
the full training pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.ranking.features import (
    FEATURE_COLUMNS,
    UserItemStats,
    build_features,
    label_candidates,
)
from src.ranking.ranker import PointwiseRanker

print("=" * 60)
print("Milestone 5: Ranking Stage — Quick Verification")
print("=" * 60)

# 1. Test feature builder
print("\n[1/4] Testing feature builder ...")
tiny_stats = UserItemStats(
    item_n_ratings={1: 100, 2: 50},
    item_avg_rating={1: 4.2, 2: 3.5},
    user_n_ratings={0: 20},
    user_genre_profile={0: {"Action": 0.5, "Comedy": 0.5}},
    item_genres={1: {"Action"}, 2: {"Comedy"}},
    item_freshness={1: 0.8, 2: 0.5},
    item_pop_score={1: 0.7, 2: 0.4},
)

als_scores = {0: {1: 5.0, 2: 3.0}}
knn_scores = {0: {1: 0.9, 2: 0.7}}
content_scores = {0: {1: 0.8, 2: 0.6}}

df = build_features(
    [0], {0: [1, 2]},
    als_scores, knn_scores, content_scores, tiny_stats,
)

assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
assert set(df.columns) == {"userId", "movieId"} | set(FEATURE_COLUMNS), \
    f"Column mismatch: {set(df.columns)}"
print("✓ Feature builder works correctly")

# 2. Test labeling
print("\n[2/4] Testing label function ...")
labels = label_candidates(df, {0: {1}})
assert list(labels) == [1, 0], f"Expected [1, 0], got {list(labels)}"
print("✓ Labeling works correctly")

# 3. Test ranker fit/predict
print("\n[3/4] Testing PointwiseRanker ...")
rng = np.random.default_rng(42)
n = 200
X = pd.DataFrame(
    rng.standard_normal((n, len(FEATURE_COLUMNS))),
    columns=FEATURE_COLUMNS,
)
X["userId"] = np.repeat([0, 1], n // 2)
X["movieId"] = np.tile(np.arange(n // 2), 2)

# Make labels correlated with first feature
prob = 1 / (1 + np.exp(-2 * X["cf_als_score"]))
y = rng.binomial(1, prob).astype(np.int32)

ranker = PointwiseRanker(max_iter=50)
ranker.fit(X, y)
scores = ranker.predict_scores(X)

assert scores.shape == (len(X),), f"Score shape mismatch: {scores.shape}"
assert all(0 <= s <= 1 for s in scores), "Scores not in [0, 1]"
print("✓ Ranker fit/predict works correctly")

# 4. Test rerank
print("\n[4/4] Testing rerank ...")
user0 = X[X["userId"] == 0].copy().reset_index(drop=True)
recs = ranker.rerank(user0, k=5)

assert len(recs) == 5, f"Expected 5 recs, got {len(recs)}"
scores_only = [s for _, s in recs]
assert scores_only == sorted(scores_only, reverse=True), "Scores not descending"
print("✓ Rerank works correctly")

# 5. Feature importances
print("\n[5/5] Testing feature importances ...")
imps = ranker.feature_importances()
assert set(imps.keys()) == set(FEATURE_COLUMNS), "Feature importance keys mismatch"
assert all(v >= 0 for v in imps.values()), "Negative importance found"
top3 = sorted(imps.items(), key=lambda x: -x[1])[:3]
print(f"  Top 3 features: {[f for f, _ in top3]}")
print("✓ Feature importances work correctly")

print("\n" + "=" * 60)
print("✅ All checks passed! Milestone 5 implementation verified.")
print("=" * 60)
print("\nNext steps:")
print("  1. Run full unit tests: python -m pytest tests/test_ranker.py -v")
print("  2. Train on real data: python scripts/train_ranker.py")
print("     (Warning: 10-20 min runtime; requires processed data)")
