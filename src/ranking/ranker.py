"""Pointwise ranking model for candidate re-ranking (Milestone 5).

Wraps sklearn's HistGradientBoostingClassifier in a recommender-friendly
API. The classifier predicts P(engagement | user, item features) and the
predicted probability is used as the re-ranking score.

Why HistGradientBoosting over LightGBM?
  - Zero extra dependencies (ships with sklearn ≥ 1.0).
  - Native missing-value support (NaN features pass through without
    imputation — useful when a component score is absent).
  - Comparable quality on tabular data of this scale.
  - The architecture doc names it as the guaranteed-available fallback.

The model is intentionally simple: 10 features, ~200 candidates per user,
binary labels. A pointwise formulation is the right starting point;
pairwise/listwise LTR adds complexity without proven lift at this stage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.ranking.features import FEATURE_COLUMNS
from src.utils.logging_config import get_logger

log = get_logger(__name__)

# Rows cached at fit time for permutation importance (stratified pos/neg).
_IMPORTANCE_SAMPLE_ROWS = 50_000


class PointwiseRanker:
    """Pointwise re-ranker over (user, item) feature vectors.

    fit() trains on labeled feature matrices; predict_scores() returns
    engagement probabilities; rerank() produces a top-k list for one user.
    """

    def __init__(
        self,
        max_iter: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 50,
        random_state: int = 42,
    ):
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )
        self._is_fitted = False
        self._importance_X: np.ndarray | None = None
        self._importance_y: np.ndarray | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "PointwiseRanker":
        """Train on feature matrix X and binary labels y.

        X should contain only FEATURE_COLUMNS (no userId/movieId).
        y: 1 = user engaged with item, 0 = candidate not engaged.
        """
        if isinstance(X, pd.DataFrame):
            X = X[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        self.model.fit(X, y)
        self._is_fitted = True
        log.info(
            "Ranker fitted: %d samples, %d features, %d iterations",
            X.shape[0], X.shape[1], self.model.n_iter_,
        )
        # Cache a stratified subsample for permutation-based importances —
        # HistGradientBoosting exposes no native feature_importances_, and
        # running permutation over the full 1M+ row matrix would be slow.
        rng = np.random.default_rng(self.model.random_state)
        n = X.shape[0]
        if n > _IMPORTANCE_SAMPLE_ROWS:
            idx_pos = np.flatnonzero(y == 1)
            idx_neg = np.flatnonzero(y == 0)
            n_pos = min(len(idx_pos), _IMPORTANCE_SAMPLE_ROWS // 2)
            n_neg = _IMPORTANCE_SAMPLE_ROWS - n_pos
            sample_idx = np.concatenate([
                rng.choice(idx_pos, n_pos, replace=False) if n_pos else idx_pos[:0],
                rng.choice(idx_neg, n_neg, replace=False) if n_neg else idx_neg[:0],
            ])
        else:
            sample_idx = np.arange(n)
        self._importance_X = X[sample_idx]
        self._importance_y = y[sample_idx]
        return self

    def predict_scores(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicted P(engagement) for each row — the re-ranking score."""
        if not self._is_fitted:
            raise RuntimeError("Ranker is not fitted; call fit() first.")
        if isinstance(X, pd.DataFrame):
            X = X[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        return self.model.predict_proba(X)[:, 1]

    def rerank(
        self,
        features_df: pd.DataFrame,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Re-rank candidates for ONE user and return top-k (item_id, score).

        features_df: rows for a single user, must contain movieId +
        FEATURE_COLUMNS. The output is sorted by predicted engagement
        probability, best first.
        """
        scores = self.predict_scores(features_df)
        items = features_df["movieId"].to_numpy()
        order = np.argsort(-scores)
        top = order[:k]
        return [(int(items[i]), float(scores[i])) for i in top]

    def feature_importances(self) -> dict[str, float]:
        """Feature name → permutation importance.

        HistGradientBoostingClassifier has no native feature_importances_
        attribute, so we compute a single-pass permutation importance on a
        cached subsample of training rows: for each feature, shuffle its
        column and measure the drop in log-loss. Useful for the
        explainability layer (Milestone 8): if cf_als_score dominates, the
        explanation says "users like you"; if genre_overlap dominates, it
        says "matches your genre preferences".
        """
        if not self._is_fitted:
            raise RuntimeError("Ranker is not fitted; call fit() first.")
        from sklearn.metrics import log_loss

        X, y = self._importance_X, self._importance_y
        if X is None or len(X) == 0:
            raise RuntimeError("No cached importance sample; refit the ranker.")

        base = log_loss(y, self.model.predict_proba(X)[:, 1], labels=[0, 1])
        imps: dict[str, float] = {}
        rng = np.random.default_rng(self.model.random_state)
        Xp = X.copy()
        for j, name in enumerate(FEATURE_COLUMNS):
            perm = rng.permutation(X.shape[0])
            Xp[:, j] = X[perm, j]
            perm_loss = log_loss(y, self.model.predict_proba(Xp)[:, 1], labels=[0, 1])
            imps[name] = float(max(perm_loss - base, 0.0))
            Xp[:, j] = X[:, j]  # restore column
        total = sum(imps.values())
        if total > 0:
            imps = {k: v / total for k, v in imps.items()}
        return imps

    def save(self, path: str | Path) -> Path:
        """Persist the fitted model to disk via joblib."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        log.info("Ranker saved: %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PointwiseRanker":
        """Load a previously saved ranker."""
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved ranker at {path}")
        ranker = cls()
        ranker.model = joblib.load(path)
        ranker._is_fitted = True
        log.info("Ranker loaded: %s", path)
        return ranker
