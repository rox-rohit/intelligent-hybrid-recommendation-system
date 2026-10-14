"""Learning-to-rank stage: candidate scoring and re-ranking models.

Public API:
  RankingFeatureBuilder / build_features / label_candidates — feature extraction
  PointwiseRanker — HistGBM-based pointwise re-ranker
  FEATURE_COLUMNS — canonical feature schema
"""

from src.ranking.features import (
    FEATURE_COLUMNS,
    UserItemStats,
    build_features,
    label_candidates,
)
from src.ranking.ranker import PointwiseRanker

__all__ = [
    "FEATURE_COLUMNS",
    "UserItemStats",
    "build_features",
    "label_candidates",
    "PointwiseRanker",
]
