"""Recommendations module exports."""

from src.recommendations.profile_updater import ProfileUpdater, ProfileUpdateConfig
from src.recommendations.realtime_content import RealtimeContentScorer, create_realtime_scorer_from_data
from src.recommendations.mmr import mmr_rerank, mmr_rerank_with_genre
from src.recommendations.exploration import ExplorationPolicy, ExplorationConfig, compute_novelty_score
from src.recommendations.cold_start import ColdStartHandler, ColdStartConfig, create_onboarding_profile
from src.recommendations.explain import ExplanationBuilder, Explanation, Evidence, format_explanation_for_ui

__all__ = [
    "ProfileUpdater",
    "ProfileUpdateConfig",
    "RealtimeContentScorer",
    "create_realtime_scorer_from_data",
    "mmr_rerank",
    "mmr_rerank_with_genre",
    "ExplorationPolicy",
    "ExplorationConfig",
    "compute_novelty_score",
    "ColdStartHandler",
    "ColdStartConfig",
    "create_onboarding_profile",
    "ExplanationBuilder",
    "Explanation",
    "Evidence",
    "format_explanation_for_ui",
]