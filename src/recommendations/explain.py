"""Explanation builder for recommendations (Milestone 8).

Generates human-readable "why this?" explanations citing real
model components and evidence — never templated fiction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.models.hybrid import HybridRecommender, HybridWeights
from src.models.popularity import PopularityModel
from src.recommendations.profile_updater import ProfileUpdater
from src.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class Evidence:
    """Single piece of evidence for an explanation."""
    source: str           # "collaborative", "content", "popularity", "profile", "ranker", "diversity"
    detail: str           # Human-readable detail
    weight: float         # Contribution to final score (0-1)


@dataclass
class Explanation:
    """Full explanation for one recommendation."""
    primary_reason: str
    evidence: list[Evidence]
    confidence: float     # How strong is this explanation (0-1)
    dominant_source: str  # The source with highest weight


class ExplanationBuilder:
    """Builds explanations from model component scores."""

    def __init__(
        self,
        hybrid: HybridRecommender | None = None,
        ranker_feature_importances: dict[str, float] | None = None,
        content_model = None,
        item_knn = None,
        pop_model: PopularityModel | None = None,
        movie_titles: dict[int, str] | None = None,
    ):
        self.hybrid = hybrid
        self.ranker_feature_importances = ranker_feature_importances or {}
        self.content_model = content_model
        self.item_knn = item_knn
        self.pop_model = pop_model
        self.movie_titles = movie_titles or {}

        # Source -> human reason mapping
        self._source_reasons = {
            "collaborative": "Users with similar taste watched this",
            "content": "Matches your genre and theme preferences",
            "popularity": "Popular and highly rated overall",
            "profile": "Matches your viewing profile",
            "ranker": "Our ranking model prioritized this for you",
            "diversity": "Added for variety in your recommendations",
        }

    def explain(
        self,
        user_id: int,
        movie_id: int,
        final_score: float,
        component_scores: dict[str, float],      # cf, content, pop, ranker
        user_profile: dict | None = None,
        mmr_applied: bool = False,
        diversity_info: dict | None = None,
    ) -> Explanation:
        """Build explanation from component scores."""

        evidence = []

        # 1. Hybrid component contributions
        if self.hybrid and hasattr(self.hybrid, 'explain'):
            hybrid_exp = self.hybrid.explain(user_id, movie_id)
            contrib = hybrid_exp.get("contributions", {})

            for source, weight in contrib.items():
                if weight > 0.01:  # Only meaningful contributions
                    if source == "collaborative":
                        detail = self._collaborative_detail(user_id, movie_id, weight)
                    elif source == "content":
                        detail = self._content_detail(user_id, movie_id, weight)
                    elif source == "popularity":
                        detail = self._popularity_detail(movie_id, weight)
                    else:
                        detail = f"{source} contribution: {weight:.2f}"

                    evidence.append(Evidence(
                        source=source,
                        detail=detail,
                        weight=weight,
                    ))

        # 2. Ranker feature importance (if available)
        if self.ranker_feature_importances:
            ranker_evidence = self._ranker_evidence(movie_id, component_scores)
            evidence.extend(ranker_evidence)

        # 3. Profile evidence
        if user_profile and user_profile.get("genre_affinity"):
            profile_evidence = self._profile_evidence(user_profile, movie_id)
            evidence.extend(profile_evidence)

        # 4. Diversity evidence
        if mmr_applied and diversity_info:
            evidence.append(Evidence(
                source="diversity",
                detail="Selected for variety — different from other recommendations",
                weight=diversity_info.get("weight", 0.1),
            ))

        # Sort by weight
        evidence.sort(key=lambda e: -e.weight)

        # Determine dominant source
        dominant = evidence[0].source if evidence else "unknown"
        primary_reason = self._source_reasons.get(dominant, "Recommended for you")

        # Confidence = sum of top evidence weights (capped at 1.0)
        confidence = min(sum(e.weight for e in evidence[:3]), 1.0)

        return Explanation(
            primary_reason=primary_reason,
            evidence=evidence[:4],  # Top 4 evidence pieces
            confidence=confidence,
            dominant_source=dominant,
        )

    def _collaborative_detail(self, user_id: int, movie_id: int, weight: float) -> str:
        """Detail for collaborative filtering evidence."""
        if self.item_knn:
            try:
                similar = self.item_knn.similar_items(movie_id, k=3)
                if similar:
                    titles = [self.movie_titles.get(mid, f"Movie {mid}") for mid, _ in similar]
                    return f"Similar to {', '.join(titles[:2])} which you liked"
            except Exception:
                pass
        return f"Collaborative signal: {weight:.0%} of hybrid score"

    def _content_detail(self, user_id: int, movie_id: int, weight: float) -> str:
        """Detail for content-based evidence."""
        if self.content_model:
            try:
                similar = self.content_model.recommend_similar_movies(movie_id, k=1)
                if similar:
                    shared = similar[0].shared_features[:3]
                    if shared:
                        return f"Shares {', '.join(shared)} with your preferences"
            except Exception:
                pass
        return f"Content match: {weight:.0%} of hybrid score"

    def _popularity_detail(self, movie_id: int, weight: float) -> str:
        """Detail for popularity evidence."""
        if self.pop_model:
            try:
                pop_score = self.pop_model.popularity.get(movie_id, 0)
                if pop_score > 0.7:
                    return f"Very popular (top {int((1-pop_score)*100)}%)"
                elif pop_score > 0.4:
                    return f"Popular movie"
            except Exception:
                pass
        return f"Popularity: {weight:.0%} of hybrid score"

    def _profile_evidence(self, user_profile: dict, movie_id: int) -> list[Evidence]:
        """Evidence from user's genre affinity profile."""
        evidence = []
        genre_affinity = user_profile.get("genre_affinity", {})
        movie_genres = user_profile.get("movie_genres", {}).get(movie_id, set())

        if genre_affinity and movie_genres:
            # Find top matching genres
            matches = [(g, genre_affinity[g]) for g in movie_genres if g in genre_affinity]
            if matches:
                matches.sort(key=lambda x: -x[1])
                top_genre, affinity = matches[0]
                evidence.append(Evidence(
                    source="profile",
                    detail=f"You watch {affinity:.0%} {top_genre} movies",
                    weight=affinity * 0.3,  # Scaled down
                ))

        return evidence

    def _ranker_evidence(
        self,
        movie_id: int,
        component_scores: dict[str, float]
    ) -> list[Evidence]:
        """Evidence from ranker feature importances."""
        evidence = []

        # Check which features were important for this prediction
        if "cf_als_score" in self.ranker_feature_importances:
            imp = self.ranker_feature_importances["cf_als_score"]
            if imp > 0.15:
                evidence.append(Evidence(
                    source="ranker",
                    detail="Ranker strongly weighed collaborative signal for you",
                    weight=imp * 0.2,
                ))

        if "genre_overlap" in self.ranker_feature_importances:
            imp = self.ranker_feature_importances["genre_overlap"]
            if imp > 0.15:
                evidence.append(Evidence(
                    source="ranker",
                    detail="Ranker prioritized genre match for your profile",
                    weight=imp * 0.2,
                ))

        return evidence


def format_explanation_for_ui(explanation: Explanation) -> dict:
    """Format explanation for frontend consumption."""
    return {
        "primary_reason": explanation.primary_reason,
        "confidence": round(explanation.confidence, 2),
        "evidence": [
            {
                "source": e.source,
                "detail": e.detail,
                "weight": round(e.weight, 2),
            }
            for e in explanation.evidence
        ],
        "dominant_source": explanation.dominant_source,
    }