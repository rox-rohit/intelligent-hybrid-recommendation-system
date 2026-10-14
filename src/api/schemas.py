"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ── User schemas ──────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    email: Optional[str] = Field(None, max_length=255)


class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ── Interaction schemas ───────────────────────────────────────────────────

class InteractionCreate(BaseModel):
    movie_id: int
    interaction_type: str = Field(
        ...,
        pattern="^(complete|watch|view|click|like|dislike|skip|watchlist|remove_watchlist|search|rate)$"
    )
    rating: Optional[float] = Field(None, ge=0.5, le=5.0)


class InteractionResponse(BaseModel):
    id: int
    user_id: int
    movie_id: int
    interaction_type: str
    rating: Optional[float]
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class OnboardingRating(BaseModel):
    movie_id: int
    rating: float = Field(..., ge=0.5, le=5.0)


class OnboardingRequest(BaseModel):
    genres: list[str] = Field(default_factory=list, max_length=20)
    ratings: list[OnboardingRating] = Field(default_factory=list, max_length=20)


# ── Movie schemas ─────────────────────────────────────────────────────────

class MovieBasic(BaseModel):
    """Minimal movie info for recommendations."""
    movie_id: int
    title: str
    genres: list[str]
    release_year: Optional[int]
    poster_path: Optional[str] = None


class MovieDetail(MovieBasic):
    """Full movie metadata."""
    avg_rating: Optional[float]
    n_ratings: int
    popularity_score: Optional[float]


# ── Recommendation schemas ────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    movie: MovieBasic
    score: float
    rank: int
    explanation: Optional[str]
    evidence: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendationItem]
    generated_at: datetime
    method: str  # "hybrid", "content", "collaborative", "popularity", "demo"


# ── Profile schemas ───────────────────────────────────────────────────────

class UserProfileResponse(BaseModel):
    user_id: int
    genre_affinity: Optional[dict[str, float]]
    avg_rating: Optional[float]
    n_interactions: int
    last_updated: datetime

    model_config = ConfigDict(from_attributes=True)


# ── Search/Filter schemas ─────────────────────────────────────────────────

class MovieSearchParams(BaseModel):
    query: Optional[str] = Field(None, max_length=200)
    genres: Optional[list[str]] = None
    min_year: Optional[int] = Field(None, ge=1900, le=2030)
    max_year: Optional[int] = Field(None, ge=1900, le=2030)
    min_rating: Optional[float] = Field(None, ge=0, le=5)
    limit: int = Field(20, ge=1, le=100)


# ── Health check schema ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict[str, bool]
    db_connected: bool
