"""Database models for user profiles, interactions, and recommendations.

Schema design (normalized, time-aware):
  - users: basic profile (id, created_at)
  - interactions: event log (userId, movieId, type, timestamp, rating)
  - user_profiles: aggregated preferences (genre affinities, activity level)
  - recommendations: cached top-k per user (expires after 24h)

All timestamps are UTC. Ratings are explicit (1-5 stars) or implicit
(derived from interaction types: like=5, watch=4, view=3, skip=1).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    interactions: Mapped[list["Interaction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    profile: Mapped[Optional["UserProfile"]] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    movie_id: Mapped[int] = mapped_column(Integer, nullable=False)
    interaction_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # complete, watch, view, click, like, dislike, skip, watchlist, search
    rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # explicit rating (1-5 stars) if given; null for implicit interactions
    timestamp: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="interactions")

    __table_args__ = (
        Index("ix_interactions_user_time", "user_id", "timestamp"),
        Index("ix_interactions_movie", "movie_id"),
    )


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), primary_key=True
    )
    genre_affinity: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # {genre: score}
    avg_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    n_interactions: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="profile")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    movie_id: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    # rank in the top-k (1 = best)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # "users like you watched this" / "matches your Action preference"
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="recommendations")

    __table_args__ = (
        Index("ix_recommendations_user_rank", "user_id", "rank"),
    )
