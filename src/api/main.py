"""FastAPI application factory and routes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.api.database import engine, get_db_dependency, init_db
from src.api.models import Interaction, User
from src.api.schemas import (
    HealthResponse,
    InteractionCreate,
    InteractionResponse,
    MovieBasic,
    MovieSearchParams,
    RecommendationResponse,
    UserCreate,
    UserResponse,
    UserProfileResponse,
    OnboardingRequest,
)
from src.api.services import (
    get_recommendations,
    postprocess_recommendations,
    get_user_interactions,
    get_user_profile,
    record_interaction,
    search_movies,
)
from src.utils.config import get_settings
from src.utils.logging_config import configure_logging, get_logger
from src.utils.paths import DOCS_DIR, MODELS_DIR, PROCESSED_DATA_DIR

configure_logging("INFO")
log = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB on startup."""
    log.info("Starting PersonalAI API...")
    init_db()
    log.info("Database initialized")
    yield
    log.info("Shutting down...")


app = FastAPI(
    title="PersonalAI Recommendation API",
    description="Intelligent hybrid movie recommendation system",
    version="0.1.0",
    lifespan=lifespan,
    root_path="/api",  # All routes will be prefixed with /api
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health & Info ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint with model status."""
    models_loaded = {
        "als": (MODELS_DIR / "als.joblib").exists(),
        "itemknn": (MODELS_DIR / "itemknn.joblib").exists(),
        "ranker": (MODELS_DIR / "ranker.joblib").exists(),
        "hybrid_tuning": (DOCS_DIR / "results" / "hybrid_tuning.json").exists(),
        "processed_data": (PROCESSED_DATA_DIR / "ratings.parquet").exists(),
    }

    # Check DB connection
    db_ok = False
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        version="0.1.0",
        models_loaded=models_loaded,
        db_connected=db_ok,
    )


# ── User Management ───────────────────────────────────────────────────────

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db_dependency)):
    """Create a new user."""
    # Check if username exists
    existing = db.query(User).filter(User.username == user.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{user.username}' already exists",
        )

    new_user = User(username=user.username, email=user.email)
    db.add(new_user)
    db.flush()  # Get the ID

    # Create empty profile
    from src.api.models import UserProfile
    profile = UserProfile(user_id=new_user.id)
    db.add(profile)
    db.commit()  # Commit the transaction

    log.info("Created user: %s (id=%d)", new_user.username, new_user.id)
    return new_user


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db_dependency)):
    """Get user by ID."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )
    return user


@app.post("/users/login", response_model=UserResponse)
def login_user(username: str = Query(..., min_length=3), db: Session = Depends(get_db_dependency)):
    """Login existing user by username."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' not found. Please register first.",
        )
    return user


@app.get("/users/{user_id}/profile", response_model=UserProfileResponse)
def get_profile(user_id: int, db: Session = Depends(get_db_dependency)):
    """Get user's dynamic profile (genre affinities, activity, etc.)."""
    profile = get_user_profile(user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile for user {user_id} not found",
        )
    return profile


# ── Interactions ──────────────────────────────────────────────────────────

@app.post("/users/{user_id}/interactions", response_model=InteractionResponse, status_code=status.HTTP_201_CREATED)
def add_interaction(
    user_id: int,
    interaction: InteractionCreate,
    db: Session = Depends(get_db_dependency),
):
    """Record a user interaction (rate, like, watch, skip, watchlist, etc.)."""
    # Verify user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    # Create interaction
    new_interaction = Interaction(
        user_id=user_id,
        movie_id=interaction.movie_id,
        interaction_type=interaction.interaction_type,
        rating=interaction.rating,
    )
    db.add(new_interaction)
    db.flush()

    # Update profile
    from src.api.services import _update_user_profile
    _update_user_profile(db, user_id, interaction.movie_id, interaction.interaction_type, interaction.rating)
    db.commit()
    db.refresh(new_interaction)

    log.info(
        "User %d %s movie %d (rating=%s)",
        user_id, interaction.interaction_type, interaction.movie_id, interaction.rating,
    )
    return new_interaction


@app.get("/users/{user_id}/interactions", response_model=list[InteractionResponse])
def get_interactions(
    user_id: int,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db_dependency),
):
    """Get user's interaction history."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )

    interactions = (
        db.query(Interaction)
        .filter(Interaction.user_id == user_id)
        .order_by(Interaction.timestamp.desc())
        .limit(limit)
        .all()
    )
    return interactions


@app.get("/users/{user_id}/interactions/detail")
def get_interactions_detail(
    user_id: int,
    limit: int = Query(500, ge=1, le=1000),
):
    """Get user's full interaction history with details for frontend state."""
    interactions = get_user_interactions(user_id, limit)
    movies, _ = _load_processed_data()
    movie_rows = movies.set_index("movieId") if movies is not None else None
    states = {}
    for item in reversed(interactions):
        mid = int(item["movie_id"])
        state = states.setdefault(mid, {"liked": False, "watchlisted": False, "rating": None, "dismissed": False})
        kind = item["interaction_type"]
        if kind == "like":
            state["liked"] = True
        elif kind == "dislike":
            state["liked"] = False
            state["dismissed"] = True
        elif kind == "watchlist":
            state["watchlisted"] = True
        elif kind == "remove_watchlist":
            state["watchlisted"] = False
        if item.get("rating") is not None:
            state["rating"] = item["rating"]
    result = []
    for mid, state in states.items():
        if movie_rows is None or mid not in movie_rows.index:
            continue
        row = movie_rows.loc[mid]
        raw_genres = row.get("genres_list", [])
        try:
            genres = list(raw_genres) if raw_genres is not None else []
        except TypeError:
            genres = []
        result.append({
            "movie_id": mid,
            "title": row["title_clean"],
            "genres": genres,
            "release_year": int(row["release_year"]) if pd.notna(row.get("release_year")) else None,
            "poster_path": None,
            **state,
        })
    return {"interactions": result}


# ── Recommendations ───────────────────────────────────────────────────────

@app.get("/users/{user_id}/recommendations", response_model=RecommendationResponse)
def get_user_recommendations(
    user_id: int,
    k: int = Query(10, ge=1, le=50),
    method: str = Query("auto", pattern="^(auto|smart|hybrid|ranker|popularity|demo)$"),
    mode: str = Query("personalized", pattern="^(personalized|explore)$"),
    diversity: float = Query(1.0, ge=0.0, le=1.0),
):
    """Get personalized recommendations for a user.

    - **auto**: Use best available model (ranker > hybrid > popularity > demo)
    - **smart**: Personalized based on your actual interactions (genres, likes, ratings)
    - **hybrid**: Weighted blend of CF, content, popularity
    - **ranker**: Learned re-ranker (requires trained model)
    - **popularity**: Bayesian-smoothed popularity baseline
    - **demo**: Hardcoded popular movies (works without any training)
    """
    try:
        pool_size = min(50, max(k, 40))
        recs = get_recommendations(user_id, k=pool_size, method=method)
        recs = postprocess_recommendations(recs, k=k, mode=mode, diversity=diversity)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return RecommendationResponse(
        user_id=user_id,
        recommendations=recs,
        generated_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        method=method,
    )


# ── Movie Search & Discovery ──────────────────────────────────────────────

@app.get("/movies/search", response_model=list[MovieBasic])
def search_movies_endpoint(
    q: Optional[str] = Query(None, alias="query", max_length=200),
    genres: Optional[str] = Query(None, description="Comma-separated genres"),
    min_year: Optional[int] = Query(None, ge=1900, le=2030),
    max_year: Optional[int] = Query(None, ge=1900, le=2030),
    min_rating: Optional[float] = Query(None, ge=0, le=5),
    limit: int = Query(20, ge=1, le=100),
):
    """Search movies by title, genre, year, or rating."""
    genre_list = [g.strip() for g in genres.split(",")] if genres else None
    return search_movies(
        query=q,
        genres=genre_list,
        min_year=min_year,
        max_year=max_year,
        min_rating=min_rating,
        limit=limit,
    )


@app.get("/movies/{movie_id}", response_model=MovieBasic)
def get_movie(movie_id: int):
    """Get movie details by ID."""
    movies, _ = _load_processed_data()
    if movies is None:
        raise HTTPException(status_code=503, detail="Movie data not available")

    movie = movies[movies["movieId"] == movie_id]
    if movie.empty:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")

    row = movie.iloc[0]
    return MovieBasic(
        movie_id=int(row["movieId"]),
        title=row["title_clean"],
        genres=row.get("genres_list", []),
        release_year=int(row["release_year"]) if pd.notna(row.get("release_year")) else None,
        poster_path=row.get("poster_path"),
    )


def _load_processed_data():
    """Load processed parquet files."""
    import pandas as pd
    from src.utils.paths import PROCESSED_DATA_DIR

    try:
        movies = pd.read_parquet(PROCESSED_DATA_DIR / "movies.parquet")
        ratings = pd.read_parquet(PROCESSED_DATA_DIR / "ratings.parquet")
        return movies, ratings
    except Exception:
        return None, None


# ── Onboarding / Cold Start ───────────────────────────────────────────────

@app.get("/genres")
def get_all_genres():
    """Get all available genres for onboarding."""
    movies, _ = _load_processed_data()
    if movies is None:
        return {"genres": []}

    all_genres = set()
    for gs in movies["genres_list"]:
        try:
            if gs is not None:
                all_genres.update(gs)
        except TypeError:
            continue

    return {"genres": sorted(all_genres)}


@app.post("/users/{user_id}/onboarding", response_model=UserProfileResponse)
def save_onboarding(
    user_id: int,
    onboarding: OnboardingRequest,
    db: Session = Depends(get_db_dependency),
):
    """Save initial genre preferences and optional movie ratings."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    movies, _ = _load_processed_data()
    if movies is None:
        raise HTTPException(status_code=503, detail="Movie data not available")
    movie_genres = {}
    for mid, genres in zip(movies["movieId"], movies["genres_list"]):
        try:
            movie_genres[int(mid)] = set(genres) if genres is not None else set()
        except TypeError:
            movie_genres[int(mid)] = set()
    available = set().union(*movie_genres.values()) if movie_genres else set()
    invalid = sorted(set(onboarding.genres) - available)
    if invalid:
        raise HTTPException(status_code=422, detail=f"Unknown genres: {', '.join(invalid)}")
    invalid_movies = [r.movie_id for r in onboarding.ratings if r.movie_id not in movie_genres]
    if invalid_movies:
        raise HTTPException(status_code=422, detail=f"Unknown movie IDs: {invalid_movies[:5]}")

    from src.api.models import UserProfile
    from src.features.interactions import rating_to_interaction_type
    from src.recommendations.cold_start import create_onboarding_profile
    from datetime import datetime, timezone

    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    if not profile:
        profile = UserProfile(user_id=user_id, genre_affinity={}, n_interactions=0)
        db.add(profile)
    rated = {r.movie_id: r.rating for r in onboarding.ratings}
    profile.genre_affinity = create_onboarding_profile(onboarding.genres, rated, movie_genres)
    for rating in onboarding.ratings:
        db.add(Interaction(
            user_id=user_id,
            movie_id=rating.movie_id,
            interaction_type=rating_to_interaction_type(rating.rating),
            rating=rating.rating,
        ))
    profile.n_interactions = (profile.n_interactions or 0) + len(onboarding.ratings)
    profile.avg_rating = (sum(rated.values()) / len(rated)) if rated else profile.avg_rating
    profile.last_updated = datetime.now(timezone.utc)
    db.commit()
    db.refresh(profile)
    return profile


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
