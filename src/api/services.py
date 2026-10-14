"""Core recommendation service — orchestrates the hybrid + ranking pipeline.

This service provides a clean API for the FastAPI routes to fetch recommendations.
It supports two modes:
  1. PRODUCTION: Loads trained models (ALS, ItemKNN, ranker) and computes real scores
  2. DEMO: Returns plausible mock recommendations without any trained models

The demo mode lets you test the full frontend→backend→recommendation flow
before investing compute in model training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.api.database import get_db
from src.api.models import Interaction, Recommendation, User, UserProfile
from src.api.schemas import MovieBasic, RecommendationItem
from src.utils.config import get_settings
from src.utils.logging_config import get_logger
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR

log = get_logger(__name__)
settings = get_settings()

# Global state for loaded models (lazy initialization)
_models_loaded = False
_models = {}
_ranker_stats = None
_processed_data_cache = None


def _optional_int(value) -> Optional[int]:
    """Convert nullable pandas/numpy year values to a JSON-safe integer."""
    if value is None or pd.isna(value):
        return None
    try:
        year = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return year or None


def _load_processed_data():
    """Load processed parquet files once for recommendation requests."""
    global _processed_data_cache
    if _processed_data_cache is not None:
        return _processed_data_cache
    try:
        movies = pd.read_parquet(PROCESSED_DATA_DIR / "movies.parquet")
        ratings = pd.read_parquet(PROCESSED_DATA_DIR / "ratings.parquet")
        _processed_data_cache = (movies, ratings)
        return _processed_data_cache
    except Exception as e:
        log.warning("Could not load processed data: %s", e)
        return None, None


def _movie_genres_map(movies: pd.DataFrame) -> dict[int, set[str]]:
    """Return a cached MovieLens movie ID to genre mapping."""
    if "movie_genres" not in _models:
        result = {}
        for mid, raw in zip(movies["movieId"], movies["genres_list"]):
            try:
                result[int(mid)] = set(raw) if raw is not None else set()
            except TypeError:
                result[int(mid)] = set()
        _models["movie_genres"] = result
    return _models["movie_genres"]


def _get_movie_stats(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Cache aggregate rating and release-year data used by ranking signals."""
    if "movie_stats" not in _models:
        stats = ratings.groupby("movieId").agg(
            mean_rating=("rating", "mean"), n_ratings=("rating", "size"),
        )
        stats["release_year"] = movies.set_index("movieId")["release_year"]
        _models["movie_stats"] = stats
    return _models["movie_stats"]


def _load_models():
    """Load trained models from disk (production mode)."""
    global _models_loaded, _models

    if _models_loaded:
        return _models_loaded

    _models_loaded = True

    # Try to load CF models (ALS + ItemKNN)
    als_path = MODELS_DIR / "als.joblib"
    knn_path = MODELS_DIR / "itemknn.joblib"
    indices_path = MODELS_DIR / "cf_indices.joblib"

    if als_path.exists() and indices_path.exists():
        try:
            from src.models.collaborative import ImplicitALSModel
            import joblib
            _models["als"] = ImplicitALSModel.load(als_path)
            _models["cf_indices"] = joblib.load(indices_path)
            log.info("Loaded ALS model from %s", als_path)
        except Exception as e:
            log.warning("Failed to load ALS model: %s", e)

    if knn_path.exists() and indices_path.exists():
        try:
            from src.models.collaborative import ItemKNN
            _models["itemknn"] = ItemKNN.load(knn_path)
            log.info("Loaded ItemKNN model from %s", knn_path)
        except Exception as e:
            log.warning("Failed to load ItemKNN model: %s", e)

    # Try to load the ranker
    ranker_path = MODELS_DIR / "ranker.joblib"
    if ranker_path.exists():
        try:
            from src.ranking.ranker import PointwiseRanker
            _models["ranker"] = PointwiseRanker.load(ranker_path)
            log.info("Loaded ranker from %s", ranker_path)
        except Exception as e:
            log.warning("Failed to load ranker: %s", e)

    # Try to load hybrid tuning results
    tuning_path = PROCESSED_DATA_DIR.parent.parent / "docs" / "results" / "hybrid_tuning.json"
    if tuning_path.exists():
        try:
            with open(tuning_path, "r") as f:
                _models["hybrid_weights"] = json.load(f)["best_weights"]
            log.info("Loaded hybrid weights: %s", _models["hybrid_weights"])
        except Exception as e:
            log.warning("Failed to load hybrid tuning: %s", e)

    return _models_loaded


def _get_popular_movies(ratings: pd.DataFrame, movies: pd.DataFrame, k: int = 20) -> list[MovieBasic]:
    """Fallback: return most popular movies by Bayesian average."""
    from src.models.popularity import PopularityModel

    movie_stats = _get_movie_stats(ratings, movies)
    pop = PopularityModel(movie_stats)
    top_pop = pop.top_popular(k=k)

    result = []
    for mid in top_pop.index:
        movie_row = movies[movies["movieId"] == mid].iloc[0]
        result.append(MovieBasic(
            movie_id=int(mid),
            title=movie_row["title_clean"],
            genres=movie_row.get("genres_list", []),
            release_year=_optional_int(movie_row.get("release_year")),
            poster_path=movie_row.get("poster_path"),
        ))
    return result


def _get_demo_recommendations(user_id: int, k: int = 10) -> list[RecommendationItem]:
    """Demo mode: return hardcoded popular movies with explanations."""
    # These are well-known popular movies from MovieLens with TMDb poster paths
    demo_movies = [
        (2571, "Matrix, The (1999)", ["Action", "Sci-Fi"], 1999, "Trending among users like you", "/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg"),
        (296, "Pulp Fiction (1994)", ["Crime", "Drama"], 1994, "Classic favorite", "/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg"),
        (593, "Silence of the Lambs, The (1991)", ["Crime", "Horror", "Thriller"], 1991, "Highly rated thriller", "/rplLJ2hPcOQmkFhTqUte0MkEaO2.jpg"),
        (1210, "Star Wars: Episode IV - A New Hope (1977)", ["Action", "Adventure", "Sci-Fi"], 1977, "Sci-fi epic", "/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg"),
        (356, "Forrest Gump (1994)", ["Comedy", "Drama", "Romance"], 1994, "Beloved classic", "/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg"),
        (260, "Star Wars: Episode V - The Empire Strikes Back (1980)", ["Action", "Adventure", "Sci-Fi"], 1980, "Highest-rated Star Wars", "/nNAeTmF4CtdSgMDplXTDPOWMYr7.jpg"),
        (589, "Terminator 2: Judgment Day (1991)", ["Action", "Sci-Fi", "Thriller"], 1991, "Action masterpiece", "/vqzNJRH4YyquRiWxCCOH0mXwZJc.jpg"),
        (1196, "Star Wars: Episode VI - Return of the Jedi (1983)", ["Action", "Adventure", "Sci-Fi"], 1983, "Sci-fi classic", "/aKxFKvqFoVzBCbdRCv8WcpRvJBH.jpg"),
        (541, "Blade Runner (1982)", ["Action", "Sci-Fi", "Thriller"], 1982, "Sci-fi noir classic", "/o6AYfQ2SULBwFvFoqwVsHFByr1u.jpg"),
        (318, "Shawshank Redemption, The (1994)", ["Drama"], 1994, "Top-rated drama", "/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg"),
    ]

    result = []
    for i, (mid, title, genres, year, explanation, poster_path) in enumerate(demo_movies[:k]):
        result.append(RecommendationItem(
            movie=MovieBasic(
                movie_id=mid,
                title=title,
                genres=genres,
                release_year=year,
                poster_path=poster_path,
            ),
            score=round(1.0 - i * 0.05, 2),
            rank=i + 1,
            explanation=explanation,
            evidence=["Curated example from the demo recommendation list"],
        ))
    return result


def _get_hybrid_recommendations(user_id: int, k: int) -> list[RecommendationItem]:
    """Production hybrid blend using trained CF models (ItemKNN + Content + Popularity)."""
    movies, ratings = _load_processed_data()
    if movies is None or ratings is None:
        return _get_demo_recommendations(user_id, k)

    # Check if we have trained CF models
    if "itemknn" not in _models or "cf_indices" not in _models:
        # Fallback to smart recommendations (genre-based)
        return _get_smart_recommendations(user_id, k)

    # Get user's interactions and filter seen items
    user_interactions = get_user_interactions(user_id)
    seen_items = {int(i["movie_id"]) for i in user_interactions}

    # Load hybrid weights (tuned on validation)
    weights = _models.get("hybrid_weights", {"cf": 0.55, "content": 0.4, "popularity": 0.05})

    # ---- 1. Get CF scores from ItemKNN (works for ANY user with history) ----
    cf_scores = {}
    if len(user_interactions) >= 3:  # Minimum items for meaningful CF
        try:
            knn_scores = _get_cf_scores_itemknn(user_id, user_interactions, movies, seen_items)
            als_scores = _get_cf_scores_als(user_interactions, seen_items)
            # Keep both trained retrieval signals. Normalize each before
            # combining so ALS scale cannot drown out ItemKNN (or vice versa).
            from src.models.hybrid import minmax
            als_norm = minmax(als_scores)
            knn_norm = minmax(knn_scores)
            cf_scores = {
                mid: 0.7 * als_norm.get(mid, 0.0)
                     + 0.3 * knn_norm.get(mid, 0.0)
                for mid in set(als_scores) | set(knn_scores)
            }
        except Exception as e:
            log.warning("CF scoring failed for user %d: %s", user_id, e)

    # ---- 2. Get Content scores (genre affinity from user taste) ----
    content_scores = _get_content_scores(user_id, user_interactions, movies, seen_items)

    # ---- 3. Get Popularity scores ----
    pop_scores = _get_popularity_scores(movies, ratings, seen_items)

    # ---- 4. Blend using HybridRecommender ----
    from src.models.hybrid import HybridRecommender, HybridWeights

    # Wrap scores as callables for HybridRecommender
    def cf_scorer(uid): return cf_scores
    def content_scorer(uid): return content_scores
    def pop_scorer(uid): return pop_scores

    hybrid = HybridRecommender(
        cf_scorer=cf_scorer,
        content_scorer=content_scorer,
        popularity_scorer=pop_scorer,
        weights=HybridWeights(
            cf=weights.get("cf", 0.55),
            content=weights.get("content", 0.4),
            popularity=weights.get("popularity", 0.05)
        )
    )

    # Get top 2k candidates for filtering
    blended = hybrid.score_items(user_id, candidate_items=None, seen_items=seen_items)
    ranked = sorted(blended.items(), key=lambda x: (-x[1], x[0]))[:k * 2]

    # Convert to RecommendationItem with explanations
    result = []
    genre_profile = _get_user_taste_profile(user_id)["genres"]
    from src.models.hybrid import minmax
    cf_explanation_scores = minmax(cf_scores)
    content_explanation_scores = minmax(content_scores)
    pop_explanation_scores = minmax(pop_scores)
    for i, (mid, score) in enumerate(ranked):
        movie_row = movies[movies["movieId"] == mid]
        if movie_row.empty:
            continue
        movie_row = movie_row.iloc[0]

        # Determine dominant component for explanation
        explanation = _get_hybrid_explanation(
            mid, cf_explanation_scores, content_explanation_scores, pop_explanation_scores, weights,
        )
        evidence = []
        if cf_scores.get(mid, 0) > 0:
            evidence.append("ALS and ItemKNN found a collaborative match")
        matched_genres = [g for g in movie_row.get("genres_list", []) if g in genre_profile]
        if content_scores.get(mid, 0) > 0 and matched_genres:
            evidence.append("Matches your " + ", ".join(matched_genres[:2]) + " preferences")
        if pop_scores.get(mid, 0) > 0:
            evidence.append("Strong Bayesian rating and popularity signal")

        result.append(RecommendationItem(
            movie=MovieBasic(
                movie_id=int(mid),
                title=movie_row["title_clean"],
                genres=movie_row.get("genres_list", []),
                release_year=_optional_int(movie_row.get("release_year")),
                poster_path=movie_row.get("poster_path"),
            ),
            score=round(float(score), 3),
            rank=i + 1,
            explanation=explanation,
            evidence=evidence,
        ))

    return result[:k]


def _get_cf_scores_itemknn(user_id: int, interactions: list[dict], movies: pd.DataFrame, seen_items: set[int]) -> dict[int, float]:
    """Get CF scores from ItemKNN model for a user.

    ItemKNN is user-agnostic at inference: it only needs the user's item history
    and the precomputed item-item similarity matrix. Works for new API users!
    """
    itemknn = _models["itemknn"]
    cf_indices = _models["cf_indices"]
    item_index = cf_indices["item_index"]  # external movieId -> internal col id
    item_by_col = cf_indices["item_by_col"]  # internal col id -> external movieId
    n_items = cf_indices.get("n_items", itemknn.n_items)

    # Build user's history in CF internal IDs
    hist_items = []
    hist_conf = []
    for inter in interactions:
        mid = int(inter["movie_id"])
        if mid in item_index:
            internal_id = int(item_index[mid])
            # Weight by interaction type
            weight = _interaction_weight(inter)
            hist_items.append(internal_id)
            hist_conf.append(weight)

    if not hist_items:
        return {}

    # Direct computation: score(i) = sum_j conf(j) * sim(i, j)
    # itemknn.similarities is (n_items, n_items) CSR
    import scipy.sparse as sp
    import numpy as np

    sims = itemknn.similarities[hist_items]  # (len(hist), n_items)
    # Weight by confidence
    weighted_sims = sims.multiply(np.array(hist_conf)[:, None])
    scores = np.asarray(weighted_sims.sum(axis=0)).ravel()

    # Mask seen items
    seen_internal = {int(item_index[mid]) for mid in seen_items if mid in item_index}
    if seen_internal:
        scores[list(seen_internal)] = -np.inf

    # Convert to external movieId -> score dict
    result = {}
    for internal_id, score in enumerate(scores):
        if score > 0:
            mid = item_by_col.get(internal_id)
            if mid is not None:
                result[mid] = float(score)

    return result


def _get_cf_scores_als(interactions: list[dict], seen_items: set[int]) -> dict[int, float]:
    """Score catalog items from ALS item factors for API users.

    API account IDs are independent of MovieLens user IDs, so build a
    lightweight preference vector from the user's interacted MovieLens
    items rather than treating the API ID as a trained ALS row number.
    """
    if "als" not in _models or "cf_indices" not in _models:
        return {}
    als_model = _models["als"].model
    indices = _models["cf_indices"]
    item_index = indices["item_index"]
    item_by_col = indices["item_by_col"]
    factors = als_model.item_factors
    weighted = []
    for inter in interactions:
        mid = int(inter["movie_id"])
        col = item_index.get(mid) if hasattr(item_index, "get") else None
        if col is None:
            continue
        weight = max(0.0, _interaction_weight(inter))
        if weight:
            weighted.append((int(col), weight))
    if not weighted:
        return {}
    profile = sum(factors[col] * weight for col, weight in weighted) / sum(w for _, w in weighted)
    scores = factors @ profile
    seen_cols = {int(item_index[mid]) for mid in seen_items if mid in item_index}
    result = {}
    for col in np.argsort(-scores)[: min(5000, len(scores))]:
        mid = item_by_col.get(int(col))
        if mid is not None and mid not in seen_items and int(col) not in seen_cols:
            result[int(mid)] = float(scores[col])
    return result


def _get_content_scores(user_id: int, interactions: list[dict], movies: pd.DataFrame, seen_items: set[int]) -> dict[int, float]:
    """Get content-based scores from user's genre affinity."""
    if not interactions:
        return {}

    profile = _get_user_taste_profile(user_id)
    genre_scores = profile["genres"]
    liked_movies = profile["liked_movies"]
    rated_movies = profile["rated_movies"]

    if not genre_scores:
        return {}

    movie_scores = {}
    max_genre_score = max(genre_scores.values()) if genre_scores else 1

    for _, movie in movies.iterrows():
        mid = int(movie["movieId"])
        if mid in seen_items:
            continue

        movie_genres = movie.get("genres_list", [])
        try:
            # Handle numpy arrays from parquet
            if movie_genres is None or len(movie_genres) == 0:
                continue
        except TypeError:
            # NaN or unhashable
            continue

        genre_match = sum(genre_scores.get(g, 0) for g in movie_genres)
        max_possible = sum(max_genre_score for _ in movie_genres)

        if genre_match > 0 and max_possible > 0:
            movie_scores[mid] = genre_match / max_possible

    return movie_scores


def _get_popularity_scores(movies: pd.DataFrame, ratings: pd.DataFrame, seen_items: set[int]) -> dict[int, float]:
    """Get popularity scores for all movies."""
    from src.models.popularity import PopularityModel

    movie_stats = _get_movie_stats(ratings, movies)

    pop = PopularityModel(movie_stats)
    top_pop = pop.top_popular(k=5000)  # Large pool, returns DataFrame

    # top_pop has columns: bayesian, popularity_score
    # Index is movieId
    result = {}
    for mid in top_pop.index:
        if mid not in seen_items:
            score = top_pop.loc[mid, "bayesian"]
            result[int(mid)] = float(score)
    return result


def _interaction_weight(interaction: dict) -> float:
    """Convert interaction to confidence weight."""
    weights = {
        "like": 3.0,
        "rate": 1.0,
        "rate": 1.0,
        "watchlist": 2.5,
        "remove_watchlist": 0.0,
        "complete": 2.0,
        "watch": 1.5,
        "view": 1.0,
        "click": 0.5,
        "dislike": -2.0,
        "skip": -1.0,
    }
    weight = weights.get(interaction.get("interaction_type", ""), 1.0)

    # Boost by rating if present
    rating = interaction.get("rating")
    if rating:
        weight *= (rating / 3.0)

    return weight


def _get_hybrid_explanation(mid: int, cf_scores: dict, content_scores: dict, pop_scores: dict, weights: dict) -> str:
    """Determine which component dominates for explanation."""
    cf = cf_scores.get(mid, 0) * weights.get("cf", 0.55)
    ct = content_scores.get(mid, 0) * weights.get("content", 0.4)
    pop = pop_scores.get(mid, 0) * weights.get("popularity", 0.05)

    dominant = max(
        [("collaborative", cf), ("content", ct), ("popularity", pop)],
        key=lambda x: x[1]
    )[0]

    reasons = {
        "collaborative": "Users with similar taste liked this",
        "content": "Matches your genre preferences",
        "popularity": "Popular and highly rated"
    }
    return reasons.get(dominant, "Recommended for you")


def _get_ranker_recommendations(user_id: int, k: int) -> list[RecommendationItem]:
    """Production ranker re-ranking using trained HistGBM model."""
    # Check if ranker is loaded
    if "ranker" not in _models:
        log.info("Ranker not loaded, falling back to hybrid")
        return _get_hybrid_recommendations(user_id, k)

    movies, ratings = _load_processed_data()
    if movies is None or ratings is None:
        return _get_hybrid_recommendations(user_id, k)

    interactions = get_user_interactions(user_id)
    seen = {int(row["movie_id"]) for row in interactions}
    # Candidate retrieval uses the same tuned hybrid as the first-stage
    # model; request a wider pool before asking the learned ranker to order it.
    candidates = _get_hybrid_recommendations(user_id, 100)
    if not candidates:
        return candidates

    global _ranker_stats
    try:
        from src.models.popularity import PopularityModel
        from src.ranking.features import UserItemStats, build_features
        if _ranker_stats is None:
            movie_stats = _get_movie_stats(ratings, movies)
            popularity = PopularityModel(movie_stats)
            item_genres = {}
            for mid, raw in zip(movies["movieId"], movies["genres_list"]):
                try:
                    item_genres[int(mid)] = set(raw) if raw is not None and len(raw) else set()
                except TypeError:
                    item_genres[int(mid)] = set()
            _ranker_stats = UserItemStats(
                item_n_ratings={int(mid): int(n) for mid, n in movie_stats["n_ratings"].items()},
                item_avg_rating={int(mid): float(value) for mid, value in movie_stats["mean_rating"].items()},
                item_genres=item_genres,
                item_freshness={int(mid): float(value) for mid, value in popularity.freshness.items()},
                item_pop_score={int(mid): float(value) for mid, value in popularity.popularity.items()},
            )

        genres_by_id = {}
        for _, row in movies.iterrows():
            raw = row.get("genres_list", [])
            try:
                genres_by_id[int(row["movieId"])] = set(raw) if raw is not None else set()
            except TypeError:
                genres_by_id[int(row["movieId"])] = set()
        profile = {}
        for inter in interactions:
            mid = int(inter["movie_id"])
            weight = max(0.0, _interaction_weight(inter))
            for genre in genres_by_id.get(mid, set()):
                profile[genre] = profile.get(genre, 0.0) + weight
        total = sum(profile.values())
        if total:
            profile = {genre: value / total for genre, value in profile.items()}

        item_ids = [rec.movie.movie_id for rec in candidates]
        content_all = _get_content_scores(user_id, interactions, movies, seen)
        als_all = _get_cf_scores_als(interactions, seen)
        knn_all = _get_cf_scores_itemknn(user_id, interactions, movies, seen) if "itemknn" in _models else {}
        _ranker_stats.user_genre_profile[user_id] = profile
        _ranker_stats.user_n_ratings[user_id] = len(interactions)
        features = build_features(
            [user_id], {user_id: item_ids},
            {user_id: als_all}, {user_id: knn_all}, {user_id: content_all}, _ranker_stats,
        )
        if features.empty:
            return candidates[:k]
        ranked = _models["ranker"].rerank(features, k=k)
        candidate_by_id = {rec.movie.movie_id: rec for rec in candidates}
        result = []
        for rank, (mid, score) in enumerate(ranked, start=1):
            rec = candidate_by_id.get(mid)
            if rec is not None:
                rec.score = round(float(score), 4)
                rec.rank = rank
                rec.explanation = f"Learned from recommendation patterns: {rec.explanation}"
                rec.evidence = list(rec.evidence or []) + ["The learned ranker combined your preference and item signals"]
                result.append(rec)
        return result or candidates[:k]
    except Exception:
        log.exception("Ranker feature generation failed for user %d; using hybrid order", user_id)
        return candidates[:k]


def record_interaction(
    user_id: int,
    movie_id: int,
    interaction_type: str,
    rating: Optional[float] = None,
) -> None:
    """Record a user interaction (like, watch, skip, etc.).

    This feeds the dynamic profile update (Milestone 6) and
    provides training data for future model retraining.
    """
    with get_db() as db:
        # Create interaction record
        interaction = Interaction(
            user_id=user_id,
            movie_id=movie_id,
            interaction_type=interaction_type,
            rating=rating,
        )
        db.add(interaction)

        # Update user profile (incremental)
        _update_user_profile(db, user_id, movie_id, interaction_type, rating)

        log.info(
            "Recorded interaction: user=%d movie=%d type=%s rating=%s",
            user_id, movie_id, interaction_type, rating,
        )


def _update_user_profile(
    db,  # Session
    user_id: int,
    movie_id: int,
    interaction_type: str,
    rating: Optional[float],
) -> None:
    """Incremental user profile update based on new interaction."""
    from datetime import datetime, timezone
    from src.recommendations.profile_updater import ProfileUpdater

    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    if not profile:
        profile = UserProfile(user_id=user_id, n_interactions=0, avg_rating=None, genre_affinity={})
        db.add(profile)

    genres: set[str] = set()
    movies, _ = _load_processed_data()
    if movies is not None:
        row = movies.loc[movies["movieId"] == movie_id, "genres_list"]
        if not row.empty:
            raw = row.iloc[0]
            try:
                genres = set(raw) if raw is not None else set()
            except TypeError:
                genres = set()

    now = datetime.now(timezone.utc)
    last_updated = profile.last_updated
    if last_updated is not None and last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=timezone.utc)
    update = ProfileUpdater().update_on_interaction(
        genre_affinity=dict(profile.genre_affinity or {}),
        movie_genres=genres,
        interaction_type=interaction_type,
        rating=rating,
        timestamp=now,
        last_updated=last_updated,
        n_interactions=profile.n_interactions or 0,
        avg_rating=profile.avg_rating,
    )
    profile.genre_affinity = update["genre_affinity"]
    profile.n_interactions = update["n_interactions"]
    profile.avg_rating = update["avg_rating"]
    profile.last_updated = now


def get_user_profile(user_id: int) -> UserProfile | None:
    """Fetch user's current profile."""
    with get_db() as db:
        return db.query(UserProfile).filter(UserProfile.user_id == user_id).first()


def search_movies(
    query: Optional[str] = None,
    genres: Optional[list[str]] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    min_rating: Optional[float] = None,
    limit: int = 20,
) -> list[MovieBasic]:
    """Search movies by title, genre, year, rating."""
    movies, ratings = _load_processed_data()
    if movies is None:
        return []

    df = movies.copy()

    if query:
        df = df[df["title_clean"].str.contains(query, case=False, na=False)]

    if genres:
        df = df[df["genres_list"].apply(lambda gs: any(g in gs for g in genres))]

    if min_year:
        df = df[df["release_year"] >= min_year]
    if max_year:
        df = df[df["release_year"] <= max_year]

    if min_rating is not None and ratings is not None:
        # Filter by average rating
        movie_avg = ratings.groupby("movieId")["rating"].mean()
        high_rated = movie_avg[movie_avg >= min_rating].index
        df = df[df["movieId"].isin(high_rated)]

    df = df.head(limit)

    return [
        MovieBasic(
            movie_id=int(row["movieId"]),
            title=row["title_clean"],
            genres=row.get("genres_list", []),
            release_year=int(row["release_year"]) if pd.notna(row.get("release_year")) else None,
            poster_path=row.get("poster_path"),
        )
        for _, row in df.iterrows()
    ]


def get_user_interactions(user_id: int, limit: int = 500) -> list[dict]:
    """Get all user interactions for personalized recommendations."""
    from src.api.models import Interaction
    with get_db() as db:
        interactions = (
            db.query(Interaction)
            .filter(Interaction.user_id == user_id)
            .order_by(Interaction.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "movie_id": i.movie_id,
                "interaction_type": i.interaction_type,
                "rating": i.rating,
                "timestamp": i.timestamp.isoformat() if i.timestamp else None,
            }
            for i in interactions
        ]


def _get_user_taste_profile(user_id: int) -> dict:
    """Build a taste profile from user's interactions."""
    movies, ratings = _load_processed_data()
    if movies is None:
        return {"genres": {}, "liked_movies": [], "rated_movies": {}}

    user_interactions = get_user_interactions(user_id)

    saved_profile = get_user_profile(user_id)
    genre_scores = {}
    liked_movies = []
    rated_movies = {}
    genres_by_movie = _movie_genres_map(movies)

    for interaction in user_interactions:
        movie_id = interaction["movie_id"]
        interaction_type = interaction["interaction_type"]
        rating = interaction.get("rating")

        # Get movie genres
        movie_genres = genres_by_movie.get(int(movie_id), set())

        # Weight based on interaction type
        weights = {
            "like": 3.0,
            "watchlist": 2.5,
            "remove_watchlist": 0.0,
            "complete": 2.0,
            "watch": 1.5,
            "view": 1.0,
            "click": 0.5,
            "dislike": -2.0,
            "skip": -1.0,
        }
        weight = weights.get(interaction_type, 1.0)

        # Boost by rating if present
        if rating:
            weight *= (rating / 3.0)  # Normalize around 3 stars
            rated_movies[movie_id] = rating

        if interaction_type in ("like", "watchlist", "complete"):
            liked_movies.append(movie_id)

        for genre in movie_genres:
            genre_scores[genre] = genre_scores.get(genre, 0) + weight

    if saved_profile and saved_profile.genre_affinity:
        # The persisted profile already includes interaction evidence. Use it
        # directly to avoid counting every event twice on each recommendation.
        genre_scores = {genre: float(score) * 6.0 for genre, score in saved_profile.genre_affinity.items()}

    return {
        "genres": genre_scores,
        "liked_movies": liked_movies,
        "rated_movies": rated_movies,
    }


def _get_smart_recommendations(user_id: int, k: int = 20) -> list[RecommendationItem]:
    """Generate recommendations based on user's actual taste profile."""
    movies, ratings = _load_processed_data()
    if movies is None:
        return _get_demo_recommendations(user_id, k)

    profile = _get_user_taste_profile(user_id)
    liked_movies = profile["liked_movies"]
    genre_scores = profile["genres"]
    rated_movies = profile["rated_movies"]

    if not liked_movies and not genre_scores:
        # New user - return popular movies wrapped in RecommendationItem
        popular = _get_popular_movies(ratings, movies, k)
        return [
            RecommendationItem(
                movie=m,
                score=round(1.0 - i * 0.05, 2),
                rank=i + 1,
                explanation="Popular overall",
                evidence=["High Bayesian average rating", "Strong rating count across the catalog"],
            )
            for i, m in enumerate(popular)
        ]

    # Score all movies based on genre affinity
    movie_scores = {}
    genres_by_movie = _movie_genres_map(movies)
    rating_stats = _models.get("rating_stats")
    if rating_stats is None:
        rating_stats = {
            int(mid): (float(row["mean_rating"]), int(row["n_ratings"]))
            for mid, row in _get_movie_stats(ratings, movies).iterrows()
        }
        _models["rating_stats"] = rating_stats
    movie_rows = movies.set_index("movieId")

    for _, movie in movies.iterrows():
        mid = int(movie["movieId"])

        # Skip already liked/rated movies
        if mid in liked_movies or mid in rated_movies:
            continue

        movie_genres = genres_by_movie.get(mid, set())
        if not movie_genres:
            continue

        # Calculate genre match score
        genre_match = sum(genre_scores.get(g, 0) for g in movie_genres)
        max_possible = sum(max(genre_scores.values()) for _ in movie_genres) if genre_scores else 1

        if genre_match > 0:
            # Normalize and add popularity boost
            popularity = 0.3
            rating_stat = rating_stats.get(mid)
            if rating_stat:
                avg_rating, n_ratings = rating_stat
                popularity = (avg_rating / 5.0) * min(n_ratings / 100, 1.0) + 0.3

            score = (genre_match / max_possible) * 0.7 + popularity * 0.3
            movie_scores[mid] = score

    # Get top scored movies
    top_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    result = []
    for i, (mid, score) in enumerate(top_movies):
        movie_row = movie_rows.loc[mid]
        # Find matching genres for explanation
        movie_genres = movie_row.get("genres_list", [])
        matching_genres = [g for g in movie_genres if g in genre_scores]
        explanation = f"Matches your {', '.join(matching_genres[:2])} preference" if matching_genres else "Recommended for you"

        result.append(RecommendationItem(
            movie=MovieBasic(
                movie_id=int(mid),
                title=movie_row["title_clean"],
                genres=movie_genres,
                release_year=_optional_int(movie_row.get("release_year")),
                poster_path=movie_row.get("poster_path"),
            ),
            score=round(score, 3),
            rank=i + 1,
            explanation=explanation,
            evidence=(
                ["Matches your " + ", ".join(matching_genres[:2]) + " preferences"] if matching_genres else []
            ) + ["Popularity and rating history helped surface this title"],
        ))

    # Fill with popular if not enough
    if len(result) < k:
        popular = _get_popular_movies(ratings, movies, k - len(result))
        for m in popular:
            if m.movie_id not in liked_movies and m.movie_id not in rated_movies:
                result.append(RecommendationItem(
                    movie=m,
                    score=round(0.3 - len(result) * 0.02, 2),
                    rank=len(result) + 1,
                    explanation="Popular overall",
                    evidence=["High Bayesian average rating", "Strong rating count across the catalog"],
                ))

    return result


def get_recommendations(
    user_id: int,
    k: int = 10,
    method: str = "auto",
) -> list[RecommendationItem]:
    """Get top-k recommendations for a user.

    Args:
        user_id: Target user ID
        k: Number of recommendations to return
        method: "auto" | "smart" | "hybrid" | "ranker" | "popularity" | "demo"

    Returns:
        List of RecommendationItem with movie info, score, rank, explanation
    """
    # Check if user exists
    with get_db() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")

    # Load models
    _load_models()

    # Smart recommendations based on user's actual interactions
    if method == "smart":
        smart_recs = _get_smart_recommendations(user_id, k)
        if smart_recs:
            return smart_recs

    # If we have a trained ranker and method requests it, use it
    if method in ("auto", "ranker") and "ranker" in _models:
        return _get_ranker_recommendations(user_id, k)

    # If we have hybrid weights, use hybrid blend
    if method in ("auto", "hybrid") and "hybrid_weights" in _models:
        return _get_hybrid_recommendations(user_id, k)

    # Try to use processed data for popularity fallback
    movies, ratings = _load_processed_data()
    if movies is not None and ratings is not None:
        if method in ("auto", "popularity"):
            popular = _get_popular_movies(ratings, movies, k)
            return [
                RecommendationItem(
                    movie=m,
                    score=round(1.0 - i * 0.05, 2),
                    rank=i + 1,
                    explanation="Popular overall",
                    evidence=["High Bayesian average rating", "Strong rating count across the catalog"],
                )
                for i, m in enumerate(popular)
            ]

    # Final fallback: demo mode
    log.info("Falling back to demo mode for user %d", user_id)
    return _get_demo_recommendations(user_id, k)


def postprocess_recommendations(
    recommendations: list[RecommendationItem],
    k: int,
    mode: str = "personalized",
    diversity: float = 1.0,
) -> list[RecommendationItem]:
    """Apply optional novelty and diversity controls to a ranked pool."""
    if not recommendations:
        return []
    movies, ratings = _load_processed_data()
    if ratings is None or movies is None:
        return recommendations[:k]

    ranked = [(rec.movie.movie_id, float(rec.score)) for rec in recommendations]
    pop_scores = _models.get("popularity_scores")
    if pop_scores is None:
        from src.models.popularity import PopularityModel
        model = PopularityModel(_get_movie_stats(ratings, movies))
        pop_scores = {int(mid): float(score) for mid, score in model.popularity.items()}
        _models["popularity_scores"] = pop_scores

    if mode == "explore":
        from src.recommendations.exploration import ExplorationPolicy
        explored = ExplorationPolicy().apply(ranked, pop_scores, mode="explore")
        by_id = {rec.movie.movie_id: rec for rec in recommendations}
        ranked = explored
        for mid, score in ranked:
            if score > by_id[mid].score:
                by_id[mid].score = round(float(score), 4)
                by_id[mid].explanation = "Explore mode: a less-seen pick, plus " + (by_id[mid].explanation or "your taste profile")
                by_id[mid].evidence = list(by_id[mid].evidence or []) + ["Explore mode boosted a less popular catalog item"]

    if 0.0 <= diversity < 1.0:
        try:
            from src.features.content_features import build_movie_feature_matrix
            from src.recommendations.mmr import mmr_rerank
            if "content_matrix" not in _models:
                genome_path = PROCESSED_DATA_DIR / "genome_scores.parquet"
                genome = pd.read_parquet(genome_path)
                matrix, movie_index, _ = build_movie_feature_matrix(genome, movies)
                _models["content_matrix"] = (matrix, movie_index)
            matrix, movie_index = _models["content_matrix"]
            before_ids = [mid for mid, _ in ranked[:k]]
            diversified = mmr_rerank(ranked, matrix, movie_index, lambda_param=diversity, k=k)
            rec_by_id = {rec.movie.movie_id: rec for rec in recommendations}
            ranked = diversified
            selected = []
            for rank, (mid, score) in enumerate(ranked, start=1):
                rec = rec_by_id[mid]
                rec.rank = rank
                rec.score = round(float(score), 4)
                if mid not in before_ids:
                    rec.explanation = "Selected for variety: " + (rec.explanation or "a different genre mix")
                    rec.evidence = list(rec.evidence or []) + ["MMR selected this item to add variety"]
                selected.append(rec)
            return selected
        except Exception:
            log.exception("Diversity reranking failed; returning relevance order")

    by_id = {rec.movie.movie_id: rec for rec in recommendations}
    result = []
    for rank, (mid, _) in enumerate(ranked[:k], start=1):
        rec = by_id.get(mid)
        if rec is not None:
            rec.rank = rank
            result.append(rec)
    return result
