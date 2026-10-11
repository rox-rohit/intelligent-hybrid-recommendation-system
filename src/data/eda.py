"""EDA computations and plots. Logic lives here; the notebook orchestrates.

Every function answers a specific, stated question — no decoration charts.
All plots save to docs/figures/ as PNG (matplotlib, Agg backend, no display).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: safe in scripts, notebooks, CI

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logging_config import get_logger
from src.utils.paths import DOCS_DIR

log = get_logger(__name__)
FIGURES_DIR = DOCS_DIR / "figures"

# Consistent, colorblind-safe style for all project figures.
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    log.info("Figure saved: %s", path)


# --- Movies ---------------------------------------------------------------

def plot_genre_distribution(movies: pd.DataFrame) -> pd.Series:
    """Q: Which genres dominate the catalog — and is 'no genre' common?"""
    counts = (
        movies.explode("genres_list")["genres_list"]
        .dropna()
        .value_counts()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    counts.plot.barh(ax=ax, color="#4477AA")
    ax.set_title("Catalog size by genre")
    ax.set_xlabel("Number of movies")
    _save(fig, "01_genre_distribution")
    return counts


def plot_release_year_distribution(movies: pd.DataFrame) -> pd.Series:
    """Q: How is the catalog distributed over time — and is it recency-biased?"""
    years = movies["release_year"].dropna()
    decades = (years // 10 * 10).astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    decades.plot.bar(ax=ax, color="#4477AA")
    ax.set_title("Catalog size by release decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Number of movies")
    _save(fig, "02_release_year_distribution")
    return decades


# --- Users / ratings --------------------------------------------------------

def plot_rating_distribution(ratings: pd.DataFrame) -> pd.Series:
    """Q: Do users round to whole stars? What is the modal rating?"""
    counts = ratings["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot.bar(ax=ax, color="#EE6677")
    ax.set_title("Rating value distribution")
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Number of ratings")
    _save(fig, "03_rating_distribution")
    return counts


def plot_user_activity(ratings: pd.DataFrame) -> pd.Series:
    """Q: How unequal is user activity — a few power raters or a flat base?"""
    per_user = ratings.groupby("userId").size()
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = [0, 5, 10, 20, 50, 100, 500, 1000, 10_000]
    pd.cut(per_user, bins=bins).value_counts().sort_index().plot.bar(
        ax=ax, color="#4477AA"
    )
    ax.set_title("Ratings per user (binned)")
    ax.set_xlabel("Number of ratings")
    ax.set_ylabel("Number of users")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "04_user_activity")
    return per_user


def plot_active_vs_inactive(per_user: pd.Series, threshold: int = 50) -> dict:
    """Q: What share of users are inactive (<threshold ratings) and what
    share of all ratings do they account for? (Matters for CF coverage.)"""
    active = per_user >= threshold
    stats = {
        "threshold": threshold,
        "share_users_active": float(active.mean()),
        "share_ratings_from_active": float(
            per_user[active].sum() / per_user.sum()
        ),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    share = per_user.groupby(active).sum() / per_user.sum()
    share.index = ["inactive (<50)", "active (>=50)"]
    share.plot.bar(ax=ax, color=["#CC6677", "#4477AA"])
    ax.set_title("Share of all ratings: active vs inactive users")
    ax.set_ylabel("Share of ratings")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _save(fig, "05_active_vs_inactive")
    return stats


# --- Relationships ----------------------------------------------------------

def plot_popularity_vs_rating(movies_stats: pd.DataFrame) -> pd.DataFrame:
    """Q: Do widely-rated movies receive higher mean ratings? (Answer:
    weak positive —MovieLens raters select what they expect to like.)

    movies_stats: DataFrame indexed by movieId with columns
    [mean_rating, n_ratings].
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    # hexbin, not scatter: 62k points would be an unreadable blob
    ax.hexbin(
        movies_stats["n_ratings"], movies_stats["mean_rating"],
        gridsize=40, bins="log", cmap="Blues", mincnt=1,
    )
    ax.set_title("Popularity vs. mean rating per movie")
    ax.set_xlabel("Number of ratings (log hexbin)")
    ax.set_ylabel("Mean rating")
    fig.colorbar(ax.collections[0], ax=ax, label="count (log)")
    _save(fig, "06_popularity_vs_rating")
    return movies_stats


def plot_genre_vs_rating(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.Series:
    """Q: Which genres do users rate highest on average — and does that
    simply reflect rating behavior rather than movie quality?"""
    genre_map = movies.set_index("movieId")["genres_list"]
    merged = ratings[["movieId", "rating"]].copy()
    merged["genres_list"] = merged["movieId"].map(genre_map)
    exploded = merged.explode("genres_list").dropna(subset=["genres_list"])
    genre_mean = exploded.groupby("genres_list")["rating"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    genre_mean.plot.barh(ax=ax, color="#228833")
    ax.set_title("Mean rating by genre")
    ax.set_xlabel("Mean rating")
    _save(fig, "07_genre_vs_rating")
    return genre_mean


def plot_year_vs_popularity(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.Series:
    """Q: Are recent movies rated more (recency bias in the catalog)?"""
    stats = ratings.groupby("movieId").size().rename("n_ratings").to_frame()
    stats["release_year"] = movies.set_index("movieId")["release_year"]
    stats["decade"] = (stats["release_year"] // 10 * 10).dropna().astype(int)
    med = stats.groupby("decade")["n_ratings"].median()
    fig, ax = plt.subplots(figsize=(9, 5))
    med.plot.line(ax=ax, marker="o", color="#66CCEE")
    ax.set_title("Median ratings per movie by release decade")
    ax.set_xlabel("Release decade")
    ax.set_ylabel("Median number of ratings")
    _save(fig, "08_year_vs_popularity")
    return med
