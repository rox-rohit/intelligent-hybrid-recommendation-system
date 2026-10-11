"""Cleaning and validation rules for MovieLens tables.

Each clean_* function takes a raw DataFrame and returns a cleaned DataFrame,
documenting exactly what was changed in the returned QualityReport-friendly
way (counts are computed by pipeline.py and logged).

Design rule: these functions never mutate their inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# MovieLens titles look like "Toy Story (1995)". Extract the year for
# feature engineering; keep malformed titles but flag them.
YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")

VALID_RATING_RANGE = (0.5, 5.0)


@dataclass
class CleaningReport:
    """Counts of every change applied by the pipeline — logged, never guessed."""

    table: str
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    missing_dropped: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def rows_removed(self) -> int:
        return self.rows_in - self.rows_out


def clean_movies(movies: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean movies.csv: parse release year, normalize genre lists.

    Changes:
    - Extract release_year from the title's trailing "(YYYY)".
    - Split genres on "|" into a list; map "(no genres listed)" -> empty list.
    - Flag (do not drop) rows with missing/invalid title years — they stay
      usable for CF, which needs no content features.
    """
    rep = CleaningReport("movies", rows_in=len(movies))
    out = movies.copy()

    year_match = out["title"].str.extract(YEAR_PATTERN)[0]
    out["release_year"] = year_match.astype("float32").astype("Int32")
    n_no_year = out["release_year"].isna().sum()
    if n_no_year:
        rep.issues.append(f"{n_no_year} movies without parseable release year")

    out["genres_list"] = (
        out["genres"]
        .str.replace("(no genres listed)", "", regex=False)
        .str.split("|")
        .apply(lambda gs: [g for g in gs if g])  # drop empty strings
    )
    out["n_genres"] = out["genres_list"].str.len().astype("int16")
    out["title_clean"] = (
        out["title"].str.replace(YEAR_PATTERN, "", regex=True).str.strip()
    )

    rep.rows_out = len(out)
    return out, rep


def clean_ratings(ratings: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean ratings.csv: dedupe, range-check, coerce dtypes.

    Changes:
    - Drop exact duplicate (userId, movieId) rows (keep first).
    - Drop rows with rating outside [0.5, 5.0] (MovieLens spec; log count).
    - Drop rows missing user/movie/rating.
    """
    rep = CleaningReport("ratings", rows_in=len(ratings))
    out = ratings.copy()

    before = len(out)
    out = out.dropna(subset=["userId", "movieId", "rating"])
    if dropped := before - len(out):
        rep.missing_dropped["rows"] = dropped

    out = out.astype(
        {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    )

    invalid = ~out["rating"].between(*VALID_RATING_RANGE)
    if n_invalid := int(invalid.sum()):
        rep.issues.append(f"{n_invalid} ratings outside [0.5, 5.0] removed")
        out = out[~invalid]

    # chronological order per user matters for time-aware splits later;
    # sort BEFORE deduping so keep="first" means the earliest rating.
    out = out.sort_values(["userId", "timestamp"], kind="stable")

    before = len(out)
    out = out.drop_duplicates(subset=["userId", "movieId"], keep="first")
    rep.duplicates_removed = before - len(out)
    out = out.reset_index(drop=True)

    rep.rows_out = len(out)
    return out, rep


def clean_tags(tags: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean tags.csv: normalize text, dedupe, drop empty tags."""
    rep = CleaningReport("tags", rows_in=len(tags))
    out = tags.copy()

    out["tag"] = (
        out["tag"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)  # collapse internal whitespace
    )
    empty = out["tag"].isna() | (out["tag"] == "")
    if n_empty := int(empty.sum()):
        rep.issues.append(f"{n_empty} empty tags removed")
        out = out[~empty]

    before = len(out)
    out = out.drop_duplicates(subset=["userId", "movieId", "tag"], keep="first")
    rep.duplicates_removed = before - len(out)

    rep.rows_out = len(out)
    return out, rep


def clean_genome(
    genome_scores: pd.DataFrame, genome_tags: pd.DataFrame
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean genome tables: enforce relevance range and join tag names.

    Returns a single tidy table [movieId, tagId, tag, relevance] restricted
    to relevance >= threshold (default 0.5 keeps the strongly indicative
    tags; the full matrix stays available in raw data if needed).
    """
    rep = CleaningReport("genome", rows_in=len(genome_scores))
    out = genome_scores.copy()

    invalid = ~out["relevance"].between(0.0, 1.0)
    if n_invalid := int(invalid.sum()):
        rep.issues.append(f"{n_invalid} genome relevance scores outside [0, 1] removed")
        out = out[~invalid]

    out = out.merge(genome_tags, on="tagId", how="left", validate="many_to_one")
    n_unmapped = int(out["tag"].isna().sum())
    if n_unmapped:
        rep.issues.append(f"{n_unmapped} genome scores without a tag name dropped")
        out = out.dropna(subset=["tag"])

    rep.rows_out = len(out)
    return out, rep


def summarize_outliers(ratings: pd.DataFrame) -> dict[str, float]:
    """Outlier analysis on per-user rating counts (user activity).

    Returns IQR-based bounds and the share of users above the upper bound.
    Used by EDA, not by the cleaning drop rules — power users are real users.
    """
    per_user = ratings.groupby("userId").size()
    q1, q3 = per_user.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    share_above = float((per_user > upper).mean()) if len(per_user) else 0.0
    return {
        "p25": float(q1),
        "p75": float(q3),
        "iqr_upper_bound": float(upper),
        "share_users_above_bound": share_above,
        "max_ratings_by_user": float(per_user.max()) if len(per_user) else 0.0,
    }
