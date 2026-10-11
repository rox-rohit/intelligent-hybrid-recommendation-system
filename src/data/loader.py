"""Load raw MovieLens 25M files into typed, memory-conscious DataFrames.

Memory notes (8 GB machine):
- ratings.csv is ~25M rows. With float32/int32 dtypes it costs ~300 MB,
  which is fine; the default loader reads all of it. Callers that only
  need per-user slices should filter immediately after loading.
- genome-scores.csv is ~15M rows (62k movies x 1,128 tags); int32/float32.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import MOVIELENS_DIR

# Explicit dtypes: smaller memory, faster parses, no silent object columns.
DTYPES = {
    "movies": {"movieId": "int32", "title": "string", "genres": "string"},
    "ratings": {
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
        "timestamp": "int64",  # unix seconds; fits 2038+ comfortably
    },
    "tags": {
        "userId": "int32",
        "movieId": "int32",
        "tag": "string",
        "timestamp": "int64",
    },
    "links": {"movieId": "int32", "imdbId": "int64", "tmdbId": "Int64"},
    "genome_scores": {"movieId": "int32", "tagId": "int32", "relevance": "float32"},
    "genome_tags": {"tagId": "int32", "tag": "string"},
}

# Parquet versions of the raw files, written once by preprocess.py and
# reused by every downstream stage (faster than CSV re-parsing).
PROCESSED_FILES = {
    "movies": "movies.parquet",
    "ratings": "ratings.parquet",
    "tags": "tags.parquet",
    "links": "links.parquet",
    "genome_scores": "genome_scores.parquet",
    "genome_tags": "genome_tags.parquet",
}


def require_dataset() -> Path:
    """Return the MovieLens directory, raising a helpful error if absent."""
    if not MOVIELENS_DIR.exists():
        raise FileNotFoundError(
            f"MovieLens 25M not found at {MOVIELENS_DIR}. "
            "Run: python scripts/download_data.py"
        )
    return MOVIELENS_DIR


def load_raw(name: str, nrows: int | None = None) -> pd.DataFrame:
    """Load a raw MovieLens CSV by canonical name with correct dtypes.

    name: one of movies | ratings | tags | links | genome_scores | genome_tags
    nrows: optional row cap (used by tests and quick experiments).
    """
    if name not in DTYPES:
        raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(DTYPES)}")
    base = require_dataset()
    path = base / f"{name.replace('_', '-')}.csv"
    return pd.read_csv(path, dtype=DTYPES[name], nrows=nrows)


def load_processed(name: str) -> pd.DataFrame:
    """Load a processed (cleaned) parquet written by the preprocessing step."""
    from src.utils.paths import PROCESSED_DATA_DIR

    if name not in PROCESSED_FILES:
        raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(PROCESSED_FILES)}")
    path = PROCESSED_DATA_DIR / PROCESSED_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset missing: {path}. Run: python scripts/preprocess.py"
        )
    return pd.read_parquet(path)
