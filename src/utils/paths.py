"""Central path resolution. Import from here instead of hardcoding paths."""

from __future__ import annotations

from pathlib import Path

# Project root = the directory containing src/, data/, scripts/, ...
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# MovieLens 25M extraction target
MOVIELENS_DIR = RAW_DATA_DIR / "ml-25m"


def ensure_dirs() -> None:
    """Create the standard data/model directories if missing."""
    for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
