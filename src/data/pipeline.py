"""Raw -> processed pipeline orchestration.

Reads data/raw/ml-25m/*.csv (never modified), applies cleaning rules from
cleaning.py, writes parquet outputs to data/processed/, and emits a
data-quality report consumed by the EDA notebook and README.

Usage:
    python scripts/preprocess.py            (thin wrapper, see scripts/)
    or: python -m src.data.pipeline
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

from src.data import cleaning
from src.data.loader import PROCESSED_FILES, load_raw
from src.utils.logging_config import configure_logging, get_logger
from src.utils.paths import PROCESSED_DATA_DIR

log = get_logger(__name__)

# Keep only genome tag rows at/above this relevance for the processed
# content representation (raw matrix remains untouched in data/raw).
GENOME_RELEVANCE_THRESHOLD = 0.5


def run_pipeline() -> dict:
    """Execute the full raw -> processed pipeline; return the quality report."""
    configure_logging("INFO")
    t0 = time.perf_counter()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    reports: list[cleaning.CleaningReport] = []

    log.info("Loading raw movies ...")
    movies, rep = cleaning.clean_movies(load_raw("movies"))
    reports.append(rep)

    log.info("Loading raw ratings (25M rows — this takes a minute) ...")
    ratings, rep = cleaning.clean_ratings(load_raw("ratings"))
    reports.append(rep)

    log.info("Loading raw tags ...")
    tags, rep = cleaning.clean_tags(load_raw("tags"))
    reports.append(rep)

    log.info("Loading raw links ...")
    links = load_raw("links")
    reports.append(cleaning.CleaningReport("links", rows_in=len(links), rows_out=len(links)))

    log.info("Loading raw genome (15M rows) ...")
    genome_scores = load_raw("genome_scores")
    genome_tags = load_raw("genome_tags")
    genome, rep = cleaning.clean_genome(genome_scores, genome_tags)
    reports.append(rep)

    # --- Referential integrity: keep only movies that appear in ratings ---
    # (genome covers a movie subset; unrated movies cannot be evaluated.)
    rated_movie_ids = set(ratings["movieId"].unique())
    n_unrated = int((~movies["movieId"].isin(rated_movie_ids)).sum())
    if n_unrated:
        log.info("Dropping %d movies with zero ratings (no evaluation possible).", n_unrated)
        movies = movies[movies["movieId"].isin(rated_movie_ids)].reset_index(drop=True)
        links = links[links["movieId"].isin(rated_movie_ids)].reset_index(drop=True)
        genome = genome[genome["movieId"].isin(rated_movie_ids)].reset_index(drop=True)

    # --- Feature validation: cross-table id consistency ---
    orphans_tags = int((~tags["movieId"].isin(movies["movieId"])).sum())
    if orphans_tags:
        log.warning("%d tags reference unknown movies (kept; genome-only movies may still be useful).", orphans_tags)

    # --- Outlier analysis (reported, not dropped: power users are real) ---
    outliers = cleaning.summarize_outliers(ratings)
    log.info("User activity outliers: %s", json.dumps(outliers))

    # --- Write parquet outputs ---
    outputs = {
        "movies": movies,
        "ratings": ratings,
        "tags": tags,
        "links": links,
        "genome_scores": genome,  # tidy joined version, thresholded below
        "genome_tags": genome_tags,
    }
    # Apply genome threshold to the processed copy (full matrix stays in raw).
    outputs["genome_scores"] = outputs["genome_scores"][
        outputs["genome_scores"]["relevance"] >= GENOME_RELEVANCE_THRESHOLD
    ].reset_index(drop=True)

    for name, df in outputs.items():
        path = PROCESSED_DATA_DIR / PROCESSED_FILES[name]
        df.to_parquet(path, index=False)
        log.info("Wrote %s (%d rows).", path.name, len(df))

    elapsed = time.perf_counter() - t0
    quality = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "tables": {
            r.table: {
                "rows_in": r.rows_in,
                "rows_out": r.rows_out,
                "duplicates_removed": r.duplicates_removed,
                "missing_dropped": r.missing_dropped,
                "issues": r.issues,
            }
            for r in reports
        },
        "user_activity_outliers": outliers,
        "genome_relevance_threshold": GENOME_RELEVANCE_THRESHOLD,
        "movies_dropped_no_ratings": n_unrated,
    }
    report_path = PROCESSED_DATA_DIR / "quality_report.json"
    report_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    log.info("Quality report -> %s (pipeline finished in %.1fs)", report_path, elapsed)
    return quality


if __name__ == "__main__":
    sys.exit(run_pipeline())
