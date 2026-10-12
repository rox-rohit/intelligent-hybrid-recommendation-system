"""Train the content-based model on real processed data and smoke-test it.

Builds the movie feature matrix from the genome + genres, then prints
similar-movie results for a few famous movies so correctness is judged
on domain knowledge (does Interstellar really surface Nolan sci-fi?).

Also times the build — it must be a precompute step, not per-request.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_processed
from src.models.content_based import ContentBasedRecommender
from src.utils.logging_config import configure_logging, get_logger

log = get_logger(__name__)

# Famous MovieLens movieIds for the smoke test (from movies.csv):
#   1   = Toy Story (1995)
#   593 = Silence of the Lambs (1991)
#   912 = Sunset Blvd. (1950)
#   356 = Forrest Gump (1994)
#   79132 = Inception (2010)
#   109487 = Interstellar (2014)
SMOKE_IDS = [1, 593, 912, 356, 79132, 109487]


def main() -> int:
    configure_logging("INFO")

    log.info("Loading processed tables ...")
    t0 = time.perf_counter()
    movies = load_processed("movies")
    genome = load_processed("genome_scores")
    genome_tags = load_processed("genome_tags")
    log.info("Loaded in %.1fs (%d movies, %d genome rows)",
             time.perf_counter() - t0, len(movies), len(genome))

    log.info("Building content recommender ...")
    t0 = time.perf_counter()
    rec = ContentBasedRecommender.from_processed_data(genome, movies, genome_tags)
    build_s = time.perf_counter() - t0
    log.info("Feature matrix + recommender built in %.1fs", build_s)

    print("\n=== Content-based smoke test (real data) ===\n")
    for mid in SMOKE_IDS:
        try:
            recs = rec.recommend_similar_movies(mid, k=5)
        except KeyError:
            print(f"movieId {mid}: not in processed catalog, skipping")
            continue
        title = rec._titles.get(mid, f"movieId {mid}")
        print(f"\nBecause you liked: {title} (id={mid})")
        for r in recs:
            print(f"  {r.similarity:5.1%}  {r.title}  — {rec.explain_text(r)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
