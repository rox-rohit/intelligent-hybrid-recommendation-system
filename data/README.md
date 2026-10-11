# Data Directory

## Layout

| Folder | Contents | Committed? |
|---|---|---|
| `raw/` | Immutable source data as downloaded — never modified in place | No (license + size) |
| `processed/` | Cleaned, pipeline-produced datasets | No (regenerable) |
| `external/` | Secondary sources (if any are added later) | No |

Reproduce everything with:
```
python scripts/download_data.py
```

## Sources & Licensing

### MovieLens 25M (primary — the only data source)

- **Source:** GroupLens Research, University of Minnesota — https://grouplens.org/datasets/movielens/
- **Download:** https://files.grouplens.org/datasets/movielens/ml-25m.zip
- **Contents:**
  - `movies.csv` — 62,423 movies: title (with release year), genre list
  - `ratings.csv` — 25,000,095 ratings (0.5–5.0) from 162,541 users, with timestamps
  - `tags.csv` — 1,093,360 user-applied content tags
  - `genome-scores.csv` + `genome-tags.csv` — the **movie genome**:
    relevance scores for 1,128 curated content tags per movie (e.g.
    "sci-fi", "space travel", "thought-provoking", "christopher nolan").
    This is the content representation used by the content-based model.
  - `links.csv` — MovieLens ↔ IMDb ↔ TMDB id mapping (kept for reference/future use)
- **License (verbatim, from the dataset README):**
  > "The dataset may only be used for educational and research purposes.
  > Any commercial use is prohibited. It is distributed without any warranty.
  > Cite: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
  > History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19."

  This project is an educational/portfolio system, which satisfies the
  non-commercial research use requirement.
- **Integrity:** the download script (`scripts/download_data.py`) verifies the
  archive MD5 before extraction.

### Why no TMDB / external metadata

TMDB is unreachable from the developer's location (region-blocked), so the
project deliberately uses **only MovieLens-provided content data**: genres,
user tags, and the genome tag-relevance matrix. The genome was built by
GroupLens from collaborative signals and human raters and is a strong,
fully-licensed content representation — no scraping, no murky provenance.
If richer metadata (cast/director) is added later, this section and
`docs/architecture.md` will document the source and licensing first.

## Data Hygiene Rules

1. **Raw data is immutable.** The pipeline reads from `raw/` and writes to
   `processed/`; nothing under `raw/` is ever edited, filtered, or overwritten.
2. **No fabricated data.** All statistics, metrics, and charts derive from the
   source above. Where interaction types (view, click, skip, ...) are simulated
   for demonstration of the implicit-feedback pipeline, the simulation is
   explicitly documented in the code and README.
3. **Regenerable outputs only.** Anything under `processed/` can be deleted and
   rebuilt by the pipeline scripts; only code and documentation are committed.
