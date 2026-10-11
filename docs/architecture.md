# PersonalAI — System Architecture

*(Milestone 0 snapshot; updated as components are built.)*

## High-Level Flow

```
User ──► Interactions (rate / like / watch / click / skip / watchlist)
        │
        ▼
  User Profile  ◄───── updates on every interaction
        │
        ▼
  Feature Engineering
        │
        ├─► Content-Based Model        (TF-IDF / embeddings over metadata)
        ├─► Collaborative Filtering    (item-kNN, implicit-feedback ALS)
        ├─► User Preference Model      (genre/director/era affinities)
        └─► Popularity / Freshness     (Bayesian-smoothed, recency-decayed)
        │
        ▼
  Candidate Generation      (~100–200 items)
        │
        ▼
  Ranking Model            (gradient-boosted, time-aware training)
        │
        ▼
  Diversity + Exploration  (MMR; personalized vs explore mode)
        │
        ▼
  Top-10 Recommendations + Explanations
        │
        ▼
  User ──► New Feedback ──► (loop)
```

## Components & Milestones

| # | Milestone | Primary deliverable | Status |
|---|---|---|---|
| 0 | Scaffolding | repo, deps, data download, licensing docs | ✅ |
| 1 | Data pipeline + EDA | `src/data/`, notebook 01 | ⬜ |
| 2 | Content-based | `recommend_similar_movies()` + explanations | ⬜ |
| 3 | Collaborative | item-kNN + MF comparison | ⬜ |
| 4 | Hybrid + eval harness | tuned weighted blend; @K metrics | ⬜ |
| 5 | Ranking model | LightGBM/HGB pointwise ranker, time-aware split | ⬜ |
| 6 | Profiles + implicit feedback | dynamic profiles, interaction weighting | ⬜ |
| 7 | Cold start / diversity / explore | onboarding + MMR + ε-greedy | ⬜ |
| 8 | Explainability | feature-attributed "why this?" | ⬜ |
| 9 | FastAPI + DB | endpoints, SQLAlchemy models, logging | ⬜ |
| 10 | Frontend | React + Vite SPA | ⬜ |
| 11 | Tests + Docker | pytest suite, Dockerfile, compose | ⬜ |
| 12 | Documentation | full README, experiment history | ⬜ |

## Design Decisions (locked during proposal)

- **Runtime:** Python 3.12 (system 3.15 lacks prebuilt wheels for the ML
  stack — pandas/pyarrow/implicit/lightgbm — verified during Milestone 0;
  3.12 is installed alongside 3.15 for this project's venv).
- **Dataset:** MovieLens 25M (GroupLens), checksum-verified download.
- **Content metadata:** **MovieLens genome tags** — 1,128 curated content tags
  with per-movie relevance scores (plus genres and user tags). TMDB was
  rejected: it is unreachable from the developer's region. The genome is a
  fully-licensed, purpose-built content representation — no scraping.
  Explanations cite matched genome tags ("shares *sci-fi*, *space travel*,
  *thought-provoking*..."), genres, and collaborative evidence.
- **DB:** SQLAlchemy ORM — SQLite (dev) / PostgreSQL (Docker, production profile).
- **CF:** item-based kNN + implicit-feedback ALS (Hu/Koren/Volinsky confidence
  weighting); user-based kNN documented as infeasible at 162k users.
- **Ranking:** pointwise gradient boosting on (user, movie) features;
  HistGradientBoosting is the guaranteed-available fallback for LightGBM.
- **Diversity:** MMR with genre-similarity penalty; report relevance/ILD tradeoff.
- **Exploration:** ε-greedy + novelty scoring; two modes (personalized / explore).
- **Explanations:** derived from real model components (matched genome tags,
  profile affinities, CF neighbor evidence) — never templated fiction.
- **Implicit feedback:** MovieLens contains only explicit ratings; interaction
  types (view/click/skip/...) are simulated from realistic patterns and
  **documented as such** — no fabricated "real" behavior claims.

## Performance Strategy

- Precomputed content vectors and CF factors; no per-request retraining.
- Sparse (CSR) interaction matrices; float32/int32 throughout.
- Candidate generation narrows 62k movies → ~200 before feature generation.
- Caching layer for API responses (to be added in Milestone 9).
