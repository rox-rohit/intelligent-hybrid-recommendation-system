# PersonalAI — Intelligent Hybrid Recommendation System

> **Status:** Milestone 0 — project scaffolding. This README grows with the project.

A production-style personalized movie recommendation platform answering:
*"What movies is this particular user most likely to enjoy, and why?"*

Built end-to-end: data pipeline → content-based + collaborative models → hybrid
ranking with learned weights → diversity, exploration, and explainability →
FastAPI + PostgreSQL backend → React frontend → evaluation on real metrics.

## Repository Layout

```
personalai/
├── data/           # raw (immutable) / processed / external — see data/README.md
├── notebooks/      # EDA & experiments (logic lives in src/)
├── src/            # the ML engine: data, features, models, ranking, evaluation
├── app/            # FastAPI backend + React frontend
├── models/         # versioned model artifacts (model_v1, model_v2, ...)
├── tests/          # unit tests for critical components
├── scripts/        # data download, TMDB fetch, training entry points
└── docs/           # architecture and experiment documentation
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
python scripts/download_data.py   # MovieLens 25M (~250 MB, checksum-verified)
pytest                            # smoke tests
```

## Data Sources

- **MovieLens 25M** (GroupLens) — ratings, tags, timestamps. Educational/research use.
- **TMDB API** — cast, director, overview, keywords, posters. Cited per API terms.

Full licensing details: [`data/README.md`](data/README.md).

---
Detailed architecture, algorithms, evaluation results, and screenshots will be
documented here as milestones complete.
