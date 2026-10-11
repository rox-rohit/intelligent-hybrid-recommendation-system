"""Run the raw -> processed data pipeline. Thin wrapper over src.data.pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline import run_pipeline  # noqa: E402

if __name__ == "__main__":
    run_pipeline()
