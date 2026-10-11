"""Shared pytest fixtures: project root on sys.path, data dirs, quiet logging."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _ensure_data_dirs():
    paths.ensure_dirs()
    yield
