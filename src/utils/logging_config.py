"""Application-wide logging configuration."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # idempotent across repeated calls (notebooks, tests, API)
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call configure_logging() once at entry points."""
    return logging.getLogger(name)
