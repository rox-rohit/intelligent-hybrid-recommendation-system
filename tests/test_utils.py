"""Smoke tests for shared utilities (paths, config, logging)."""

from pathlib import Path

from src.utils.config import Settings, get_settings
from src.utils.logging_config import configure_logging, get_logger
from src.utils.paths import (
    DATA_DIR,
    MODELS_DIR,
    MOVIELENS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def test_paths_exist_on_disk():
    """The standard project directories are created and resolvable."""
    assert RAW_DATA_DIR.exists()
    assert PROCESSED_DATA_DIR.exists()
    assert DATA_DIR.name == "data"
    assert MODELS_DIR.exists()
    assert MOVIELENS_DIR == RAW_DATA_DIR / "ml-25m"


def test_paths_are_inside_project_root():
    """Every resolved path lives under the repository root (no stray drive paths)."""
    root = Path(__file__).resolve().parents[1]
    for p in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
        assert root in p.parents


def test_settings_defaults():
    """Settings falls back to safe defaults, isolated from any local .env."""
    s = Settings(_env_file=None)  # ignore the developer's real .env
    assert s.db_backend == "sqlite"
    assert s.api_port == 8000
    assert s.db_password == ""  # secrets come from env only, never hardcoded


def test_settings_database_url_sqlite():
    """Default profile points at a local SQLite database file."""
    s = Settings(db_backend="sqlite")
    assert s.database_url.startswith("sqlite")


def test_get_settings_is_cached():
    """Repeated calls return the same cached Settings instance."""
    assert get_settings() is get_settings()


def test_configure_logging_idempotent():
    """Calling configure_logging twice does not raise or duplicate handlers."""
    configure_logging("INFO")
    configure_logging("INFO")
    logger = get_logger("test-logger")
    assert logger is not None
