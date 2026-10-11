"""Environment configuration via pydantic-settings (reads .env, never commits it)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, populated from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    db_backend: str = "sqlite"  # "sqlite" | "postgres"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "personalai"
    db_user: str = "postgres"
    db_password: str = ""

    # --- Application ---
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def database_url(self) -> str:
        """SQLAlchemy URL for the configured backend."""
        if self.db_backend == "postgres":
            return (
                f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        return "sqlite:///personalai.db"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor — the .env file is read once per process."""
    return Settings()
