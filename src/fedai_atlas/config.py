"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """FedAI Atlas configuration. Loads from .env file and environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API keys
    anthropic_api_key: str
    voyage_api_key: str = ""

    # Model versions — pinned for reproducible evals
    model_opus: str = "claude-opus-4-7"
    model_sonnet: str = "claude-sonnet-4-6"
    model_haiku: str = "claude-haiku-4-5-20251001"

    # Embedding model
    embedding_model: str = "voyage-3"

    # Data paths
    data_dir: Path = Path("data")
    vector_db_path: Path = Path("data/lancedb")
    sqlite_path: Path = Path("data/fedai.db")


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
