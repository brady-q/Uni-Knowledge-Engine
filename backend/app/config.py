"""Environment-driven settings.

Every knob the app reads is declared here, so there is exactly one place to
look for "what does this deployment need?". Settings are constructed once and
attached to ``app.state`` rather than imported ad hoc, which keeps tests free
to build an app with different settings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # No default: an API that silently starts against the wrong database is
    # worse than one that refuses to start.
    database_url: str

    storage_dir: Path = Path("/data/storage")

    # Comma-separated rather than a list, because pydantic-settings parses
    # list-typed env vars as JSON, which is a hostile thing to put in a .env.
    cors_origins: str = "http://localhost:3000"

    # Server-side only, and unused until answer generation exists. Declared
    # now so secret handling is established before there is a secret to leak.
    anthropic_api_key: str | None = Field(default=None, repr=False)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached accessor for process-wide settings."""
    return Settings()
