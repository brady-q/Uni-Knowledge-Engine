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

from app.llm import DEFAULT_EFFORT, DEFAULT_MAX_TOKENS, Effort

# Named here as well as defaulted below so that the test which loads the real
# weights can assert against the model this deployment actually ships with,
# rather than hard-coding a second copy of the string that would go stale.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Same reason, for the test that calls the real API.
DEFAULT_ANSWER_MODEL = "claude-opus-5"


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

    # Ceiling on a single uploaded PDF. Enforced twice: once against the
    # Content-Length header so an oversized body is refused before it is read,
    # and again while streaming, because that header is a client's claim and
    # not a fact.
    max_upload_bytes: int = 50 * 1024 * 1024

    # Comma-separated rather than a list, because pydantic-settings parses
    # list-typed env vars as JSON, which is a hostile thing to put in a .env.
    cors_origins: str = "http://localhost:3000"

    # Sentence-Transformers model id. This one emits the 384 dimensions
    # chunks.embedding is declared with, so it is not freely swappable: a model
    # of a different width needs a migration and a re-embed of every chunk.
    # See models.EMBEDDING_DIMENSIONS, which the loader checks it against.
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    # How many chunks are pushed through the model at once. Larger is faster
    # per chunk and holds more memory; 32 is well within the api container's
    # footprint for a 384-dimension MiniLM.
    embedding_batch_size: int = Field(default=32, ge=1)

    # Start loading the model during startup rather than on the first request
    # that needs it. Turned off in tests, which must never download weights.
    embedding_preload: bool = True

    # --- answers ---
    #
    # Server-side only. Never reaches the web container (see docker-compose),
    # and repr=False so it cannot be leaked by logging a Settings object.
    # Absent is a supported state: the API starts, /health/ready says why
    # answering is unavailable, and asking a question is a 503.
    anthropic_api_key: str | None = Field(default=None, repr=False)

    # Model ids are not date-suffixed aliases here; the answer log records the
    # concrete version the API actually served, read off the response.
    answer_model: str = DEFAULT_ANSWER_MODEL

    # Ceiling on one answer, adaptive thinking included. See app.llm for why
    # these two defaults are what they are; both are here because they are the
    # cost and quality dials somebody running this will want to turn.
    answer_max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1024)
    answer_effort: Effort = DEFAULT_EFFORT

    # Per-attempt, and there is one retry, so an answer can take twice this
    # long in the worst case. Well under the SDK's ten-minute default, because
    # a browser is waiting.
    answer_timeout_seconds: float = Field(default=60.0, gt=0)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached accessor for process-wide settings."""
    return Settings()
