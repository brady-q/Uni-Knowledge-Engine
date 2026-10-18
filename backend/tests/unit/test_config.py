"""Settings parsing that has real logic worth testing."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings

DSN = "postgresql+asyncpg://u:p@db:5432/uke"


def test_cors_origins_are_split_and_trimmed() -> None:
    settings = Settings(
        database_url=DSN,
        cors_origins="http://localhost:3000, https://example.com ",
    )

    assert settings.cors_origin_list == ["http://localhost:3000", "https://example.com"]


def test_empty_cors_origins_disables_the_middleware() -> None:
    assert Settings(database_url=DSN, cors_origins="").cors_origin_list == []


def test_database_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast rather than start against an unintended default database.

    DATABASE_URL is cleared explicitly: the api container exports it, and this
    test asserts what happens when nothing supplies one.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_api_key_is_hidden_from_reprs() -> None:
    """Guards against a secret landing in a log line or traceback."""
    settings = Settings(database_url=DSN, anthropic_api_key="sk-ant-secret")

    assert "sk-ant-secret" not in repr(settings)


def test_storage_dir_is_a_path() -> None:
    settings = Settings(database_url=DSN, storage_dir=Path("/data/storage"))

    assert settings.storage_dir == Path("/data/storage")
