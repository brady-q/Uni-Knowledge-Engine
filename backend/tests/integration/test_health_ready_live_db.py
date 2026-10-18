"""Readiness against a real Postgres+pgvector.

The stubbed unit tests prove the endpoint's contract; this proves the SQL is
valid and that the database we actually ship really does have pgvector. It
skips (rather than fails) when no database is reachable, so `pytest` stays
green on a laptop with Docker stopped.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.config import Settings
from app.db import create_engine
from app.main import create_app

pytestmark = pytest.mark.integration

DEFAULT_URL = "postgresql+asyncpg://uke:uke_dev_password@localhost:5432/uke"


def database_url() -> str:
    # In the api container DATABASE_URL already points at the db service;
    # host-side runs set TEST_DATABASE_URL to a localhost DSN.
    return os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL") or DEFAULT_URL


@pytest.fixture
async def live_settings(tmp_path: Path) -> Settings:
    url = database_url()
    engine = create_engine(url)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - any failure means "no database here"
        pytest.skip(f"no database at {url}: {type(exc).__name__}")
    finally:
        await engine.dispose()

    return Settings(
        environment="test",
        log_level="WARNING",
        database_url=url,
        storage_dir=tmp_path / "storage",
    )


@pytest.fixture
async def live_client(live_settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(live_settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_readiness_is_ok_against_the_real_database(live_client: AsyncClient) -> None:
    response = await live_client.get("/health/ready")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["db"]["status"] == "ok"


async def test_pgvector_is_present_on_the_shipped_image(live_client: AsyncClient) -> None:
    """A plain `postgres` image would pass the db check and fail here."""
    body = (await live_client.get("/health/ready")).json()

    check = body["checks"]["pgvector"]
    assert check["status"] == "ok", check
    assert check["detail"] is not None


async def test_lifespan_creates_the_storage_directory(live_settings: Settings) -> None:
    """Uploaded PDFs land here, so startup must guarantee the directory exists."""
    assert not live_settings.storage_dir.exists()

    app = create_app(live_settings)
    async with app.router.lifespan_context(app):
        assert live_settings.storage_dir.is_dir()
