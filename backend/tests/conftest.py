"""Shared fixtures.

The default test app points at a database that cannot exist. That is not
laziness: it guarantees no unit test quietly depends on a live Postgres, and it
makes the readiness failure path testable without any infrastructure. Tests
that want a healthy database override the readiness dependency instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app
from app.routers.health import get_readiness_report
from app.services.health import ComponentHealth, ReadinessReport

# Port 1 is never listening, so connection failure is immediate rather than a
# timeout that would slow the suite down.
UNREACHABLE_DATABASE_URL = "postgresql+asyncpg://test:test@127.0.0.1:1/uke_test"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        log_level="WARNING",
        database_url=UNREACHABLE_DATABASE_URL,
        storage_dir=tmp_path / "storage",
        cors_origins="http://localhost:3000",
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def healthy_report() -> ReadinessReport:
    return ReadinessReport(
        checks={
            "db": ComponentHealth("ok"),
            "pgvector": ComponentHealth("ok", "installed 0.8.1"),
        }
    )


async def _asgi_client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Drive the app in-process, running its lifespan (ASGITransport does not)."""
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
async def client(app: FastAPI, healthy_report: ReadinessReport) -> AsyncIterator[AsyncClient]:
    """Client whose dependencies report a healthy system."""
    app.dependency_overrides[get_readiness_report] = lambda: healthy_report
    try:
        async for c in _asgi_client(app):
            yield c
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
async def unmocked_client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Client with the real probes, against the unreachable database."""
    async for c in _asgi_client(app):
        yield c
