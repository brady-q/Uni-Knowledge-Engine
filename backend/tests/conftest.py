"""Shared fixtures.

The default test app points at a database that cannot exist. That is not
laziness: it guarantees no unit test quietly depends on a live Postgres, and it
makes the readiness failure path testable without any infrastructure. Tests
that want a healthy database override the readiness dependency instead.

The same idea covers the embedding model, via the autouse guard below: a suite
that downloads 90 MB of weights on a cold cache is a suite that fails on a
train. The embedder object is still constructed by the lifespan - it just never
loads, because nothing in the default suite reaches the real one. Tests that
need vectors override ``get_embedder`` with ``tests.embeddings.FakeEmbedder``;
the real model is exercised only by ``tests/slow``, which constructs its own.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app
from app.routers.health import get_readiness_report
from app.services.health import ComponentHealth, ReadinessReport
from tests.answers import FAKE_MODEL_NAME

# Port 1 is never listening, so connection failure is immediate rather than a
# timeout that would slow the suite down.
UNREACHABLE_DATABASE_URL = "postgresql+asyncpg://test:test@127.0.0.1:1/uke_test"


@pytest.fixture(scope="session", autouse=True)
def never_preload_the_model() -> Iterator[None]:
    """Stop any app built anywhere in the suite from loading real weights.

    Set in the environment rather than passed to each fixture because that is
    the only place every ``Settings`` in the suite reads from - including the
    ones tests construct for themselves. A per-fixture flag is a rule someone
    has to remember; this one cannot be forgotten, only overridden on purpose.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("EMBEDDING_PRELOAD", "false")
        yield


@pytest.fixture(autouse=True)
def never_call_the_real_api(request: pytest.FixtureRequest) -> Iterator[None]:
    """Stop any app built anywhere in the suite from having a usable API key.

    The embedding guard above protects a cold cache and a train journey; this
    one protects a credit card. Every test that needs an answer substitutes
    ``tests.answers.FakeAnswerModel`` for ``get_answer_model``, but an override
    that is forgotten would otherwise fall through to a real, billed call - so
    the key is removed from the environment instead of trusted to be unused,
    and a missing key makes such a call raise before any HTTP happens.

    Function-scoped, and skipped for tests marked ``live``: those exist
    precisely to call the real API, are deselected by the default ``addopts``,
    and are the one place the key must survive.
    """
    if request.node.get_closest_marker("live"):
        yield
        return
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("ANTHROPIC_API_KEY", raising=False)
        yield


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
            "embedder": ComponentHealth("ok", "loaded fake-bag-of-words"),
            "claude": ComponentHealth("ok", f"configured for {FAKE_MODEL_NAME}"),
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
