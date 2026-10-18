"""Fixtures for tests that need a real Postgres.

CRUD tests truncate tables between cases, and doing that to the database
someone is developing against would be hostile. So these fixtures create and
migrate a dedicated ``<database>_test`` alongside it, on demand.

Migrations are applied by shelling out to ``alembic upgrade head`` rather than
by importing Alembic, for two reasons: ``migrations/env.py`` calls
``asyncio.run`` at import time, which cannot happen inside pytest-asyncio's
running loop, and running the real command means the suite proves the
*migration* works, not merely that the models agree with themselves.

Everything here skips rather than fails when no server is reachable, so
`pytest` stays green on a laptop with Docker stopped.

What these tests do *not* use for real is either model. Vectors come from
``tests.embeddings.FakeEmbedder``, which is a real 384-dimension normalised
vector per text and therefore exercises pgvector, the HNSW opclass and the
retrieval SQL exactly as the model would - without a download, and with a
ranking the test can predict. Answers come from ``tests.answers``, which lets a
test choose the markers the model writes and therefore assert on what happens
to an invented one. What only the real models can do is matched in
``tests/slow``.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.db import create_engine, create_sessionmaker
from app.embedding import get_embedder
from app.llm import get_answer_model
from app.main import create_app
from tests.answers import FakeAnswerModel
from tests.embeddings import FakeEmbedder

type AlembicRunner = Callable[..., subprocess.CompletedProcess[str]]

# alembic.ini lives at the backend root; the subprocess needs that as its cwd.
BACKEND_DIR = Path(__file__).resolve().parents[2]

DEFAULT_URL = "postgresql+asyncpg://uke:uke_dev_password@localhost:5432/uke"

# Truncated between tests. Listed explicitly, in dependency order, so a CASCADE
# can never quietly reach a table this file does not know about.
MANAGED_TABLES = ("citations", "questions", "chunks", "documents", "courses")


def _configured_url() -> str:
    """The development database. In the api container DATABASE_URL already
    points at the db service; host-side runs set TEST_DATABASE_URL."""
    return os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL") or DEFAULT_URL


def _render(url: URL) -> str:
    # str() on a URL masks the password with ***, which would not connect.
    return url.render_as_string(hide_password=False)


async def _ping(url: str) -> None:
    engine = create_engine(url)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    finally:
        await engine.dispose()


async def _create_database_if_missing(admin_url: str, name: str) -> None:
    # CREATE DATABASE cannot run inside a transaction, hence AUTOCOMMIT, and it
    # cannot run from a connection to the database being created, hence the
    # detour through the "postgres" maintenance database.
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            exists = await connection.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": name}
            )
            if not exists:
                await connection.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        await engine.dispose()


def _alembic(url: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run an alembic subcommand against ``url``.

    ``migrations/env.py`` reads the URL from Settings, so pointing DATABASE_URL
    at the test database is all it takes to redirect the command. Invoked as
    ``python -m alembic`` so it resolves to the interpreter running the tests
    rather than whatever happens to be first on PATH.
    """
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=BACKEND_DIR,
        env={**os.environ, "DATABASE_URL": url},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="session")
def dev_database_url() -> str:
    """The development database itself, for tests that assert on the real thing."""
    url = _configured_url()
    try:
        asyncio.run(_ping(url))
    except Exception as exc:  # noqa: BLE001 - any failure means "no database here"
        pytest.skip(f"no database at {url}: {type(exc).__name__}")
    return url


@pytest.fixture(scope="session")
def isolated_database_url() -> str:
    """A migrated, disposable database next to the development one.

    Session-scoped and synchronous on purpose: it runs before pytest-asyncio
    opens a loop for the first test, so ``asyncio.run`` here is safe.
    """
    configured = make_url(_configured_url())
    name = f"{configured.database}_test"
    admin_url = _render(configured.set(database="postgres"))
    try:
        asyncio.run(_create_database_if_missing(admin_url, name))
    except Exception as exc:  # noqa: BLE001 - any failure means "no database here"
        pytest.skip(f"no server at {configured.host}:{configured.port}: {type(exc).__name__}")

    url = _render(configured.set(database=name))
    result = _alembic(url, "upgrade", "head")
    if result.returncode != 0:
        pytest.fail(f"alembic upgrade head failed:\n{result.stdout}\n{result.stderr}")
    return url


@pytest.fixture(scope="session")
def alembic(isolated_database_url: str) -> AlembicRunner:
    """Run alembic against the migrated test database, for schema assertions."""

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return _alembic(isolated_database_url, *args)

    return run


@pytest.fixture
async def isolated_engine(isolated_database_url: str) -> AsyncIterator[AsyncEngine]:
    """Function-scoped deliberately: the event loop is too.

    An asyncpg pool holds connections bound to the loop that opened them, so a
    longer-lived engine would hand a later test sockets attached to a loop that
    has already closed.
    """
    engine = create_engine(isolated_database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def clean_database(isolated_engine: AsyncEngine) -> None:
    """Empty the managed tables before the test runs.

    Truncating up front rather than cleaning up afterwards means a test that
    fails mid-way leaves its rows behind for inspection without breaking the
    next one.
    """
    async with isolated_engine.begin() as connection:
        await connection.execute(text(f"TRUNCATE {', '.join(MANAGED_TABLES)} CASCADE"))


@pytest.fixture
async def db_session(
    isolated_engine: AsyncEngine, clean_database: None
) -> AsyncIterator[AsyncSession]:
    """A session on the test database, for asserting on rows the API wrote."""
    async with create_sessionmaker(isolated_engine)() as session:
        yield session


@pytest.fixture
def isolated_settings(isolated_database_url: str, tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        log_level="WARNING",
        database_url=isolated_database_url,
        storage_dir=tmp_path / "storage",
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    """The fake the app is wired to, exposed so a test can assert on its calls."""
    return FakeEmbedder()


@pytest.fixture
def answer_model() -> FakeAnswerModel:
    """The same trade as ``embedder``, for the half of the pipeline that bills.

    Exposed so a test can set the markers the "model" writes, which is how the
    interesting cases - an invented citation, an abstention - are provoked
    deterministically rather than hoped for.
    """
    return FakeAnswerModel()


@pytest.fixture
async def api_client(
    isolated_settings: Settings,
    clean_database: None,
    embedder: FakeEmbedder,
    answer_model: FakeAnswerModel,
) -> AsyncIterator[AsyncClient]:
    """The real app, real sessions, real Postgres, real migrations.

    The two models are the only overrides. The embedder covers retrieval *and*
    ingestion, because the pipeline receives whatever ``get_embedder`` resolved
    rather than reading the model off ``app.state`` itself - which is exactly
    why that dependency exists. The answer model is overridden for a blunter
    reason: everything below this line is free to run, and a real one is not.
    """
    app = create_app(isolated_settings)
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_answer_model] = lambda: answer_model
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
