"""Readiness against a real Postgres+pgvector.

The stubbed unit tests prove the endpoint's contract; this proves the SQL is
valid and that the database we actually ship really does have pgvector. It
skips (rather than fails) when no database is reachable, so `pytest` stays
green on a laptop with Docker stopped.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.embedding import get_embedder
from app.llm import get_answer_model
from app.main import create_app
from tests.answers import FakeAnswerModel
from tests.embeddings import FakeEmbedder

pytestmark = pytest.mark.integration


@pytest.fixture
def live_settings(dev_database_url: str, tmp_path: Path) -> Settings:
    """Points at the development database, not the isolated test one: these
    tests exist to prove the database we actually ship has what we need."""
    return Settings(
        environment="test",
        log_level="WARNING",
        database_url=dev_database_url,
        storage_dir=tmp_path / "storage",
    )


@pytest.fixture
async def live_client(live_settings: Settings) -> AsyncIterator[AsyncClient]:
    """Real database, fake models.

    The point of these tests is what the shipped Postgres image provides, and
    the readiness report is only "ok" if *every* check is - so without
    substitutes, the aggregate assertion below would be measuring a 90 MB
    download and the presence of somebody's API key rather than the database.
    """
    app = create_app(live_settings)
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_answer_model] = lambda: FakeAnswerModel()
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
    # Every check has to be present for the aggregate to mean anything.
    assert set(body["checks"]) == {"db", "pgvector", "embedder", "claude"}


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
