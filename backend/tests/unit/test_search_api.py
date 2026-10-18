"""HTTP behaviour of the search route, with no database and no model.

The retrieval service is replaced with one that returns a fixed ranking, so
what is under test is strictly the router's job: parameter validation, the
flattening of a match into a citation-shaped result, and the mapping of a
missing course onto a 404 and an unusable model onto a 503.

Whether the ranking itself is any good is a question for
``tests/integration/test_search_retrieval.py`` (real pgvector, fake vectors)
and ``tests/slow/test_real_embedding_model.py`` (real vectors).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.errors import EmbeddingError, NotFoundError
from app.models import Chunk, Document, DocumentStatus
from app.repositories.chunks import ChunkMatch
from app.routers.search import get_retrieval_service
from app.schemas import MAX_SEARCH_LIMIT, QUERY_MAX_LENGTH

COURSE_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


def make_match(
    *,
    ordinal: int = 0,
    page_start: int = 3,
    page_end: int = 4,
    text: str = "Gradient descent adjusts parameters.",
    distance: float = 0.25,
    filename: str = "lecture.pdf",
) -> ChunkMatch:
    """A detached match, as the repository would return it."""
    document = Document(
        id=uuid.uuid4(),
        course_id=COURSE_ID,
        filename=filename,
        content_hash="0" * 64,
        byte_size=1024,
        page_count=12,
        status=DocumentStatus.READY,
    )
    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=document.id,
        ordinal=ordinal,
        page_start=page_start,
        page_end=page_end,
        text=text,
        embedding=[0.0] * 384,
    )
    return ChunkMatch(chunk=chunk, document=document, distance=distance)


class FakeRetrievalService:
    """Same surface as RetrievalService, with a canned ranking."""

    def __init__(self) -> None:
        self.matches: list[ChunkMatch] = []
        self.raises: Exception | None = None
        self.calls: list[tuple[uuid.UUID, str, int]] = []

    async def search(self, course_id: uuid.UUID, query: str, *, limit: int) -> list[ChunkMatch]:
        self.calls.append((course_id, query, limit))
        if self.raises is not None:
            raise self.raises
        return self.matches[:limit]


@pytest.fixture
def service() -> FakeRetrievalService:
    return FakeRetrievalService()


@pytest.fixture
async def client(app: FastAPI, service: FakeRetrievalService) -> AsyncIterator[AsyncClient]:
    app.dependency_overrides[get_retrieval_service] = lambda: service
    try:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as http:
                yield http
    finally:
        app.dependency_overrides.clear()


def search_url(course_id: uuid.UUID = COURSE_ID) -> str:
    return f"/api/v1/courses/{course_id}/search"


# --- results ------------------------------------------------------------------


async def test_a_match_carries_everything_a_citation_needs(
    client: AsyncClient, service: FakeRetrievalService
) -> None:
    match = make_match(ordinal=7, page_start=11, page_end=12, filename="week3.pdf")
    service.matches = [match]

    response = await client.get(search_url(), params={"q": "gradient descent"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["query"] == "gradient descent"
    assert body["items"] == [
        {
            "chunk_id": str(match.chunk.id),
            "document_id": str(match.document.id),
            "filename": "week3.pdf",
            "ordinal": 7,
            "page_start": 11,
            "page_end": 12,
            "text": match.chunk.text,
            # 1 - distance: a score that improves as it rises.
            "score": 0.75,
        }
    ]


async def test_the_ranking_is_passed_through_untouched(
    client: AsyncClient, service: FakeRetrievalService
) -> None:
    """Ordering is the database's, not the router's; re-sorting here would hide
    a broken ORDER BY behind a correct-looking response."""
    service.matches = [
        make_match(ordinal=0, distance=0.1),
        make_match(ordinal=1, distance=0.4),
        make_match(ordinal=2, distance=0.9),
    ]

    body = (await client.get(search_url(), params={"q": "loss"})).json()

    assert [item["ordinal"] for item in body["items"]] == [0, 1, 2]
    assert [item["score"] for item in body["items"]] == pytest.approx([0.9, 0.6, 0.1])


async def test_no_matches_is_an_empty_200(client: AsyncClient) -> None:
    """A course with nothing to say about the query is not an error."""
    response = await client.get(search_url(), params={"q": "quantum chromodynamics"})

    assert response.status_code == 200
    assert response.json()["items"] == []


async def test_the_embedding_is_never_exposed(
    client: AsyncClient, service: FakeRetrievalService
) -> None:
    """384 floats per result is not part of the public contract."""
    service.matches = [make_match()]

    body = (await client.get(search_url(), params={"q": "loss"})).json()

    assert "embedding" not in body["items"][0]


# --- the query ----------------------------------------------------------------


async def test_the_query_is_trimmed_before_it_is_embedded(
    client: AsyncClient, service: FakeRetrievalService
) -> None:
    """Leading whitespace is not a search term, and it is not free either: it
    is tokens the model spends on nothing."""
    response = await client.get(search_url(), params={"q": "  learning rate  "})

    assert response.json()["query"] == "learning rate"
    assert service.calls[0][1] == "learning rate"


async def test_a_blank_query_is_422(client: AsyncClient, service: FakeRetrievalService) -> None:
    """Nearest neighbours of nothing in particular is not a meaningful result."""
    for blank in ("", "   ", "\t\n"):
        assert (await client.get(search_url(), params={"q": blank})).status_code == 422

    assert service.calls == []


async def test_a_missing_query_is_422(client: AsyncClient) -> None:
    assert (await client.get(search_url())).status_code == 422


async def test_an_overlong_query_is_422(client: AsyncClient) -> None:
    """Past the model's window the text is silently truncated, so the limit is
    stated rather than left to the tokenizer."""
    response = await client.get(search_url(), params={"q": "x" * (QUERY_MAX_LENGTH + 1)})

    assert response.status_code == 422


async def test_a_query_at_the_limit_is_accepted(client: AsyncClient) -> None:
    response = await client.get(search_url(), params={"q": "x" * QUERY_MAX_LENGTH})

    assert response.status_code == 200


# --- limits and scoping -------------------------------------------------------


async def test_the_default_limit_is_the_answer_prompt_size(
    client: AsyncClient, service: FakeRetrievalService
) -> None:
    await client.get(search_url(), params={"q": "loss"})

    assert service.calls[0][2] == 6


async def test_the_limit_is_passed_through(
    client: AsyncClient, service: FakeRetrievalService
) -> None:
    await client.get(search_url(), params={"q": "loss", "limit": 3})

    assert service.calls[0][2] == 3


async def test_out_of_range_limits_are_422(client: AsyncClient) -> None:
    for limit in (0, -1, MAX_SEARCH_LIMIT + 1):
        response = await client.get(search_url(), params={"q": "loss", "limit": limit})
        assert response.status_code == 422, limit


async def test_a_malformed_course_id_is_422_not_404(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/courses/not-a-uuid/search?q=loss")).status_code == 422


async def test_searching_a_missing_course_is_404(
    client: AsyncClient, service: FakeRetrievalService
) -> None:
    """Distinct from an empty result: one is a stale id, the other an answer."""
    service.raises = NotFoundError(f"no course with id {COURSE_ID}")

    response = await client.get(search_url(), params={"q": "loss"})

    assert response.status_code == 404
    assert str(COURSE_ID) in response.json()["detail"]


async def test_an_unusable_model_is_503(client: AsyncClient, service: FakeRetrievalService) -> None:
    """Still loading, or a failed download. Neither is the client's fault and
    neither is permanent, so it is not a 500 and not a 4xx."""
    service.raises = EmbeddingError("the embedding model is not available yet; try again shortly")

    response = await client.get(search_url(), params={"q": "loss"})

    assert response.status_code == 503
    assert "try again" in response.json()["detail"]
