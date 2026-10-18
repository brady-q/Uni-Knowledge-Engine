"""HTTP behaviour of the question routes, with no database and no model.

The answer service is replaced with one that returns a fixed result, so what is
under test is strictly the router's job: validating the body, flattening a
resolved citation into the shape a client renders, setting the ``Location`` a
``201`` is required to carry, and mapping each way this can fail onto the right
status - including the two that are new here, ``502`` for a model that replied
with nothing usable and ``409`` for a corpus that moved mid-answer.

Whether the answers are any good is a question for
``tests/integration/test_answers.py`` (real Postgres, real migrations, fake
model) and ``tests/slow/test_real_claude.py`` (the real thing).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.errors import (
    ConflictError,
    EmbeddingError,
    GenerationError,
    NotFoundError,
    UnusableAnswerError,
)
from app.models import Chunk, Citation, Document, DocumentStatus, Question
from app.routers.answers import get_answer_service
from app.schemas import DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT, QUERY_MAX_LENGTH
from app.services.answers import Answer, ResolvedCitation

COURSE_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
QUESTION_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


def make_citation(
    *,
    marker: int,
    filename: str = "week3.pdf",
    page_start: int = 11,
    page_end: int = 12,
    text: str = "Gradient descent adjusts parameters.",
) -> ResolvedCitation:
    document = Document(
        id=uuid.uuid4(),
        course_id=COURSE_ID,
        filename=filename,
        content_hash="0" * 64,
        byte_size=1024,
        page_count=40,
        status=DocumentStatus.READY,
    )
    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=document.id,
        ordinal=marker,
        page_start=page_start,
        page_end=page_end,
        text=text,
    )
    return ResolvedCitation(marker=marker, chunk=chunk, document=document)


def make_answer(
    *,
    question: str = "What is gradient descent?",
    answer: str = "It steps downhill [S1].",
    answered: bool = True,
    citations: list[ResolvedCitation] | None = None,
    retrieved: int = 6,
    dropped_markers: int = 0,
    model: str | None = "claude-opus-5-20260101",
) -> Answer:
    """A detached result, as the service would return it."""
    resolved = [make_citation(marker=1)] if citations is None else citations
    row = Question(
        id=QUESTION_ID,
        course_id=COURSE_ID,
        question=question,
        answer=answer,
        answered=answered,
        retrieved=retrieved,
        dropped_markers=dropped_markers,
        model=model,
        input_tokens=1_234 if model else None,
        output_tokens=56 if model else None,
        latency_ms=2_100,
        created_at=datetime(2026, 9, 5, 12, 0, tzinfo=UTC),
        # Set so the ORM object is a faithful stand-in even though the router
        # reads citations off the Answer rather than through the relationship.
        citations=[Citation(marker=citation.marker, chunk=citation.chunk) for citation in resolved],
    )
    return Answer(question=row, citations=resolved)


class FakeAnswerService:
    """Same surface as AnswerService, with a canned result."""

    def __init__(self) -> None:
        self.result = make_answer()
        self.page: list[Answer] = []
        self.total = 0
        self.raises: Exception | None = None
        self.asked: list[tuple[uuid.UUID, str, int]] = []
        self.listed: list[tuple[uuid.UUID, int, int]] = []

    async def ask(self, course_id: uuid.UUID, question: str, *, limit: int) -> Answer:
        self.asked.append((course_id, question, limit))
        if self.raises is not None:
            raise self.raises
        return self.result

    async def get(self, course_id: uuid.UUID, question_id: uuid.UUID) -> Answer:
        if self.raises is not None:
            raise self.raises
        return self.result

    async def list(
        self, course_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[Answer], int]:
        self.listed.append((course_id, limit, offset))
        if self.raises is not None:
            raise self.raises
        return self.page, self.total


@pytest.fixture
def service() -> FakeAnswerService:
    return FakeAnswerService()


@pytest.fixture
async def client(app: FastAPI, service: FakeAnswerService) -> AsyncIterator[AsyncClient]:
    app.dependency_overrides[get_answer_service] = lambda: service
    try:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as http:
                yield http
    finally:
        app.dependency_overrides.clear()


def questions_url(course_id: uuid.UUID = COURSE_ID) -> str:
    return f"/api/v1/courses/{course_id}/questions"


# --- the answer ---------------------------------------------------------------


async def test_asking_returns_the_answer_and_its_citations(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    citation = make_citation(marker=2, filename="week5.pdf", page_start=7, page_end=8)
    service.result = make_answer(answer="Due in week six [S2].", citations=[citation])

    response = await client.post(questions_url(), json={"question": "When is it due?"})

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["answer"] == "Due in week six [S2]."
    assert body["answered"] is True
    assert body["citations"] == [
        {
            # The label as it appears in the answer, so a client renders
            # footnotes by matching "[S2]" in the prose against this.
            "marker": "S2",
            "chunk_id": str(citation.chunk.id),
            "document_id": str(citation.document.id),
            "filename": "week5.pdf",
            "page_start": 7,
            "page_end": 8,
            "text": citation.chunk.text,
        }
    ]


async def test_a_created_answer_says_where_to_fetch_it_again(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    """What a 201 is required to carry, and what makes the query log reachable
    from the answer rather than only by listing."""
    response = await client.post(questions_url(), json={"question": "When is it due?"})

    assert response.headers["Location"].endswith(f"{questions_url()}/{QUESTION_ID}")


async def test_the_answer_reports_how_it_was_produced(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    """Retrieval size, dropped markers, model and cost are all part of the
    response because the point of this system is that it can be audited."""
    service.result = make_answer(retrieved=6, dropped_markers=2)

    body = (await client.post(questions_url(), json={"question": "why?"})).json()

    assert body["retrieved"] == 6
    assert body["dropped_markers"] == 2
    assert body["model"] == "claude-opus-5-20260101"
    assert (body["input_tokens"], body["output_tokens"]) == (1_234, 56)
    assert body["latency_ms"] == 2_100


async def test_an_unanswerable_question_is_a_201_not_an_error(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    """A course that does not cover something is not a failure. The question
    was asked, the log row exists, and the client is told plainly."""
    service.result = make_answer(
        answer="Nothing in this course's documents covers that question.",
        answered=False,
        citations=[],
        retrieved=0,
        model=None,
    )

    response = await client.post(questions_url(), json={"question": "who won in 1998?"})

    assert response.status_code == 201
    body = response.json()
    assert body["answered"] is False
    assert body["citations"] == []
    # No model was called, so there is nothing to name and nothing to bill.
    assert body["model"] is None
    assert body["input_tokens"] is None


async def test_a_citation_carries_no_score(client: AsyncClient, service: FakeAnswerService) -> None:
    """How close a passage was to the query is a retrieval diagnostic that
    /search reports. A citation read back out of the log has none, and a field
    that is present on fresh answers and absent on stored ones is worse than
    no field at all."""
    body = (await client.post(questions_url(), json={"question": "why?"})).json()

    assert "score" not in body["citations"][0]


# --- the request --------------------------------------------------------------


async def test_the_default_limit_is_the_answer_prompt_size(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    await client.post(questions_url(), json={"question": "why?"})

    assert service.asked[0][2] == DEFAULT_SEARCH_LIMIT


async def test_the_limit_is_passed_through(client: AsyncClient, service: FakeAnswerService) -> None:
    await client.post(questions_url(), json={"question": "why?", "limit": 3})

    assert service.asked[0][2] == 3


async def test_the_question_is_trimmed_before_it_is_used(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    await client.post(questions_url(), json={"question": "  when is it due?  "})

    assert service.asked[0][1] == "when is it due?"


async def test_a_blank_question_is_422(client: AsyncClient, service: FakeAnswerService) -> None:
    for blank in ("", "   ", "\t\n"):
        response = await client.post(questions_url(), json={"question": blank})
        assert response.status_code == 422, blank

    assert service.asked == []


async def test_a_missing_question_is_422(client: AsyncClient) -> None:
    assert (await client.post(questions_url(), json={})).status_code == 422


async def test_an_overlong_question_is_422(client: AsyncClient) -> None:
    """The question is embedded before it is answered, and the model truncates
    past its window - so an essay is refused rather than silently half-read."""
    response = await client.post(questions_url(), json={"question": "x" * (QUERY_MAX_LENGTH + 1)})

    assert response.status_code == 422


async def test_out_of_range_limits_are_422(client: AsyncClient) -> None:
    for limit in (0, -1, MAX_SEARCH_LIMIT + 1):
        response = await client.post(questions_url(), json={"question": "why?", "limit": limit})
        assert response.status_code == 422, limit


async def test_an_unknown_field_is_422(client: AsyncClient) -> None:
    """A misspelled key is a client bug, and accepting it silently means the
    caller believes it set something it did not."""
    response = await client.post(questions_url(), json={"question": "why?", "temperature": 0.7})

    assert response.status_code == 422


async def test_a_malformed_course_id_is_422_not_404(client: AsyncClient) -> None:
    response = await client.post("/api/v1/courses/not-a-uuid/questions", json={"question": "why?"})

    assert response.status_code == 422


# --- failures -----------------------------------------------------------------


async def test_asking_about_a_missing_course_is_404(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    service.raises = NotFoundError(f"no course with id {COURSE_ID}")

    response = await client.post(questions_url(), json={"question": "why?"})

    assert response.status_code == 404
    assert str(COURSE_ID) in response.json()["detail"]


async def test_an_unusable_embedding_model_is_503(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    """The question has to be embedded before it can be answered, so retrieval
    being unavailable is answering being unavailable."""
    service.raises = EmbeddingError("the embedding model is not available yet; try again shortly")

    response = await client.post(questions_url(), json={"question": "why?"})

    assert response.status_code == 503


async def test_an_unavailable_answer_model_is_503(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    service.raises = GenerationError("the answer service is busy; try again shortly")

    response = await client.post(questions_url(), json={"question": "why?"})

    assert response.status_code == 503
    assert "try again" in response.json()["detail"]


async def test_a_model_that_replied_with_nothing_usable_is_502(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    """Not 503: the upstream was reachable and did answer. What came back - a
    refusal, or an answer cut off mid-sentence - is the problem, and retrying
    is not the advice."""
    service.raises = UnusableAnswerError("the model declined to answer this question")

    response = await client.post(questions_url(), json={"question": "why?"})

    assert response.status_code == 502


async def test_a_corpus_that_moved_mid_answer_is_409(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    """A document was re-ingested between retrieval and the log write, so a
    cited chunk no longer exists. Asking again really does fix it."""
    service.raises = ConflictError("this course's documents changed; ask again")

    response = await client.post(questions_url(), json={"question": "why?"})

    assert response.status_code == 409


# --- the log ------------------------------------------------------------------


async def test_listing_returns_a_page_of_answers(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    service.page = [make_answer(question="first"), make_answer(question="second")]
    service.total = 2

    response = await client.get(questions_url())

    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["question"] for item in body["items"]] == ["first", "second"]
    assert (body["total"], body["limit"], body["offset"]) == (2, 20, 0)


async def test_listing_passes_pagination_through(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    await client.get(questions_url(), params={"limit": 5, "offset": 10})

    assert service.listed[0][1:] == (5, 10)


async def test_listing_a_missing_course_is_404(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    service.raises = NotFoundError(f"no course with id {COURSE_ID}")

    assert (await client.get(questions_url())).status_code == 404


async def test_an_empty_log_is_an_empty_200(client: AsyncClient) -> None:
    body = (await client.get(questions_url())).json()

    assert body["items"] == []
    assert body["total"] == 0


async def test_fetching_one_answer_uses_the_same_shape_as_asking(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    """One serialisation for both, which is what makes it impossible for a
    stored citation and a fresh one to disagree about what a citation is."""
    asked = (await client.post(questions_url(), json={"question": "why?"})).json()

    fetched = (await client.get(f"{questions_url()}/{QUESTION_ID}")).json()

    assert asked == fetched


async def test_fetching_a_missing_answer_is_404(
    client: AsyncClient, service: FakeAnswerService
) -> None:
    service.raises = NotFoundError(f"no question with id {QUESTION_ID} in course {COURSE_ID}")

    response = await client.get(f"{questions_url()}/{QUESTION_ID}")

    assert response.status_code == 404
