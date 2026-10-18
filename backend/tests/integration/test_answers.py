"""The answer path end to end, against a real Postgres with real migrations.

Upload a PDF, let it ingest, ask a question, and check what came back and what
was written down. The unit tests prove the resolver and the router in
isolation; these prove the things only the whole pipeline can be wrong about:

* that a marker resolves to the page range and filename of the chunk the
  retrieval actually returned, rather than to whatever the fake was told to say
* that the query log and its citation rows are really written, and that reading
  them back produces the same answer that was served
* that a citation cannot outlive the document it points at, because the
  database will not let it

Both models are fakes. Vectors come from ``tests.embeddings.FakeEmbedder`` and
answers from ``tests.answers.FakeAnswerModel``, whose whole purpose here is
that a test can choose the markers "the model" writes - including markers that
name nothing, which is the case the design exists for and which a real model
would refuse to produce on demand.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import GenerationError
from app.models import Chunk, Citation, Document, Question
from app.services.answers import NO_SOURCES_ANSWER, UNSUPPORTED_ANSWER
from tests.answers import FakeAnswerModel
from tests.pdfs import make_pdf

pytestmark = pytest.mark.integration

# Two topics, worded so a bag-of-words query lands on one and misses the other,
# and padded so each page is worth more than a single chunk.
DEADLINES = [
    "The second assignment is due at the end of week six and is submitted "
    "through the course portal rather than by email.",
    "Work handed in after the week six deadline loses ten percent per day, up "
    "to a maximum of three days late.",
]
GRADING = [
    "The final grade is the weighted mean of two assignments and one exam, "
    "with the exam counting for half of the total.",
    "There is no scaling applied to the exam mark, so the published weighting "
    "is the whole of the grading scheme.",
]


def page(sentences: list[str], repeats: int = 5) -> list[str]:
    return [f"Section {index}. {' '.join(sentences)}" for index in range(repeats)]


DEADLINES_PAGE = page(DEADLINES)
GRADING_PAGE = page(GRADING)


async def create_course(client: AsyncClient, code: str = "CS229") -> str:
    response = await client.post(
        "/api/v1/courses", json={"code": code, "title": "Machine Learning"}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def upload(
    client: AsyncClient,
    course_id: str,
    pages: list[list[str]],
    name: str = "handbook.pdf",
) -> dict[str, Any]:
    response = await client.post(
        f"/api/v1/courses/{course_id}/documents",
        files={"file": (name, make_pdf(pages), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


async def ask(client: AsyncClient, course_id: str, question: str, **body: Any) -> dict[str, Any]:
    response = await client.post(
        f"/api/v1/courses/{course_id}/questions", json={"question": question, **body}
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


# --- an answered question -----------------------------------------------------


async def test_a_marker_resolves_to_the_page_the_passage_came_from(
    api_client: AsyncClient, answer_model: FakeAnswerModel, db_session: AsyncSession
) -> None:
    """The claim this project is built on, checked against real rows.

    The fake model is told to cite [S1]. What comes back has to be the document
    and page range of the top-ranked chunk *as stored*, not anything the fake
    was in a position to make up - it was never shown a filename or a page.
    """
    course_id = await create_course(api_client)
    document = await upload(api_client, course_id, [DEADLINES_PAGE, GRADING_PAGE])
    answer_model.text = "It is due at the end of week six [S1]."

    body = await ask(api_client, course_id, "when is the second assignment due?")

    assert body["answered"] is True
    assert body["answer"] == "It is due at the end of week six [S1]."
    [citation] = body["citations"]
    assert citation["marker"] == "S1"
    assert citation["document_id"] == document["id"]
    assert citation["filename"] == "handbook.pdf"

    chunk = (
        await db_session.execute(select(Chunk).where(Chunk.id == uuid.UUID(citation["chunk_id"])))
    ).scalar_one()
    assert (citation["page_start"], citation["page_end"]) == (chunk.page_start, chunk.page_end)
    assert citation["text"] == chunk.text
    # The page the citation names is a real page of the real PDF. Read off the
    # row rather than the upload response, which returns before ingestion has
    # run and therefore before the page count is known.
    stored = (
        await db_session.execute(select(Document).where(Document.id == chunk.document_id))
    ).scalar_one()
    assert stored.page_count is not None
    assert 1 <= chunk.page_start <= chunk.page_end <= stored.page_count


async def test_the_model_is_shown_the_passages_and_nothing_else(
    api_client: AsyncClient, answer_model: FakeAnswerModel, db_session: AsyncSession
) -> None:
    """The mechanism, rather than the outcome: page numbers cannot be
    hallucinated because they are never in the context to begin with."""
    course_id = await create_course(api_client)
    document = await upload(api_client, course_id, [DEADLINES_PAGE], name="week1-handbook.pdf")

    await ask(api_client, course_id, "when is the assignment due?")

    question, passages = answer_model.calls[-1]
    assert question == "when is the assignment due?"
    assert passages
    chunks = (
        (
            await db_session.execute(
                select(Chunk).where(Chunk.document_id == uuid.UUID(document["id"]))
            )
        )
        .scalars()
        .all()
    )
    stored = {chunk.text for chunk in chunks}
    for passage in passages:
        assert passage in stored
    joined = "\n".join(passages)
    assert "week1-handbook.pdf" not in joined
    assert document["id"] not in joined
    assert not any(str(chunk.id) in joined for chunk in chunks)


async def test_the_retrieval_limit_decides_how_many_passages_are_offered(
    api_client: AsyncClient, answer_model: FakeAnswerModel
) -> None:
    course_id = await create_course(api_client)
    await upload(api_client, course_id, [DEADLINES_PAGE, GRADING_PAGE])

    body = await ask(api_client, course_id, "how is the course graded?", limit=2)

    assert len(answer_model.passages) == 2
    assert body["retrieved"] == 2


async def test_a_repeated_marker_is_one_citation(
    api_client: AsyncClient, answer_model: FakeAnswerModel, db_session: AsyncSession
) -> None:
    course_id = await create_course(api_client)
    await upload(api_client, course_id, [DEADLINES_PAGE])
    answer_model.text = "Week six [S1], and late work is penalised [S1]."

    body = await ask(api_client, course_id, "when is it due?")

    assert len(body["citations"]) == 1
    # And the log agrees: the unique constraint on (question_id, marker) would
    # have rejected a second row for the same marker anyway.
    assert await db_session.scalar(select(func.count()).select_from(Citation)) == 1


# --- an invented marker -------------------------------------------------------


async def test_an_invented_marker_is_removed_and_counted(
    api_client: AsyncClient, answer_model: FakeAnswerModel, db_session: AsyncSession
) -> None:
    """The model cites a passage it was never given. The claim it supported
    keeps its wording, loses its footnote, and the log records that it
    happened - which is what makes the guarantee measurable rather than
    asserted."""
    course_id = await create_course(api_client)
    await upload(api_client, course_id, [DEADLINES_PAGE])
    answer_model.text = "Week six [S1], and the exam is in week twelve [S9]."

    body = await ask(api_client, course_id, "when is it due?", limit=2)

    assert body["answer"] == "Week six [S1], and the exam is in week twelve."
    assert body["dropped_markers"] == 1
    assert [citation["marker"] for citation in body["citations"]] == ["S1"]

    row = (await db_session.execute(select(Question))).scalar_one()
    assert row.dropped_markers == 1
    assert row.answered is True


async def test_an_answer_whose_every_marker_was_invented_is_not_served(
    api_client: AsyncClient, answer_model: FakeAnswerModel, db_session: AsyncSession
) -> None:
    """The worst case, and the one worth being strict about.

    A confident answer with all of its support removed is exactly what this
    system promises not to produce, so the prose is discarded rather than
    published with an empty citation list - and the log keeps the count that
    says why.
    """
    course_id = await create_course(api_client)
    await upload(api_client, course_id, [DEADLINES_PAGE])
    answer_model.text = "The exam is in week twelve [S8] and worth everything [S9]."

    body = await ask(api_client, course_id, "when is the exam?")

    assert body["answered"] is False
    assert body["citations"] == []
    assert body["answer"] == UNSUPPORTED_ANSWER
    assert "week twelve" not in body["answer"]
    assert body["dropped_markers"] == 2
    assert (await db_session.execute(select(Question))).scalar_one().answer == UNSUPPORTED_ANSWER


async def test_an_honest_abstention_keeps_the_model_s_own_words(
    api_client: AsyncClient, answer_model: FakeAnswerModel
) -> None:
    """Uncited *and* not claiming to have answered is a legitimate result, so
    it is passed through rather than replaced."""
    course_id = await create_course(api_client)
    await upload(api_client, course_id, [DEADLINES_PAGE])
    answer_model.text = "The course materials do not say."
    answer_model.answered = False

    body = await ask(api_client, course_id, "who teaches this course?")

    assert body["answered"] is False
    assert body["answer"] == "The course materials do not say."
    assert body["dropped_markers"] == 0


# --- nothing to answer from ---------------------------------------------------


async def test_a_course_with_no_documents_is_answered_without_calling_the_model(
    api_client: AsyncClient, answer_model: FakeAnswerModel, db_session: AsyncSession
) -> None:
    """Retrieval came back empty, so there is no prompt worth sending and no
    reason to spend anything. The row still gets written: "somebody asked and
    we had nothing" is the most useful line in this log."""
    course_id = await create_course(api_client)

    body = await ask(api_client, course_id, "when is the assignment due?")

    assert answer_model.calls == []
    assert body["answered"] is False
    assert body["answer"] == NO_SOURCES_ANSWER
    assert body["retrieved"] == 0
    # NULL, which is what distinguishes "no model was called" from "a model
    # was called and said nothing".
    assert body["model"] is None
    assert (body["input_tokens"], body["output_tokens"]) == (None, None)

    row = (await db_session.execute(select(Question))).scalar_one()
    assert row.model is None
    assert row.retrieved == 0


async def test_asking_a_course_that_does_not_exist_is_404(api_client: AsyncClient) -> None:
    response = await api_client.post(
        f"/api/v1/courses/{uuid.uuid4()}/questions", json={"question": "why?"}
    )

    assert response.status_code == 404


async def test_an_unavailable_model_is_503_and_logs_nothing(
    api_client: AsyncClient, answer_model: FakeAnswerModel, db_session: AsyncSession
) -> None:
    """No answer means no log row: the query log is a record of answers given,
    and a failed third-party call is an incident, not an answer."""
    course_id = await create_course(api_client)
    await upload(api_client, course_id, [DEADLINES_PAGE])
    answer_model.raises = GenerationError("the answer service is busy; try again shortly")

    response = await api_client.post(
        f"/api/v1/courses/{course_id}/questions", json={"question": "when is it due?"}
    )

    assert response.status_code == 503
    assert await db_session.scalar(select(func.count()).select_from(Question)) == 0


# --- the query log ------------------------------------------------------------


async def test_the_log_records_what_was_asked_and_what_it_cost(
    api_client: AsyncClient, answer_model: FakeAnswerModel, db_session: AsyncSession
) -> None:
    course_id = await create_course(api_client)
    await upload(api_client, course_id, [DEADLINES_PAGE])

    body = await ask(api_client, course_id, "when is the second assignment due?")

    row = (await db_session.execute(select(Question))).scalar_one()
    assert row.id == uuid.UUID(body["id"])
    assert row.course_id == uuid.UUID(course_id)
    assert row.question == "when is the second assignment due?"
    assert row.model == answer_model.model_name
    assert (row.input_tokens, row.output_tokens) == (
        answer_model.input_tokens,
        answer_model.output_tokens,
    )
    # Retrieval plus generation plus resolution, and all of that took *some*
    # time; the fakes make it small but it cannot be negative.
    assert row.latency_ms >= 0


async def test_an_answer_can_be_fetched_again_unchanged(
    api_client: AsyncClient, answer_model: FakeAnswerModel
) -> None:
    """The Location a 201 promises, and the property that matters underneath
    it: the citations are re-resolved from the rows and come out the same."""
    course_id = await create_course(api_client)
    await upload(api_client, course_id, [DEADLINES_PAGE, GRADING_PAGE])
    answer_model.text = "Week six [S1], and half the grade is the exam [S2]."

    created = await api_client.post(
        f"/api/v1/courses/{course_id}/questions", json={"question": "when is it due?"}
    )
    fetched = await api_client.get(created.headers["Location"])

    assert fetched.status_code == 200, fetched.text
    assert fetched.json() == created.json()


async def test_the_log_lists_newest_first(
    api_client: AsyncClient, answer_model: FakeAnswerModel
) -> None:
    course_id = await create_course(api_client)
    await upload(api_client, course_id, [DEADLINES_PAGE])

    for question in ("first question?", "second question?", "third question?"):
        await ask(api_client, course_id, question)

    body = (await api_client.get(f"/api/v1/courses/{course_id}/questions")).json()

    assert [item["question"] for item in body["items"]] == [
        "third question?",
        "second question?",
        "first question?",
    ]
    assert body["total"] == 3


async def test_the_log_is_scoped_to_its_course(
    api_client: AsyncClient, answer_model: FakeAnswerModel
) -> None:
    """A question asked of one course must not be readable through another,
    which is the same reason there is no unscoped route to fetch one."""
    first = await create_course(api_client, code="CS229")
    second = await create_course(api_client, code="CS230")
    await upload(api_client, first, [DEADLINES_PAGE])
    asked = await ask(api_client, first, "when is it due?")

    listed = (await api_client.get(f"/api/v1/courses/{second}/questions")).json()
    fetched = await api_client.get(f"/api/v1/courses/{second}/questions/{asked['id']}")

    assert listed["items"] == []
    assert fetched.status_code == 404


async def test_listing_a_course_that_does_not_exist_is_404(api_client: AsyncClient) -> None:
    """Distinct from a course that has never been asked anything, which is an
    empty list."""
    response = await api_client.get(f"/api/v1/courses/{uuid.uuid4()}/questions")

    assert response.status_code == 404


# --- a citation cannot outlive its source -------------------------------------


async def test_deleting_a_document_takes_its_citations_with_it(
    api_client: AsyncClient, answer_model: FakeAnswerModel, db_session: AsyncSession
) -> None:
    """ON DELETE CASCADE from chunks, and deliberately so: a marker resolving
    to a chunk that no longer exists is the unresolvable citation the schema is
    built to make impossible. The answer survives in the log; its evidence does
    not, and the response says as much by listing nothing."""
    course_id = await create_course(api_client)
    document = await upload(api_client, course_id, [DEADLINES_PAGE])
    asked = await ask(api_client, course_id, "when is it due?")
    assert asked["citations"]

    deleted = await api_client.delete(f"/api/v1/courses/{course_id}/documents/{document['id']}")
    assert deleted.status_code == 204

    fetched = (await api_client.get(f"/api/v1/courses/{course_id}/questions/{asked['id']}")).json()
    assert fetched["citations"] == []
    assert fetched["answer"] == asked["answer"]
    assert await db_session.scalar(select(func.count()).select_from(Citation)) == 0
    assert await db_session.scalar(select(func.count()).select_from(Question)) == 1


async def test_deleting_a_course_takes_its_questions_with_it(
    api_client: AsyncClient, answer_model: FakeAnswerModel, db_session: AsyncSession
) -> None:
    course_id = await create_course(api_client)
    await upload(api_client, course_id, [DEADLINES_PAGE])
    await ask(api_client, course_id, "when is it due?")

    assert (await api_client.delete(f"/api/v1/courses/{course_id}")).status_code == 204

    assert await db_session.scalar(select(func.count()).select_from(Question)) == 0
    assert await db_session.scalar(select(func.count()).select_from(Citation)) == 0
    assert await db_session.scalar(select(func.count()).select_from(Document)) == 0
