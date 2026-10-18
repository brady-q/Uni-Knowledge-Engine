"""Asking a question of a course, and reading back what has been asked.

Nested under a course like search and documents, and for the strongest version
of the same reason: an answer assembled from another course's material is a
correctness failure, and the only way to be sure it cannot happen is for there
to be no route that could express it.

``POST`` rather than ``GET``, unlike ``/search``. Answering is not a read - it
writes a query-log row and it spends money at a third party - and neither of
those is something a browser should be free to repeat when someone presses
back. The two ``GET`` routes below are the read half: the log, which is what
``M6``'s Q&A screen renders as history.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.embedding import Embedder, get_embedder
from app.llm import AnswerModel, get_answer_model, marker
from app.schemas import (
    DEFAULT_QUESTION_PAGE_SIZE,
    MAX_QUESTION_PAGE_SIZE,
    AnswerCitation,
    AnswerPage,
    AnswerRead,
    QuestionAsk,
)
from app.services.answers import Answer, AnswerService

router = APIRouter(prefix="/courses/{course_id}/questions", tags=["answers"])

OpenApiResponses = dict[int | str, dict[str, Any]]

NOT_FOUND_RESPONSE: OpenApiResponses = {
    status.HTTP_404_NOT_FOUND: {"description": "No such course, or no such question in it"}
}
ASK_FAILURE_RESPONSES: OpenApiResponses = {
    **NOT_FOUND_RESPONSE,
    status.HTTP_409_CONFLICT: {
        "description": "The course's documents changed mid-answer; ask again"
    },
    status.HTTP_502_BAD_GATEWAY: {
        "description": "The model replied with nothing usable: a refusal, or an answer cut short"
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "description": "The embedding model or the answer model is unavailable"
    },
}


def get_answer_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    model: Annotated[AnswerModel, Depends(get_answer_model)],
) -> AnswerService:
    return AnswerService(session, embedder, model)


ServiceDep = Annotated[AnswerService, Depends(get_answer_service)]
CourseId = Annotated[uuid.UUID, Path(description="Course identifier")]
QuestionId = Annotated[uuid.UUID, Path(description="Question identifier")]


def _to_answer(answer: Answer) -> AnswerRead:
    """Flatten a service result into the response DTO.

    The one place an answer becomes JSON, used by all three routes, so a
    citation looks the same whether it was generated a moment ago or read out
    of the log. Every field of a citation except the marker is taken off the
    stored chunk and document rows - which is the citation guarantee, expressed
    as a function that has nothing else available to it.
    """
    question = answer.question
    return AnswerRead(
        id=question.id,
        course_id=question.course_id,
        question=question.question,
        answer=question.answer,
        answered=question.answered,
        citations=[
            AnswerCitation(
                marker=marker(citation.marker),
                chunk_id=citation.chunk.id,
                document_id=citation.document.id,
                filename=citation.document.filename,
                page_start=citation.chunk.page_start,
                page_end=citation.chunk.page_end,
                text=citation.chunk.text,
            )
            for citation in answer.citations
        ],
        retrieved=question.retrieved,
        dropped_markers=question.dropped_markers,
        model=question.model,
        input_tokens=question.input_tokens,
        output_tokens=question.output_tokens,
        latency_ms=question.latency_ms,
        created_at=question.created_at,
    )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Ask a question about a course",
    responses=ASK_FAILURE_RESPONSES,
)
async def ask_question(
    course_id: CourseId,
    payload: QuestionAsk,
    service: ServiceDep,
    request: Request,
    response: Response,
) -> AnswerRead:
    """Answer a question from this course's documents, with citations.

    Every marker in ``answer`` has a matching entry in ``citations``, and every
    citation names a real document and page range read out of the database.
    Markers the model invented are removed from the answer before it is
    returned, and counted in ``dropped_markers``.

    ``answered`` is false when the material did not support an answer - because
    the course has nothing on the subject, its documents are still being
    embedded, or the passages retrieved turned out not to be relevant. That is
    a valid outcome and a ``201``, not an error: the question was asked, and the
    log row saying so is what a course owner needs to see.
    """
    answer = await service.ask(course_id, payload.question, limit=payload.limit)
    response.headers["Location"] = str(
        request.url_for("get_question", course_id=course_id, question_id=answer.question.id)
    )
    return _to_answer(answer)


@router.get("", summary="List the questions asked of a course", responses=NOT_FOUND_RESPONSE)
async def list_questions(
    course_id: CourseId,
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=MAX_QUESTION_PAGE_SIZE)] = DEFAULT_QUESTION_PAGE_SIZE,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AnswerPage:
    """The query log for this course, newest first.

    Citations are re-resolved on every read, so an answer whose document has
    since been deleted comes back with fewer citations rather than with a page
    number that no longer points anywhere.
    """
    answers, total = await service.list(course_id, limit=limit, offset=offset)
    return AnswerPage(
        items=[_to_answer(answer) for answer in answers],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{question_id}", summary="Fetch one answered question", responses=NOT_FOUND_RESPONSE)
async def get_question(
    course_id: CourseId, question_id: QuestionId, service: ServiceDep
) -> AnswerRead:
    return _to_answer(await service.get(course_id, question_id))
