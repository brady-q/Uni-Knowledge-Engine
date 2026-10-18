"""Semantic search within a course.

Nested under a course like the document routes, and for the same reason: the
scope of a retrieval is not a detail a client can be trusted to remember to
send. There is no cross-course search endpoint, because a chunk from an
unrelated unit appearing in an answer's citations is a correctness failure
rather than a ranking one.

``GET`` rather than ``POST``: this is a read, it is repeatable, and a query that
fits in ``QUERY_MAX_LENGTH`` characters fits in a URL. The question-answering
route in ``app.routers.answers`` is a ``POST``, because that one writes a
query-log row and spends money at a third party.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.embedding import Embedder, get_embedder
from app.repositories.chunks import ChunkMatch
from app.schemas import (
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    SearchMatch,
    SearchQuery,
    SearchResponse,
)
from app.services.retrieval import RetrievalService

router = APIRouter(prefix="/courses/{course_id}/search", tags=["search"])

OpenApiResponses = dict[int | str, dict[str, Any]]

SEARCH_FAILURE_RESPONSES: OpenApiResponses = {
    status.HTTP_404_NOT_FOUND: {"description": "No course with that id"},
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "description": "The embedding model is still loading, or failed to load"
    },
}


def get_retrieval_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> RetrievalService:
    return RetrievalService(session, embedder)


ServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]
CourseId = Annotated[uuid.UUID, Path(description="Course identifier")]


def _to_match(match: ChunkMatch) -> SearchMatch:
    """Flatten a retrieved row into the citation-shaped DTO.

    Every field is read off the stored rows - the chunk's recorded page range
    and the document's own filename - which is the mechanism the citation
    guarantee rests on, applied here as well as in the answer path.
    """
    return SearchMatch(
        chunk_id=match.chunk.id,
        document_id=match.document.id,
        filename=match.document.filename,
        ordinal=match.chunk.ordinal,
        page_start=match.chunk.page_start,
        page_end=match.chunk.page_end,
        text=match.chunk.text,
        score=match.score,
    )


@router.get(
    "",
    summary="Search a course's documents",
    responses=SEARCH_FAILURE_RESPONSES,
)
async def search_course(
    course_id: CourseId,
    service: ServiceDep,
    q: Annotated[
        SearchQuery, Query(description="What to search for", examples=["gradient descent"])
    ],
    limit: Annotated[int, Query(ge=1, le=MAX_SEARCH_LIMIT)] = DEFAULT_SEARCH_LIMIT,
) -> SearchResponse:
    """Rank this course's passages against a query, closest first.

    Matches are semantic, not lexical: a query about "learning rate" retrieves
    a passage about "step size" without sharing a word with it.

    An empty ``items`` list is a valid answer and means one of three things -
    the course has no documents, its documents are still being embedded, or
    nothing in them is related to the query. Poll the documents' ``status`` to
    tell the second case from the others.
    """
    matches = await service.search(course_id, q, limit=limit)
    return SearchResponse(
        query=q,
        limit=limit,
        items=[_to_match(match) for match in matches],
    )
