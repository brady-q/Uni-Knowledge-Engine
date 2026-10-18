"""Semantic retrieval: the read half of the pipeline.

Two steps, and the split between them is the point. Embedding the query is the
only part that involves a model; ranking is a SQL statement over vectors that
were computed at ingestion time. Nothing here reads a PDF, and nothing here
decides what a chunk *means* - it returns rows, in order, with their scores.

This is also the seam the answer path is built on. ``app.services.answers``
calls ``search`` and hands the resulting ``ChunkMatch`` list to Claude behind
opaque ``[S1]…[S6]`` markers; the citations it resolves come from these rows and
from nowhere else. Keeping retrieval independent of generation means the
retrieval quality can be measured on its own, which is what the eval harness in
``M7`` needs.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding import Embedder
from app.errors import NotFoundError
from app.repositories.chunks import ChunkMatch, ChunkRepository
from app.repositories.courses import CourseRepository


class RetrievalService:
    def __init__(self, session: AsyncSession, embedder: Embedder) -> None:
        self._courses = CourseRepository(session)
        self._chunks = ChunkRepository(session)
        self._embedder = embedder

    async def search(self, course_id: uuid.UUID, query: str, *, limit: int) -> list[ChunkMatch]:
        """Rank a course's chunks against a query, closest first.

        The course is looked up first so that "no such course" is a 404 and
        stays distinguishable from "this course has nothing to say about that",
        which is an empty result and a perfectly good answer. Without the
        lookup both would be an empty list, and a mistyped id would look like a
        course whose documents had all failed to ingest.

        An empty list is also what a course whose documents are still being
        embedded returns - the query excludes chunks without vectors - so a
        caller that finds nothing should check the documents' status before
        concluding the corpus does not cover the question.
        """
        if await self._courses.get(course_id) is None:
            raise NotFoundError(f"no course with id {course_id}")

        embedding = await self._embedder.embed_query(query)
        return await self._chunks.search_by_embedding(course_id, embedding, limit=limit)
