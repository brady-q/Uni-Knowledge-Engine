"""Queries against the ``questions`` and ``citations`` tables.

The query log is written once per answer and read back to render history, and
both of those are one statement. What makes this repository worth its file is
the *loading*: rendering a stored citation means reaching through the chunk it
points at to the document that chunk belongs to, and every read here says so
explicitly.

That is not a performance nicety. A lazy load on any of those relationships
would be synchronous IO inside an async request, which asyncpg refuses outright
- so the choice is between eager loading declared here and a
``MissingGreenlet`` in the router. Declaring it also means one query per read
instead of one per citation.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from app.models import Chunk, Citation, Question

# One SELECT for the questions, one for their citations joined through to the
# documents. selectinload for the collection because a joined one-to-many
# multiplies the parent rows; joinedload for the two many-to-one hops because
# each adds a row nothing has to be de-duplicated from.
_WITH_CITATIONS = selectinload(Question.citations).options(
    joinedload(Citation.chunk).joinedload(Chunk.document)
)


class QuestionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(self, question: Question) -> None:
        """Stage the log row and its citations. No SQL until the flush."""
        self._session.add(question)

    async def get_in_course(self, course_id: uuid.UUID, question_id: uuid.UUID) -> Question | None:
        """Fetch one logged question, scoped to its course.

        Scoped rather than fetched by primary key alone: the route is nested
        under a course, and a question id from another course must miss here
        rather than be served under the wrong scope.
        """
        result = await self._session.execute(
            select(Question)
            .where(Question.id == question_id, Question.course_id == course_id)
            .options(_WITH_CITATIONS)
        )
        return result.scalar_one_or_none()

    async def list_for_course(
        self, course_id: uuid.UUID, *, limit: int, offset: int
    ) -> Sequence[Question]:
        # Newest first, which is what a Q&A history shows. The id breaks ties:
        # created_at is not unique, and an ordering that is not total lets
        # LIMIT/OFFSET repeat or skip a row between pages.
        result = await self._session.execute(
            select(Question)
            .where(Question.course_id == course_id)
            .order_by(Question.created_at.desc(), Question.id.desc())
            .limit(limit)
            .offset(offset)
            .options(_WITH_CITATIONS)
        )
        return result.scalars().all()

    async def count_for_course(self, course_id: uuid.UUID) -> int:
        result = await self._session.execute(
            select(func.count()).select_from(Question).where(Question.course_id == course_id)
        )
        return result.scalar_one()
