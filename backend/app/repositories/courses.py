"""Queries against the ``courses`` table."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Course


class CourseRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(self, course: Course) -> None:
        """Stage an insert. No SQL runs until the session is flushed."""
        self._session.add(course)

    async def get(self, course_id: uuid.UUID) -> Course | None:
        """Fetch by primary key, using the session's identity map if it is warm."""
        return await self._session.get(Course, course_id)

    async def get_by_code(self, code: str) -> Course | None:
        result = await self._session.execute(select(Course).where(Course.code == code))
        return result.scalar_one_or_none()

    async def list(self, *, limit: int, offset: int) -> Sequence[Course]:
        # Ordered by code because an unordered LIMIT/OFFSET has no defined row
        # order, so paging through it can repeat or skip rows. The UNIQUE index
        # on code both makes the order total and serves the sort.
        result = await self._session.execute(
            select(Course).order_by(Course.code).limit(limit).offset(offset)
        )
        return result.scalars().all()

    async def count(self) -> int:
        result = await self._session.execute(select(func.count()).select_from(Course))
        return result.scalar_one()

    async def delete(self, course: Course) -> None:
        """Delete the row. Documents and chunks go with it, via ON DELETE CASCADE."""
        await self._session.delete(course)
