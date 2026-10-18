"""Queries against the ``documents`` table."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(self, document: Document) -> None:
        """Stage an insert. No SQL runs until the session is flushed."""
        self._session.add(document)

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return await self._session.get(Document, document_id)

    async def get_in_course(self, course_id: uuid.UUID, document_id: uuid.UUID) -> Document | None:
        """Fetch by id, but only if it belongs to this course.

        Scoping the lookup rather than fetching and then comparing means a
        mismatched pair is a 404 for the same reason a missing id is: the URL
        names a document that does not exist *there*, and answering differently
        would confirm the existence of a document in another course.
        """
        result = await self._session.execute(
            select(Document).where(Document.id == document_id, Document.course_id == course_id)
        )
        return result.scalar_one_or_none()

    async def list_for_course(
        self, course_id: uuid.UUID, *, limit: int, offset: int
    ) -> Sequence[Document]:
        # Newest first, which is what someone who just uploaded wants to see.
        # id breaks ties: two documents uploaded in the same transaction share
        # a created_at, and an unordered LIMIT/OFFSET can repeat or skip rows.
        result = await self._session.execute(
            select(Document)
            .where(Document.course_id == course_id)
            .order_by(Document.created_at.desc(), Document.id)
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def count_for_course(self, course_id: uuid.UUID) -> int:
        result = await self._session.execute(
            select(func.count()).select_from(Document).where(Document.course_id == course_id)
        )
        return result.scalar_one()

    async def delete(self, document: Document) -> None:
        """Delete the row. Chunks go with it, via ON DELETE CASCADE."""
        await self._session.delete(document)

    async def list_ids_for_course(self, course_id: uuid.UUID) -> Sequence[uuid.UUID]:
        """Just the ids, for deleting the bytes a cascade is about to orphan.

        Selects the id column rather than whole entities: this is only ever
        used to derive storage paths, and a course can hold a lot of documents.
        """
        result = await self._session.execute(
            select(Document.id).where(Document.course_id == course_id)
        )
        return result.scalars().all()
