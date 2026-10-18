"""Course business rules.

Plain Python over a session: no FastAPI import, no status codes, no request or
response objects. That is what lets these methods be called from a test, a
script or a background task, and it is why the layer below (repositories) is
allowed to know about SQL while this one is not.

This service owns the transaction. Repositories stage work; ``commit`` happens
here, once per operation, so a partially applied change is not possible.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import violated_constraint
from app.errors import ConflictError, NotFoundError
from app.models import COURSE_CODE_UNIQUE, Course
from app.repositories.courses import CourseRepository
from app.repositories.documents import DocumentRepository
from app.schemas import CourseCreate, CourseUpdate
from app.storage import DocumentStorage


class CourseService:
    def __init__(self, session: AsyncSession, storage: DocumentStorage) -> None:
        self._session = session
        self._storage = storage
        self._courses = CourseRepository(session)
        self._documents = DocumentRepository(session)

    async def create(self, payload: CourseCreate) -> Course:
        course = Course(
            code=payload.code,
            title=payload.title,
            description=payload.description,
        )
        self._courses.add(course)
        return await self._commit(course)

    async def get(self, course_id: uuid.UUID) -> Course:
        course = await self._courses.get(course_id)
        if course is None:
            raise NotFoundError(f"no course with id {course_id}")
        return course

    async def list(self, *, limit: int, offset: int) -> tuple[Sequence[Course], int]:
        # Two statements in one transaction, so the page and its total describe
        # the same snapshot of the table rather than two different ones.
        return await self._courses.list(limit=limit, offset=offset), await self._courses.count()

    async def update(self, course_id: uuid.UUID, payload: CourseUpdate) -> Course:
        course = await self.get(course_id)
        for field, value in payload.changes().items():
            setattr(course, field, value)
        return await self._commit(course)

    async def delete(self, course_id: uuid.UUID) -> None:
        """Delete a course, its documents, its chunks and their stored files.

        The rows go by ON DELETE CASCADE, but nothing cascades to the
        filesystem, so the document ids have to be read *before* the delete -
        afterwards there is no record of which files to remove, and every PDF
        ever uploaded to the course leaks.

        Bytes go last, for the same reason as in DocumentService: an orphaned
        file wastes disk, while a live row whose file is gone is a citation
        that can never be resolved.
        """
        course = await self.get(course_id)
        document_ids = await self._documents.list_ids_for_course(course_id)
        await self._courses.delete(course)
        await self._session.commit()
        for document_id in document_ids:
            self._storage.delete(document_id)

    async def _commit(self, course: Course) -> Course:
        """Commit, translating the one constraint a client can trip into 409.

        Deliberately no "does this code already exist?" query first: two
        concurrent requests can both pass that check before either inserts. The
        UNIQUE index is evaluated atomically, so attempting the write and
        handling the rejection is the only correct order. A pre-check would be
        a nicer error message and a real race.
        """
        # Read before committing. A rollback expires a persistent instance, and
        # touching an expired attribute afterwards triggers a lazy refresh,
        # which under asyncio raises MissingGreenlet rather than loading.
        code = course.code
        try:
            await self._session.commit()
        except IntegrityError as error:
            await self._session.rollback()
            if violated_constraint(error) == COURSE_CODE_UNIQUE:
                raise ConflictError(f"course code {code!r} is already in use") from error
            # Any other integrity failure is a bug here, not a client mistake,
            # so it must not be dressed up as a 409.
            raise
        return course
