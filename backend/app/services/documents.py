"""Document business rules: upload, listing, deletion.

Two stores have to stay in step here - a row in Postgres and a file on disk -
and there is no transaction spanning both. The order is chosen so that the
failure modes are the harmless ones:

* The row is committed **before** the file is moved into place. A duplicate
  upload is rejected by the UNIQUE index without any bytes having been
  published, and the staged file is discarded.
* The file is deleted **after** the row is deleted. The worst outcome is an
  orphaned file, which wastes disk. The reverse order risks a live document
  whose bytes are gone, which is a citation that cannot be resolved.

Ingestion itself is not started here. This service returns as soon as the
document is durable; turning it into chunks happens after the response, and
that scheduling is the router's business.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from pathlib import PurePosixPath

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import violated_constraint
from app.errors import ConflictError, NotFoundError, UnsupportedMediaError
from app.models import (
    DOCUMENT_CONTENT_HASH_UNIQUE,
    DOCUMENT_COURSE_FOREIGN_KEY,
    Chunk,
    Document,
)
from app.repositories.chunks import ChunkRepository
from app.repositories.courses import CourseRepository
from app.repositories.documents import DocumentRepository
from app.schemas import FILENAME_MAX_LENGTH
from app.storage import DocumentStorage, StagedUpload, UploadStream

logger = logging.getLogger(__name__)

# Every PDF begins with these bytes followed by a version number.
PDF_MAGIC = b"%PDF-"

FALLBACK_FILENAME = "document.pdf"


class DocumentService:
    def __init__(
        self,
        session: AsyncSession,
        storage: DocumentStorage,
        *,
        max_upload_bytes: int,
    ) -> None:
        self._session = session
        self._storage = storage
        self._max_upload_bytes = max_upload_bytes
        self._courses = CourseRepository(session)
        self._documents = DocumentRepository(session)
        self._chunks = ChunkRepository(session)

    async def upload(self, course_id: uuid.UUID, upload: UploadStream) -> Document:
        """Receive a PDF and record it as pending ingestion."""
        # A courtesy check, not the guarantee: the course can still disappear
        # before the insert. The foreign key is what actually holds, and the
        # violation is translated below. Doing it anyway means the common case
        # of a mistyped course id fails before 50 MiB has been uploaded.
        if await self._courses.get(course_id) is None:
            raise NotFoundError(f"no course with id {course_id}")

        staged = await self._storage.stage(upload, max_bytes=self._max_upload_bytes)
        try:
            _require_pdf(staged)
            document = Document(
                course_id=course_id,
                filename=_safe_filename(upload.filename),
                content_hash=staged.content_hash,
                byte_size=staged.byte_size,
            )
            self._documents.add(document)
            await self._commit_new(document, course_id)
        except BaseException:
            # Nothing references these bytes and nothing ever will.
            self._storage.discard(staged.path)
            raise

        try:
            self._storage.commit(staged, document.id)
        except OSError:
            # The row is committed but its bytes did not land. Undo the row
            # rather than leave a document that can only ever fail ingestion.
            logger.exception("could not store bytes for document %s", document.id)
            await self._documents.delete(document)
            await self._session.commit()
            self._storage.discard(staged.path)
            raise

        return document

    async def get(self, course_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        document = await self._documents.get_in_course(course_id, document_id)
        if document is None:
            raise NotFoundError(f"no document with id {document_id} in course {course_id}")
        return document

    async def list(
        self, course_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[Sequence[Document], int]:
        if await self._courses.get(course_id) is None:
            raise NotFoundError(f"no course with id {course_id}")
        # Both statements in one transaction, so the page and its total
        # describe the same snapshot rather than two different ones.
        return (
            await self._documents.list_for_course(course_id, limit=limit, offset=offset),
            await self._documents.count_for_course(course_id),
        )

    async def list_chunks(
        self, course_id: uuid.UUID, document_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[Sequence[Chunk], int]:
        document = await self.get(course_id, document_id)
        return (
            await self._chunks.list_for_document(document.id, limit=limit, offset=offset),
            await self._chunks.count_for_document(document.id),
        )

    async def delete(self, course_id: uuid.UUID, document_id: uuid.UUID) -> None:
        """Remove a document, its chunks and its bytes."""
        document = await self.get(course_id, document_id)
        await self._documents.delete(document)
        await self._session.commit()
        # Only once the row is gone. See the module docstring on ordering.
        self._storage.delete(document_id)

    async def _commit_new(self, document: Document, course_id: uuid.UUID) -> None:
        """Commit the insert, translating the two constraints a client can trip.

        As with courses, there is deliberately no "has this file been uploaded
        already?" query first: two concurrent uploads of the same PDF would
        both pass it. The UNIQUE index is evaluated atomically, so attempting
        the write and handling the rejection is the only correct order.
        """
        content_hash = document.content_hash
        try:
            await self._session.commit()
        except IntegrityError as error:
            await self._session.rollback()
            constraint = violated_constraint(error)
            if constraint == DOCUMENT_CONTENT_HASH_UNIQUE:
                raise ConflictError(
                    f"a document with content hash {content_hash} "
                    f"has already been uploaded to this course"
                ) from error
            if constraint == DOCUMENT_COURSE_FOREIGN_KEY:
                # The course was deleted between the check above and this
                # insert. The client's view of the world is stale, not wrong.
                raise NotFoundError(f"no course with id {course_id}") from error
            raise


def _require_pdf(staged: StagedUpload) -> None:
    """Reject anything that is not a PDF, by its bytes.

    The declared Content-Type is not consulted. Browsers routinely send
    ``application/octet-stream`` for a file picked off disk, so trusting the
    header would reject valid uploads, and a header can be set to anything at
    all, so it would also accept invalid ones. The magic number is the only
    part of the claim the client cannot get wrong for free.
    """
    if staged.byte_size == 0:
        raise UnsupportedMediaError("the uploaded file is empty")
    if not staged.header.startswith(PDF_MAGIC):
        raise UnsupportedMediaError("only PDF files can be uploaded")


def _safe_filename(raw: str | None) -> str:
    """Reduce a client-supplied filename to something safe to store and echo.

    This value is only ever a label - the path on disk comes from the document
    id - but it is rendered in a UI and offered as a download name, so path
    separators and control characters have to go. Backslashes are handled as
    well as forward ones, because a Windows client sends the whole local path.
    """
    candidate = (raw or "").replace("\\", "/")
    # .name drops every directory component, so "../../etc/passwd" becomes
    # "passwd" and a bare "../" becomes "".
    collapsed = " ".join(PurePosixPath(candidate).name.split())
    cleaned = "".join(character for character in collapsed if character.isprintable())
    # "." and ".." survive the steps above and are not usable names.
    if not cleaned or set(cleaned) == {"."}:
        return FALLBACK_FILENAME
    return cleaned[:FILENAME_MAX_LENGTH]
