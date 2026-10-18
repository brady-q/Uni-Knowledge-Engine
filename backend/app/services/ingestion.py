"""Turning an uploaded PDF into chunks.

This runs *after* the HTTP response has been sent, which dictates two things
the rest of the service layer does not have to think about.

First, it cannot use the request's session: that is closed the moment the
response finishes. ``ingest_document`` opens its own from the sessionmaker.

Second, it has no caller to raise to. Every failure has to be recorded
somewhere a person will see it, so the pipeline catches everything and writes
the reason onto ``documents.error`` with status ``failed``. A document that
silently stopped halfway would be indistinguishable from one still working.

The stages are also the values of ``DocumentStatus``, committed as each begins,
so ``GET /documents/{id}`` is a real progress indicator rather than a binary.
``ready`` is the only terminal success state, and it means extracted, chunked
*and* embedded: a document that reached it is searchable, which is what the rest
of the system needs the status to promise.

Known limitation, deferred deliberately: this is scheduled with FastAPI
``BackgroundTasks``, so work in flight does not survive an API restart and a
document can be left in ``extracting`` forever. Moving to a durable worker is a
later milestone; a reaper for stuck rows belongs with it.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence

import anyio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.embedding import Embedder
from app.errors import EmbeddingError, IngestionError
from app.models import Chunk, Document, DocumentStatus
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository
from app.services.chunking import TextChunk, chunk_pages
from app.services.extraction import extract_pdf
from app.storage import DocumentStorage

logger = logging.getLogger(__name__)

# What the uploader is told when something failed that is not their fault. The
# real exception goes to the log; a stack trace on a web page is noise to them
# and detail we should not be publishing.
UNEXPECTED_FAILURE_DETAIL = "ingestion failed unexpectedly; see the server logs"

# Embedding is the one stage that can fail for a reason that is nobody's fault
# and will not be the case later. There is no retry, so the actionable
# instruction is to upload the file again.
EMBEDDING_UNAVAILABLE_DETAIL = (
    "the embedding model was unavailable; upload this document again once the API reports ready"
)


async def ingest_document(
    sessionmaker: async_sessionmaker[AsyncSession],
    storage: DocumentStorage,
    embedder: Embedder,
    document_id: uuid.UUID,
) -> None:
    """Entry point for the background task. Owns its session.

    The embedder is passed in rather than read off ``app.state`` because this
    function has no request and no app: it is called with whatever the router's
    dependency resolved, which is what lets a test substitute a fake.
    """
    async with sessionmaker() as session:
        await IngestionService(session, storage, embedder).run(document_id)


class IngestionService:
    def __init__(self, session: AsyncSession, storage: DocumentStorage, embedder: Embedder) -> None:
        self._session = session
        self._storage = storage
        self._embedder = embedder
        self._documents = DocumentRepository(session)
        self._chunks = ChunkRepository(session)

    async def run(self, document_id: uuid.UUID) -> None:
        """Extract, chunk, store. Never raises."""
        document = await self._documents.get(document_id)
        if document is None:
            # Deleted between upload and this task being picked up. Nothing to
            # do, and nothing wrong.
            logger.info("document %s no longer exists; skipping ingestion", document_id)
            return

        try:
            await self._pipeline(document)
        except Exception as exc:  # noqa: BLE001 - the failure has to reach the row
            await self._record_failure(document_id, exc)
            return

        logger.info("ingested document %s (%s pages)", document_id, document.page_count)

    async def _pipeline(self, document: Document) -> None:
        await self._advance(document, DocumentStatus.EXTRACTING)

        # PyMuPDF is synchronous and CPU-bound. Running it inline would block
        # the event loop - and therefore every other request this worker is
        # serving - for as long as the PDF takes to parse.
        extracted = await anyio.to_thread.run_sync(extract_pdf, self._storage.path_for(document.id))
        document.page_count = extracted.page_count
        await self._advance(document, DocumentStatus.CHUNKING)

        chunks = chunk_pages(extracted.pages)
        if not chunks:
            raise IngestionError("no text could be chunked from this document")
        await self._advance(document, DocumentStatus.EMBEDDING)

        embeddings = await self._embed([chunk.text for chunk in chunks])
        # Staged with their vectors already attached, so the chunks are
        # inserted once rather than inserted empty and then updated row by row.
        # It also means a document is never briefly visible as a set of chunks
        # that retrieval would skip.
        await self._chunks.replace_for_document(document.id, _rows(document.id, chunks, embeddings))

        # Cleared rather than left behind: a successful re-ingest of a
        # previously failed document must not keep showing the old reason.
        document.error = None
        await self._advance(document, DocumentStatus.READY)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Vectorise every chunk, translating an unusable model into a reason.

        ``EmbeddingError`` is written for an HTTP caller, and this code path has
        none. Recording it as an ingestion failure puts a sentence the uploader
        can act on onto the row, instead of the generic "see the server logs"
        that every genuinely unexpected exception gets.
        """
        try:
            return await self._embedder.embed_documents(texts)
        except EmbeddingError as exc:
            raise IngestionError(EMBEDDING_UNAVAILABLE_DETAIL) from exc

    async def _advance(self, document: Document, status: DocumentStatus) -> None:
        """Publish the current stage.

        Committed per stage rather than once at the end so progress is visible
        while a long PDF is still being processed, and so a crash leaves the
        document at the stage that killed it instead of at ``pending``.
        """
        document.status = status
        await self._session.commit()

    async def _record_failure(self, document_id: uuid.UUID, exc: Exception) -> None:
        """Write the reason onto the document, in a session known to be clean.

        The failing statement may have poisoned the transaction, and the
        instance held above is expired by the rollback - touching it would
        trigger a lazy refresh, which under asyncio raises MissingGreenlet
        rather than loading. So: roll back, re-read, then write.
        """
        if isinstance(exc, IngestionError):
            detail = exc.detail
            logger.warning("ingestion of %s failed: %s", document_id, detail)
        else:
            detail = UNEXPECTED_FAILURE_DETAIL
            logger.exception("unexpected error ingesting %s", document_id)

        try:
            await self._session.rollback()
            document = await self._documents.get(document_id)
            if document is None:
                return
            document.status = DocumentStatus.FAILED
            document.error = detail
            await self._session.commit()
        except Exception:
            # The database is unreachable, or the row is gone. There is no
            # further fallback; the log is the only remaining record.
            logger.exception("could not record ingestion failure for %s", document_id)


def _rows(
    document_id: uuid.UUID,
    chunks: Sequence[TextChunk],
    embeddings: Sequence[list[float]],
) -> list[Chunk]:
    """Map the chunker's output and its vectors onto ORM rows.

    ``strict=True`` because a mismatch here would pair a chunk with another
    chunk's vector - text that is retrieved for questions about something else
    entirely, with a citation pointing at the wrong page. Nothing about that
    would look like a bug from the outside, so it has to fail loudly.
    """
    return [
        Chunk(
            document_id=document_id,
            ordinal=chunk.ordinal,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text=chunk.text,
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
