"""Queries against the ``chunks`` table, including vector search.

Chunks are derived data: they are always written as a complete set for one
document and never edited in place, so this repository deals in replacements
rather than in individual rows.

The nearest-neighbour query lives here rather than in the service for the same
reason every other statement does - it is SQL, and the layer above it should be
readable without knowing what ``<=>`` means.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document

# Size of the candidate list HNSW keeps while descending the graph. The default
# of 40 is tuned for an unfiltered top-k; this query filters by course, and
# pgvector applies that filter *after* the index has chosen its candidates, so
# a course whose chunks are not in the global top 40 would come back short or
# empty. Over-fetching is the cheap fix at this scale. (pgvector 0.8 can
# instead resume the scan until k rows survive the filter, via
# hnsw.iterative_scan - the thing to reach for when this stops being enough.)
HNSW_EF_SEARCH = 100


@dataclass(frozen=True, slots=True)
class ChunkMatch:
    """A retrieved chunk, its document, and how close it was to the query.

    The document travels with the chunk because every consumer needs it and
    fetching it separately would be a query per result. It is also the whole
    point: the answer path reads the filename and the page range off these rows
    rather than from anything a language model produced.
    """

    chunk: Chunk
    document: Document
    distance: float

    @property
    def score(self) -> float:
        """Cosine similarity, where 1.0 is identical and 0.0 is unrelated.

        Reported instead of the raw distance because a "score" that gets better
        as it gets smaller is a trap for everyone downstream. Vectors are
        normalised at creation, so this is also their dot product.
        """
        return 1.0 - self.distance


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_for_document(self, document_id: uuid.UUID, chunks: Sequence[Chunk]) -> None:
        """Swap a document's chunks for a new set, within the caller's transaction.

        The delete is what makes re-ingesting a document safe to do twice:
        without it a retry would collide with the previous attempt's rows on
        ``uq_chunks_document_id_ordinal``. Both statements are staged in one
        transaction, so a reader never sees the document with no chunks at all.

        Issued as a bulk DELETE rather than by loading and deleting each row,
        because there is nothing in Python that needs to see them go.
        """
        await self._session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        self._session.add_all(chunks)

    async def list_for_document(
        self, document_id: uuid.UUID, *, limit: int, offset: int
    ) -> Sequence[Chunk]:
        # Reading order, which is also what uq_chunks(document_id, ordinal)
        # indexes, so this is an index scan rather than a sort.
        result = await self._session.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.ordinal)
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def count_for_document(self, document_id: uuid.UUID) -> int:
        result = await self._session.execute(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        )
        return result.scalar_one()

    async def search_by_embedding(
        self, course_id: uuid.UUID, embedding: Sequence[float], *, limit: int
    ) -> list[ChunkMatch]:
        """The ``limit`` chunks in this course closest to ``embedding``.

        Chunks with no vector are excluded rather than treated as distant: a
        NULL embedding means "not embedded yet", and pgvector would order it
        last anyway, but saying so makes the intent legible and keeps the row
        out of a result set that claims to be ranked by relevance.

        Scoped to one course by joining ``documents``. Retrieval across courses
        would put a chunk from an unrelated unit into an answer's citations,
        which is a correctness problem and not a ranking one.
        """
        # Transaction-local, so it cannot leak into another request that shares
        # this connection. SQLAlchemy has already begun a transaction by the
        # time this runs, which is what makes LOCAL meaningful.
        await self._session.execute(text(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH:d}"))

        # Cosine distance, matching the opclass ix_chunks_embedding_hnsw was
        # built with. An index built for one distance function is silently
        # never used by a query asking for another.
        distance = Chunk.embedding.cosine_distance(embedding)
        result = await self._session.execute(
            select(Chunk, Document, distance.label("distance"))
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.course_id == course_id, Chunk.embedding.is_not(None))
            # The expression again rather than the label above: ORDER BY has to
            # be the indexable form for the HNSW scan to be a candidate at all.
            .order_by(distance)
            .limit(limit)
        )
        return [
            ChunkMatch(chunk=chunk, document=document, distance=float(value))
            for chunk, document, value in result.all()
        ]
