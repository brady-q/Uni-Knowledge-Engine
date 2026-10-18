"""Cited answers: retrieve, generate, resolve, log.

This is the module the whole project exists to make defensible, so it is worth
being explicit about where the guarantee lives. Four steps:

1. **Retrieve.** ``RetrievalService`` ranks the course's chunks. Reused rather
   than reimplemented, so what ``/search`` shows is what an answer is built
   from, and the eval harness can measure one and predict the other.
2. **Generate.** The chunks' *text* is handed to the model as ``[S1]…[S6]``.
   Nothing else goes with it - no filename, no page number, no id - so there is
   no page number in the model's context for it to repeat, misremember or
   invent. See ``app.llm``.
3. **Resolve.** Every marker in the returned prose is looked up in the
   retrieved set. A marker that names a passage becomes a citation whose
   document and page range are read off the stored rows. A marker that names
   nothing is deleted from the answer and counted.
4. **Log.** The question, the answer and one row per surviving citation are
   committed together.

The part that is easy to get wrong is step 3's failure case, so it is decided
here rather than left to a caller: if *no* marker survives but the model
claimed to have answered, the answer is discarded and replaced with a plain
statement that the material does not support one. An uncited claim is exactly
what this system promises not to serve, and serving it because the prose
sounded confident would be the whole guarantee undone by a fallback.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import violated_constraint
from app.embedding import Embedder
from app.errors import ConflictError, NotFoundError
from app.llm import MARKER_PATTERN, AnswerModel
from app.models import (
    CITATION_CHUNK_FOREIGN_KEY,
    QUESTION_COURSE_FOREIGN_KEY,
    Chunk,
    Citation,
    Document,
    Question,
)
from app.repositories.chunks import ChunkMatch
from app.repositories.courses import CourseRepository
from app.repositories.questions import QuestionRepository
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

# Retrieval came back empty, so no prompt was worth sending. Says both things
# that could be true, because from here they are indistinguishable: there is
# nothing about this in the course, or the documents are not searchable yet.
NO_SOURCES_ANSWER = (
    "Nothing in this course's documents covers that question. "
    "If the documents were only just uploaded, they may still be processing."
)

# Every marker the model wrote was invented, so nothing it said is supported.
# This replaces the answer rather than annotating it.
UNSUPPORTED_ANSWER = (
    "The material retrieved for that question does not support an answer. "
    "Try rephrasing it, or asking about something the course covers more directly."
)

# What a caller is told when the corpus shifted under a request that had
# already been paid for. Rare - it needs a re-ingest or a delete to land in the
# seconds between retrieval and the log write - and retrying really does fix
# it, because the second attempt retrieves the new chunks.
CORPUS_CHANGED_DETAIL = (
    "this course's documents changed while the question was being answered; ask again"
)


@dataclass(frozen=True, slots=True)
class ResolvedCitation:
    """A marker, and the rows it resolves to.

    The document travels alongside the chunk rather than being reached through
    ``chunk.document``, so that every consumer is handed something already
    loaded. Nothing here is derived from the model's output except ``marker``.
    """

    marker: int
    chunk: Chunk
    document: Document


@dataclass(frozen=True, slots=True)
class Answer:
    """A logged question and its resolved citations.

    Both halves are needed to render one answer, and keeping them together is
    what lets a freshly generated answer and one read back out of the log go
    through the same serialisation - which is in turn what makes it impossible
    for the two to disagree about what a citation is.
    """

    question: Question
    citations: list[ResolvedCitation]


def resolve_markers(
    text: str, matches: Sequence[ChunkMatch]
) -> tuple[str, list[ResolvedCitation], int]:
    """Map ``[S#]`` markers onto retrieved rows, deleting the ones that miss.

    Returns the answer with invented markers removed, the citations in marker
    order, and how many marker *occurrences* were dropped.

    Marker order is retrieval order - passage ``n`` is always the ``n``-th
    ranked chunk - so a citation list is ordered by relevance without anything
    having to sort it, and it does not depend on where in the prose the model
    happened to mention things first.

    A marker used twice yields one citation: the second mention is the same
    source, and a UI numbering footnotes should not show it twice. It is
    deliberately left in the prose, though, because that is where the claim it
    supports is.
    """
    cited: dict[int, ResolvedCitation] = {}
    dropped = 0

    def keep_or_drop(found: re.Match[str]) -> str:
        nonlocal dropped
        index = int(found.group("index"))
        if 1 <= index <= len(matches):
            match = matches[index - 1]
            cited.setdefault(
                index,
                ResolvedCitation(marker=index, chunk=match.chunk, document=match.document),
            )
            # Verbatim, including the whitespace the pattern absorbed.
            return found.group(0)
        dropped += 1
        # Logged at warning because this is the event the design is aimed at:
        # if it is common, either the prompt or the model needs work, and
        # nobody will notice from the answers themselves - by the time they are
        # read, the evidence has been removed.
        logger.warning(
            "dropping invented marker [S%d]: only %d passage(s) were supplied", index, len(matches)
        )
        return ""

    return (
        MARKER_PATTERN.sub(keep_or_drop, text).strip(),
        [cited[key] for key in sorted(cited)],
        dropped,
    )


class AnswerService:
    def __init__(self, session: AsyncSession, embedder: Embedder, model: AnswerModel) -> None:
        self._session = session
        self._model = model
        self._retrieval = RetrievalService(session, embedder)
        self._courses = CourseRepository(session)
        self._questions = QuestionRepository(session)

    async def ask(self, course_id: uuid.UUID, question: str, *, limit: int) -> Answer:
        """Answer a question about a course and log what happened.

        Raises ``NotFoundError`` for an unknown course, ``EmbeddingError`` if
        the query cannot be embedded, and ``GenerationError`` if the model
        cannot be reached or returns nothing usable. A question that the course
        simply does not cover is not an error: it comes back with
        ``answered=False`` and no citations, and is logged like any other.
        """
        started = time.perf_counter()
        matches = await self._retrieval.search(course_id, question, limit=limit)

        # End the read transaction before the model call. Generation takes
        # seconds, and holding a pooled connection open across it would let a
        # handful of concurrent questions exhaust the pool while doing no
        # database work at all. Committing rather than rolling back is what
        # keeps the rows above usable: with expire_on_commit=False (see
        # app.db) a commit leaves loaded objects populated, where a rollback
        # would expire them and the next attribute read would be blocking IO
        # inside an async request.
        await self._session.commit()

        if matches:
            draft = await self._model.generate(question, [match.chunk.text for match in matches])
            answer, citations, dropped = resolve_markers(draft.text, matches)
            answered = draft.answered
            model, input_tokens, output_tokens = (
                draft.model,
                draft.input_tokens,
                draft.output_tokens,
            )
            if answered and not citations:
                # See the module docstring: confident and uncited is the one
                # combination that must never reach a reader.
                logger.warning(
                    "discarding an uncited answer for course %s (%d marker(s) dropped)",
                    course_id,
                    dropped,
                )
                answer, answered = UNSUPPORTED_ANSWER, False
        else:
            answer, citations, dropped = NO_SOURCES_ANSWER, [], 0
            answered = False
            # NULL, because no model was called - which is a different fact
            # from a model that was called and answered nothing.
            model, input_tokens, output_tokens = None, None, None

        row = Question(
            course_id=course_id,
            question=question,
            answer=answer,
            answered=answered,
            retrieved=len(matches),
            dropped_markers=dropped,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
            # Assigned through the relationship so the rows are inserted by
            # the same flush, in the same transaction, as the question itself.
            citations=[
                Citation(marker=citation.marker, chunk=citation.chunk) for citation in citations
            ],
        )
        self._questions.add(row)
        await self._commit(row, course_id)
        return Answer(question=row, citations=citations)

    async def get(self, course_id: uuid.UUID, question_id: uuid.UUID) -> Answer:
        question = await self._questions.get_in_course(course_id, question_id)
        if question is None:
            raise NotFoundError(f"no question with id {question_id} in course {course_id}")
        return _from_row(question)

    async def list(
        self, course_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[Answer], int]:
        if await self._courses.get(course_id) is None:
            raise NotFoundError(f"no course with id {course_id}")
        # Both statements in one transaction, so the page and its total
        # describe the same snapshot.
        questions = await self._questions.list_for_course(course_id, limit=limit, offset=offset)
        return [_from_row(question) for question in questions], (
            await self._questions.count_for_course(course_id)
        )

    async def _commit(self, row: Question, course_id: uuid.UUID) -> None:
        """Commit the log row, translating the two references that can vanish.

        Both are races against a request that took seconds, and both are the
        client's view of the world being stale rather than wrong - so neither
        is a 500.
        """
        try:
            await self._session.commit()
        except IntegrityError as error:
            await self._session.rollback()
            constraint = violated_constraint(error)
            if constraint == QUESTION_COURSE_FOREIGN_KEY:
                raise NotFoundError(f"no course with id {course_id}") from error
            if constraint == CITATION_CHUNK_FOREIGN_KEY:
                # A document was deleted or re-ingested while this answer was
                # being written, so a chunk it cites no longer exists. Storing
                # the citation anyway is exactly what the foreign key is there
                # to prevent.
                logger.info("citations for question in course %s outlived their chunks", course_id)
                raise ConflictError(CORPUS_CHANGED_DETAIL) from error
            raise


def _from_row(question: Question) -> Answer:
    """Rebuild an answer from the log.

    Reads the filename and page range through ``citation.chunk.document``, the
    same way the generated answer did, so a citation shown a week later still
    describes the document as it is now rather than as it was. The
    relationships are eagerly loaded by the repository; this function does no
    IO and must not start any.
    """
    return Answer(
        question=question,
        citations=[
            ResolvedCitation(
                marker=citation.marker, chunk=citation.chunk, document=citation.chunk.document
            )
            for citation in question.citations
        ],
    )
