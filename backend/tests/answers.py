"""A stand-in for the answer model.

The real one costs money per call, needs a key, and gives a different answer
every time - so no test in the default suite may touch it. The one test that
does is marked ``live`` and lives in ``tests/slow/``; everything else uses this.

It is not a mock of the Anthropic SDK. It implements ``app.llm.AnswerModel``,
which is the interface the answer path actually depends on, and it records what
it was asked. That is what lets a test assert the interesting things - that the
passages handed over contain no page numbers, that an invented marker is
dropped, that a course with no material comes back unanswered - without any of
them depending on what a language model happens to say today.

The ``text`` it returns is set by the test, because the markers in it *are* the
test. ``tests.embeddings.FakeEmbedder`` makes the same trade for retrieval.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from app.llm import AnswerDraft
from app.models import Chunk, Document, DocumentStatus
from app.repositories.chunks import ChunkMatch

# Cites the first passage, which is what a well-behaved answer looks like.
DEFAULT_ANSWER = "The course material addresses this directly [S1]."

FAKE_MODEL_NAME = "fake-claude-0"


class FakeAnswerModel:
    """Implements ``app.llm.AnswerModel`` with no API behind it."""

    def __init__(
        self,
        *,
        text: str = DEFAULT_ANSWER,
        answered: bool = True,
        raises: Exception | None = None,
        config_error: str | None = None,
        input_tokens: int | None = 1_200,
        output_tokens: int | None = 48,
    ) -> None:
        self.model_name = FAKE_MODEL_NAME
        self.text = text
        self.answered = answered
        # What to raise instead of answering, for the paths that only exist
        # because a third-party API can fail.
        self.raises = raises
        self.config_error = config_error
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        # Every (question, passages) pair it was handed, in order.
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def passages(self) -> list[str]:
        """The passages from the most recent call.

        A shortcut for the assertion this fake exists to make possible: that
        what the model was shown carries no filename, page number or id.
        """
        return self.calls[-1][1]

    async def generate(self, question: str, passages: Sequence[str]) -> AnswerDraft:
        self.calls.append((question, list(passages)))
        if self.raises is not None:
            raise self.raises
        return AnswerDraft(
            text=self.text,
            answered=self.answered,
            model=self.model_name,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


def make_match(
    *,
    text: str,
    page_start: int = 3,
    page_end: int | None = None,
    filename: str = "lecture.pdf",
    ordinal: int = 0,
    distance: float = 0.2,
    course_id: uuid.UUID | None = None,
) -> ChunkMatch:
    """A retrieved row, detached, as ``ChunkRepository`` would hand it over.

    Lives here rather than in one of the test modules because both the
    resolver's unit tests and the live test need to run resolution against
    something, and a second copy of this would be a second place for the
    chunk's page range - the thing a citation is made of - to be built wrongly.
    """
    document = Document(
        id=uuid.uuid4(),
        course_id=course_id or uuid.uuid4(),
        filename=filename,
        content_hash="0" * 64,
        byte_size=2048,
        page_count=40,
        status=DocumentStatus.READY,
    )
    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=document.id,
        ordinal=ordinal,
        page_start=page_start,
        page_end=page_end if page_end is not None else page_start,
        text=text,
    )
    return ChunkMatch(chunk=chunk, document=document, distance=distance)
