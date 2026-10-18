"""Pydantic DTOs for the HTTP boundary.

Kept separate from any future ORM models: what the API promises and what the
database stores are allowed to diverge.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from app.models import DocumentStatus

HealthStatus = Literal["ok", "error"]


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str


class ComponentStatus(BaseModel):
    status: HealthStatus
    detail: str | None = Field(
        default=None,
        description="Human-readable context; populated on failure and for extension versions.",
    )


class ReadinessResponse(BaseModel):
    status: HealthStatus = Field(description="'ok' only if every check below is 'ok'.")
    checks: dict[str, ComponentStatus] = Field(
        description="One entry per dependency, keyed by name (e.g. 'db', 'pgvector')."
    )


# --- courses -----------------------------------------------------------------

# Anchored, so the pattern must match the whole value. Applied *after*
# normalisation below, hence upper case only. Spaces are allowed because real
# codes look like "MATH 214" as often as "CS229".
COURSE_CODE_PATTERN = r"^[A-Z0-9][A-Z0-9 ._-]*$"

CODE_MAX_LENGTH = 32
TITLE_MAX_LENGTH = 200
DESCRIPTION_MAX_LENGTH = 2000

DEFAULT_PAGE_SIZE = 50
# A caller asking for everything gets a bounded answer instead of a query that
# tries to serialise the whole table.
MAX_PAGE_SIZE = 200


def _normalise_code(value: Any) -> Any:
    """Trim, collapse internal whitespace, upper-case.

    This is what makes "cs229" and " CS229 " the same course without needing a
    case-insensitive column type, so the plain UNIQUE index on ``courses.code``
    is sufficient. Note that "CS 229" stays distinct from "CS229": internal
    spacing is preserved because it is meaningful in real course codes.

    Non-strings are passed through untouched so the string schema can report a
    type error rather than this raising an AttributeError.
    """
    if not isinstance(value, str):
        return value
    return " ".join(value.split()).upper()


CourseCode = Annotated[
    str,
    Field(
        min_length=1,
        max_length=CODE_MAX_LENGTH,
        pattern=COURSE_CODE_PATTERN,
        examples=["CS229"],
    ),
]
CourseTitle = Annotated[
    str,
    Field(min_length=1, max_length=TITLE_MAX_LENGTH, examples=["Machine Learning"]),
]
CourseDescription = Annotated[str | None, Field(max_length=DESCRIPTION_MAX_LENGTH)]


class CourseCreate(BaseModel):
    # A misspelled field name is a client bug; silently ignoring it means the
    # caller thinks it set something it did not.
    model_config = ConfigDict(extra="forbid")

    code: CourseCode
    title: CourseTitle
    description: CourseDescription = None

    # mode="before" so normalisation happens ahead of the length and pattern
    # checks, which are written to expect an already-normalised value.
    _normalise = field_validator("code", mode="before")(_normalise_code)


class CourseUpdate(BaseModel):
    """PATCH body: every field optional, absent means "leave it alone"."""

    model_config = ConfigDict(extra="forbid")

    code: CourseCode | None = None
    title: CourseTitle | None = None
    description: CourseDescription = None

    _normalise = field_validator("code", mode="before")(_normalise_code)

    @model_validator(mode="after")
    def _reject_empty_and_null_required(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("provide at least one field to update")
        # code and title are NOT NULL in the database, so an explicit null is a
        # client error. description is nullable, where null legitimately means
        # "clear it" - which is exactly the distinction changes() preserves.
        for name in ("code", "title"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        return self

    def changes(self) -> dict[str, Any]:
        """Only the fields the client actually sent.

        ``exclude_unset`` is the whole mechanism behind PATCH semantics: it
        distinguishes an absent key, which must not touch the column, from an
        explicit null, which must set it.
        """
        return self.model_dump(exclude_unset=True)


class CourseRead(BaseModel):
    # Lets FastAPI build this straight off an ORM Course instead of requiring a
    # dict, while still deciding for itself which columns are public.
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    code: str
    title: str
    description: str | None
    created_at: datetime
    updated_at: datetime


class CoursePage(BaseModel):
    """A page of courses plus the totals a client needs to paginate.

    ``total`` is the count of all matching rows, not of ``items``, so a UI can
    render "51-100 of 240" without a second request.
    """

    items: list[CourseRead]
    total: int
    limit: int
    offset: int


# --- documents ---------------------------------------------------------------

FILENAME_MAX_LENGTH = 255

# Chunk text is long, and a page of them is a lot of bytes. Kept well below the
# course page size so listing chunks cannot accidentally return megabytes.
DEFAULT_CHUNK_PAGE_SIZE = 20
MAX_CHUNK_PAGE_SIZE = 100


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    course_id: uuid.UUID
    filename: str
    content_hash: str = Field(description="SHA-256 of the uploaded bytes, hex encoded.")
    byte_size: int
    page_count: int | None = Field(
        default=None, description="Null until extraction has run, or if it failed."
    )
    # The enum from app.models rather than a second copy of the same six
    # strings: the values appear in OpenAPI, and a list that drifts from the
    # database's CHECK constraint would document a status that cannot exist.
    status: DocumentStatus
    error: str | None = Field(
        default=None, description="Why ingestion failed. Null unless status is 'failed'."
    )
    created_at: datetime
    updated_at: datetime


class DocumentPage(BaseModel):
    items: list[DocumentRead]
    total: int
    limit: int
    offset: int


class ChunkRead(BaseModel):
    """A stored passage.

    ``embedding`` is deliberately absent: 384 floats per chunk is a large
    response body that no client has a use for, and it is not part of the
    public contract.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    ordinal: int = Field(description="Position within the document, from 0.")
    page_start: int
    page_end: int
    text: str


class ChunkPage(BaseModel):
    items: list[ChunkRead]
    total: int
    limit: int
    offset: int


# --- search ------------------------------------------------------------------

# Long enough for a real question, short enough to keep in a URL and to stay
# inside the embedding model's window - text past 256 word-piece tokens is
# truncated by the model, so accepting an essay here would silently ignore most
# of it.
QUERY_MAX_LENGTH = 500

# Six, because that is how many passages the answer prompt injects as
# [S1]...[S6]. Search and generation retrieve the same way for the same reason
# the eval harness can measure one and predict the other.
DEFAULT_SEARCH_LIMIT = 6
MAX_SEARCH_LIMIT = 20

# strip_whitespace before min_length, so "   " is a 422 rather than a query for
# a space. FastAPI applies these constraints as part of parameter validation,
# which is why the service never has to normalise what it is handed.
SearchQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=QUERY_MAX_LENGTH),
]


class SearchMatch(BaseModel):
    """One retrieved passage, with everything needed to cite it.

    Flattened rather than nesting a ChunkRead inside a DocumentRead: this is
    the shape a citation is rendered from, and the fields that matter -
    which file, which pages - should not be two levels down. ``document_id``
    is here so a client can fetch the PDF and jump to ``page_start``.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    ordinal: int = Field(description="Position within the document, from 0.")
    page_start: int
    page_end: int
    text: str
    score: float = Field(
        description="Cosine similarity to the query: 1.0 is identical, 0.0 unrelated."
    )


class SearchResponse(BaseModel):
    """Top-k results, which is why there is no ``total`` or ``offset``.

    Paging through a ranking is not a meaningful operation - page two of a
    nearest-neighbour search is the part the ranking exists to exclude - so the
    response reports what was asked for and what came back, and nothing that
    would imply a cursor.
    """

    query: str = Field(description="The query as it was interpreted, after trimming.")
    limit: int
    items: list[SearchMatch]


# --- answers -----------------------------------------------------------------

# Answers are long and carry their cited passages, so a page of them is many
# times the size of a page of courses. Kept at the chunk page size for the same
# reason.
DEFAULT_QUESTION_PAGE_SIZE = 20
MAX_QUESTION_PAGE_SIZE = 100


class QuestionAsk(BaseModel):
    """Request body for asking a question.

    A body rather than query parameters, and a POST rather than a GET, because
    answering writes a log row and costs money - neither of which a request a
    browser may repeat on a back button should do.
    """

    model_config = ConfigDict(extra="forbid")

    # The same constrained string ``/search`` takes: trimmed, non-blank, and
    # bounded by what the embedding model will actually read. The question is
    # embedded before it is answered, so a longer one would be silently
    # truncated at retrieval and then answered in full, which is worse than
    # being refused.
    question: SearchQuery = Field(examples=["When is the second assignment due?"])
    limit: int = Field(
        default=DEFAULT_SEARCH_LIMIT,
        ge=1,
        le=MAX_SEARCH_LIMIT,
        description="How many passages to retrieve and offer the model as [S1]...[Sn].",
    )


class AnswerCitation(BaseModel):
    """One resolved marker: what the model pointed at, described from the rows.

    Every field below ``marker`` is read out of the database at render time -
    the chunk's recorded page range and its document's filename - and none of
    it is anything the model produced. ``marker`` is the label as it appears in
    the answer text, so a client renders footnotes by finding ``[S1]`` in the
    prose and looking up the citation that carries it.

    No ``score``: how close the passage was to the query is a retrieval
    diagnostic, which ``/search`` reports. A citation that has been read back
    out of the log has no score to report, and a field that is present on a
    fresh answer and absent on a stored one is worse than no field.
    """

    marker: str = Field(
        description="The label used in the answer text, e.g. 'S1'.", examples=["S1"]
    )
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_start: int
    page_end: int
    text: str = Field(description="The passage the answer drew on, verbatim.")


class AnswerRead(BaseModel):
    """A question, its answer, and the evidence for it."""

    id: uuid.UUID
    course_id: uuid.UUID
    question: str
    answer: str = Field(
        description=(
            "The answer, with [S#] markers inline. Markers that resolved to no "
            "retrieved passage have already been removed."
        )
    )
    answered: bool = Field(
        description=(
            "False when the course's material did not support an answer. "
            "True implies at least one citation."
        )
    )
    citations: list[AnswerCitation] = Field(
        description="One per distinct marker in the answer, in retrieval order."
    )
    retrieved: int = Field(description="How many passages were offered to the model.")
    dropped_markers: int = Field(
        description="Markers the model invented, which were deleted from the answer."
    )
    model: str | None = Field(
        default=None, description="Null when retrieval found nothing and no model was called."
    )
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int = Field(description="Retrieval, generation and resolution together.")
    created_at: datetime


class AnswerPage(BaseModel):
    items: list[AnswerRead]
    total: int
    limit: int
    offset: int
