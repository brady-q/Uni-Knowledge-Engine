"""Pydantic DTOs for the HTTP boundary.

Kept separate from any future ORM models: what the API promises and what the
database stores are allowed to diverge.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
