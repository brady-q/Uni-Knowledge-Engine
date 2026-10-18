"""Validation at the HTTP boundary.

These are the rules that must hold before anything reaches the database, so
they are worth testing directly rather than only through a request.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import CODE_MAX_LENGTH, CourseCreate, CourseUpdate


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("cs229", "CS229"),
        ("  CS229  ", "CS229"),
        ("math 214", "MATH 214"),
        # Internal runs of whitespace collapse, so a stray double space cannot
        # create a second course that looks identical to a human.
        ("MATH   214", "MATH 214"),
        ("\tcs106b\n", "CS106B"),
    ],
)
def test_code_is_normalised(given: str, expected: str) -> None:
    """Normalising here is what lets courses.code use a plain UNIQUE index."""
    assert CourseCreate(code=given, title="x").code == expected


def test_normalisation_preserves_meaningful_spacing() -> None:
    """ "MATH 214" and "MATH214" are different codes, and both are legitimate."""
    assert (
        CourseCreate(code="math 214", title="x").code
        != CourseCreate(code="math214", title="x").code
    )


@pytest.mark.parametrize(
    "code",
    [
        "",
        "   ",
        "-CS229",  # must start alphanumeric
        "CS/229",  # slash would be ambiguous in a URL path
        "CS229!",
        "x" * (CODE_MAX_LENGTH + 1),
    ],
)
def test_invalid_codes_are_rejected(code: str) -> None:
    with pytest.raises(ValidationError):
        CourseCreate(code=code, title="x")


def test_empty_title_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CourseCreate(code="CS229", title="")


def test_unknown_fields_are_rejected() -> None:
    """A typo'd field name must fail loudly, not be silently discarded."""
    with pytest.raises(ValidationError):
        CourseCreate(code="CS229", title="x", titel="typo")  # type: ignore[call-arg]


def test_update_reports_only_the_fields_that_were_sent() -> None:
    """The mechanism behind PATCH: an absent key must not touch its column."""
    assert CourseUpdate(title="renamed").changes() == {"title": "renamed"}


def test_update_distinguishes_absent_from_explicit_null() -> None:
    """description is nullable, so an explicit null legitimately clears it."""
    assert CourseUpdate(description=None).changes() == {"description": None}
    assert "description" not in CourseUpdate(title="x").changes()


def test_update_rejects_null_for_non_nullable_columns() -> None:
    """code and title are NOT NULL, so null is a client error, not a clear."""
    for field in ("code", "title"):
        with pytest.raises(ValidationError, match="cannot be null"):
            CourseUpdate(**{field: None})


def test_update_rejects_an_empty_body() -> None:
    """A PATCH that changes nothing is a mistake worth reporting."""
    with pytest.raises(ValidationError, match="at least one field"):
        CourseUpdate()


def test_update_normalises_code_like_create_does() -> None:
    assert CourseUpdate(code="cs229").changes() == {"code": "CS229"}
