"""Unit tests for the readiness probes, using stub dependencies.

The probes must convert any failure into a value; if one raises, the readiness
endpoint degrades from an informative 503 into an opaque 500.
"""

from __future__ import annotations

from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding import Embedder
from app.services.health import check_database, check_embedder, check_pgvector


class _StubResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value


class StubSession:
    """Returns queued results in order, or raises a queued exception."""

    def __init__(self, *results: Any) -> None:
        self._results = list(results)
        self.executed = 0

    async def execute(self, statement: Any) -> _StubResult:
        self.executed += 1
        outcome = self._results.pop(0) if self._results else None
        if isinstance(outcome, Exception):
            raise outcome
        return _StubResult(outcome)


def as_session(stub: StubSession) -> AsyncSession:
    return cast(AsyncSession, stub)


async def test_check_database_ok() -> None:
    result = await check_database(as_session(StubSession(1)))

    assert result.status == "ok"
    assert result.detail is None


async def test_check_database_converts_failure_into_a_value() -> None:
    result = await check_database(as_session(StubSession(OSError("connection refused"))))

    assert result.status == "error"
    assert result.detail is not None
    assert "OSError" in result.detail
    assert "connection refused" in result.detail


async def test_check_database_truncates_long_error_details() -> None:
    result = await check_database(as_session(StubSession(OSError("x" * 5000))))

    assert result.detail is not None
    assert len(result.detail) <= 200


async def test_check_pgvector_reports_installed_version() -> None:
    result = await check_pgvector(as_session(StubSession("0.8.1")))

    assert result.status == "ok"
    assert result.detail == "installed 0.8.1"


async def test_check_pgvector_accepts_available_but_not_installed() -> None:
    """Before migration 0001 the extension exists on the server but is not created."""
    session = StubSession(None, "0.8.1")

    result = await check_pgvector(as_session(session))

    assert result.status == "ok"
    assert result.detail is not None
    assert "not yet installed" in result.detail
    assert session.executed == 2


async def test_check_pgvector_errors_when_extension_is_absent() -> None:
    """Catches running against a plain postgres image instead of pgvector/pgvector."""
    result = await check_pgvector(as_session(StubSession(None, None)))

    assert result.status == "error"
    assert result.detail == "pgvector is not available on this server"


# --- the embedder -------------------------------------------------------------


class StubEmbedder:
    """Only the three members ``check_embedder`` is allowed to look at."""

    def __init__(self, *, is_loaded: bool, load_error: str | None = None) -> None:
        self.model_name = "stub/model"
        self.is_loaded = is_loaded
        self.load_error = load_error


def test_check_embedder_reports_a_loaded_model() -> None:
    result = check_embedder(cast(Embedder, StubEmbedder(is_loaded=True)))

    assert result.status == "ok"
    assert result.detail == "loaded stub/model"


def test_check_embedder_reports_a_model_that_is_still_loading() -> None:
    """Not ready is the honest answer: a search arriving now gets a 503."""
    result = check_embedder(cast(Embedder, StubEmbedder(is_loaded=False)))

    assert result.status == "error"
    assert result.detail is not None
    assert "still loading" in result.detail


def test_check_embedder_surfaces_why_a_load_failed() -> None:
    """The reason is the whole value of the check - "not ready" alone does not
    distinguish a slow first download from a machine with no network."""
    result = check_embedder(
        cast(Embedder, StubEmbedder(is_loaded=False, load_error="OSError: no route to host"))
    )

    assert result.status == "error"
    assert result.detail == "OSError: no route to host"


def test_check_embedder_does_not_touch_the_model() -> None:
    """It inspects, it does not embed. A probe that triggered the load would
    hang the first readiness check for as long as the weights take to read."""
    embedder = StubEmbedder(is_loaded=False)

    check_embedder(cast(Embedder, embedder))

    assert embedder.is_loaded is False
