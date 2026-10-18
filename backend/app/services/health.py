"""Dependency probes behind the readiness endpoint.

These are plain async functions over an ``AsyncSession`` with no FastAPI
coupling, so they can be unit-tested against a stub session. Each probe
converts failure into a value rather than raising: readiness must be able to
report "db is down" instead of returning a 500.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding import Embedder
from app.llm import AnswerModel

Status = Literal["ok", "error"]

_PING_SQL = text("SELECT 1")
_VECTOR_INSTALLED_SQL = text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
_VECTOR_AVAILABLE_SQL = text(
    "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'"
)


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    status: Status
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    checks: dict[str, ComponentHealth]

    @property
    def ok(self) -> bool:
        return all(check.status == "ok" for check in self.checks.values())


def _describe(exc: Exception, limit: int = 200) -> str:
    """Summarise an exception without leaking a full traceback to the client."""
    return f"{type(exc).__name__}: {exc}"[:limit]


async def check_database(session: AsyncSession) -> ComponentHealth:
    """Confirm the database accepts queries, not merely that a socket opens."""
    try:
        await session.execute(_PING_SQL)
    except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
        return ComponentHealth("error", _describe(exc))
    return ComponentHealth("ok")


async def check_pgvector(session: AsyncSession) -> ComponentHealth:
    """Confirm pgvector is installed, or at least installable on this server.

    The first migration creates the extension; until then it only needs to be
    *available*. Either way, running against a plain `postgres` image is caught
    here instead of by a migration failing much later.
    """
    try:
        installed = (await session.execute(_VECTOR_INSTALLED_SQL)).scalar_one_or_none()
        if installed is not None:
            return ComponentHealth("ok", f"installed {installed}")

        available = (await session.execute(_VECTOR_AVAILABLE_SQL)).scalar_one_or_none()
        if available is not None:
            return ComponentHealth("ok", f"available {available}, not yet installed")
    except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
        return ComponentHealth("error", _describe(exc))
    return ComponentHealth("error", "pgvector is not available on this server")


def check_embedder(embedder: Embedder) -> ComponentHealth:
    """Report whether search can actually be served.

    Synchronous and side-effect free: it inspects the embedder rather than
    embedding anything. A probe that triggered the load would make an
    orchestrator's first readiness check hang for however long the weights take
    to read, and a probe run at intervals would keep a model alive that nothing
    else was using.

    "Still loading" is reported as an error, not as a caveat, because that is
    the honest answer to "should traffic be routed here?" - a search arriving
    now gets a 503. It resolves itself within a few seconds of startup, or
    within a download on the very first run.
    """
    if embedder.load_error is not None:
        return ComponentHealth("error", embedder.load_error)
    if not embedder.is_loaded:
        return ComponentHealth("error", f"{embedder.model_name} is still loading")
    return ComponentHealth("ok", f"loaded {embedder.model_name}")


def check_answer_model(model: AnswerModel) -> ComponentHealth:
    """Report whether a question could be answered at all.

    Configuration only, and deliberately so: this probe does not ask the model
    anything. A readiness check that made a real call would bill somebody every
    few seconds for the privilege of being monitored, and would add the whole
    latency of a generation to a check an orchestrator expects to be instant.

    The consequence is that it cannot tell a valid key from an invalid one -
    only a present one from a missing one. A key that is present but rejected
    is reported by the first question asked, as a 503 that says so.
    """
    error = model.config_error
    if error is not None:
        return ComponentHealth("error", error)
    return ComponentHealth("ok", f"configured for {model.model_name}")
