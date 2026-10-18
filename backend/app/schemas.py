"""Pydantic DTOs for the HTTP boundary.

Kept separate from any future ORM models: what the API promises and what the
database stores are allowed to diverge.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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
