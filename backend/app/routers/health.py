"""Liveness and readiness endpoints.

The split matters operationally: ``/health`` answers "is this process alive?"
and deliberately touches nothing external, so a database hiccup cannot get a
healthy container killed. ``/health/ready`` answers "should traffic be sent
here?" and returns 503 when a dependency is unusable.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas import ComponentStatus, LivenessResponse, ReadinessResponse
from app.services.health import ReadinessReport, check_database, check_pgvector

router = APIRouter(tags=["health"])


@router.get("/health", response_model=LivenessResponse, summary="Liveness")
async def liveness(request: Request) -> LivenessResponse:
    return LivenessResponse(
        service=request.app.title,
        version=request.app.version,
    )


async def get_readiness_report(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ReadinessReport:
    """Run every dependency probe.

    Exposed as a dependency so tests can inject a healthy or failing report
    without a database, and so later checks (an ``embedder`` entry once the
    model is loaded at startup) are added in exactly one place.
    """
    return ReadinessReport(
        checks={
            "db": await check_database(session),
            "pgvector": await check_pgvector(session),
        }
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def readiness(
    report: Annotated[ReadinessReport, Depends(get_readiness_report)],
    response: Response,
) -> ReadinessResponse:
    if not report.ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ok" if report.ok else "error",
        checks={
            name: ComponentStatus(status=check.status, detail=check.detail)
            for name, check in report.checks.items()
        },
    )
