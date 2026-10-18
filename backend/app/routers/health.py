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
from app.embedding import Embedder, get_embedder
from app.llm import AnswerModel, get_answer_model
from app.schemas import ComponentStatus, LivenessResponse, ReadinessResponse
from app.services.health import (
    ReadinessReport,
    check_answer_model,
    check_database,
    check_embedder,
    check_pgvector,
)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=LivenessResponse, summary="Liveness")
async def liveness(request: Request) -> LivenessResponse:
    return LivenessResponse(
        service=request.app.title,
        version=request.app.version,
    )


async def get_readiness_report(
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    answer_model: Annotated[AnswerModel, Depends(get_answer_model)],
) -> ReadinessReport:
    """Run every dependency probe.

    Exposed as a dependency so tests can inject a healthy or failing report
    without a database and without a model, and so a new check is added in
    exactly one place.
    """
    return ReadinessReport(
        checks={
            "db": await check_database(session),
            "pgvector": await check_pgvector(session),
            "embedder": check_embedder(embedder),
            "claude": check_answer_model(answer_model),
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
