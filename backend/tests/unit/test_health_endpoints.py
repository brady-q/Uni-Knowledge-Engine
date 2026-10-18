"""HTTP-level behaviour of /health and /health/ready."""

from __future__ import annotations

from fastapi import FastAPI
from httpx import AsyncClient

from app import __version__
from app.routers.health import get_readiness_report
from app.services.health import ComponentHealth, ReadinessReport


async def test_liveness_reports_service_identity(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Uni Knowledge Engine",
        "version": __version__,
    }


async def test_liveness_does_not_depend_on_the_database(unmocked_client: AsyncClient) -> None:
    """Liveness must stay 200 with the database unreachable.

    Otherwise a transient Postgres outage gets a healthy container restarted.
    """
    response = await unmocked_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_readiness_is_200_when_every_check_passes(client: AsyncClient) -> None:
    response = await client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["db"] == {"status": "ok", "detail": None}
    assert body["checks"]["pgvector"]["status"] == "ok"


async def test_readiness_is_503_when_a_check_fails(app: FastAPI, client: AsyncClient) -> None:
    app.dependency_overrides[get_readiness_report] = lambda: ReadinessReport(
        checks={
            "db": ComponentHealth("error", "OperationalError: connection refused"),
            "pgvector": ComponentHealth("ok"),
        }
    )

    response = await client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["checks"]["db"]["status"] == "error"
    assert "connection refused" in body["checks"]["db"]["detail"]
    # A partial failure still reports the healthy components.
    assert body["checks"]["pgvector"]["status"] == "ok"


async def test_readiness_reports_error_against_an_unreachable_database(
    unmocked_client: AsyncClient,
) -> None:
    """End-to-end failure path with the real probes and no test doubles."""
    response = await unmocked_client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["checks"]["db"]["status"] == "error"
    assert body["checks"]["db"]["detail"]


async def test_health_routes_are_documented_and_unversioned(client: AsyncClient) -> None:
    """Probes should not have to follow the versioned API prefix."""
    schema = (await client.get("/openapi.json")).json()

    assert "/health" in schema["paths"]
    assert "/health/ready" in schema["paths"]
    assert schema["info"]["version"] == __version__
