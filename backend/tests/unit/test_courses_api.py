"""HTTP behaviour of the course routes, with no database anywhere.

The service is replaced with an in-memory stand-in, the same way the health
tests replace ``get_readiness_report``. What is under test here is strictly the
router's job: status codes, headers, and the mapping from domain errors onto
responses. The app fixture still points at an unreachable database, so a route
that accidentally reached for a session would fail loudly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.errors import ConflictError, NotFoundError
from app.models import Course
from app.routers.courses import get_course_service
from app.schemas import MAX_PAGE_SIZE, CourseCreate, CourseUpdate

CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def make_course(code: str, title: str, description: str | None = None) -> Course:
    """A detached Course.

    Column defaults are applied by the database on INSERT, so an instance built
    in memory has to be given its id and timestamps explicitly.
    """
    return Course(
        id=uuid.uuid4(),
        code=code,
        title=title,
        description=description,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )


class FakeCourseService:
    """Same surface as CourseService, backed by a dict."""

    def __init__(self) -> None:
        self.courses: dict[uuid.UUID, Course] = {}

    def preload(self, *codes: str) -> None:
        for code in codes:
            course = make_course(code, f"{code} course")
            self.courses[course.id] = course

    async def create(self, payload: CourseCreate) -> Course:
        if any(course.code == payload.code for course in self.courses.values()):
            raise ConflictError(f"course code {payload.code!r} is already in use")
        course = make_course(payload.code, payload.title, payload.description)
        self.courses[course.id] = course
        return course

    async def get(self, course_id: uuid.UUID) -> Course:
        course = self.courses.get(course_id)
        if course is None:
            raise NotFoundError(f"no course with id {course_id}")
        return course

    async def list(self, *, limit: int, offset: int) -> tuple[Sequence[Course], int]:
        ordered = sorted(self.courses.values(), key=lambda course: course.code)
        return ordered[offset : offset + limit], len(ordered)

    async def update(self, course_id: uuid.UUID, payload: CourseUpdate) -> Course:
        course = await self.get(course_id)
        for field, value in payload.changes().items():
            setattr(course, field, value)
        course.updated_at = CREATED_AT + timedelta(seconds=1)
        return course

    async def delete(self, course_id: uuid.UUID) -> None:
        course = await self.get(course_id)
        del self.courses[course.id]


@pytest.fixture
def service() -> FakeCourseService:
    return FakeCourseService()


@pytest.fixture
async def courses_client(app: FastAPI, service: FakeCourseService) -> AsyncIterator[AsyncClient]:
    app.dependency_overrides[get_course_service] = lambda: service
    try:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                yield client
    finally:
        app.dependency_overrides.clear()


async def test_create_returns_201_and_the_stored_representation(
    courses_client: AsyncClient,
) -> None:
    response = await courses_client.post(
        "/api/v1/courses",
        json={"code": "cs229", "title": "Machine Learning", "description": "Stanford"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["code"] == "CS229", "the DTO should have normalised the code"
    assert body["title"] == "Machine Learning"
    assert uuid.UUID(body["id"])


async def test_create_location_header_points_at_the_new_course(
    courses_client: AsyncClient,
) -> None:
    """A 201 is required to say where the created thing now lives."""
    response = await courses_client.post(
        "/api/v1/courses", json={"code": "CS229", "title": "Machine Learning"}
    )

    location = response.headers["location"]
    assert location.endswith(f"/api/v1/courses/{response.json()['id']}")

    # And it must actually resolve, not merely look plausible.
    followed = await courses_client.get(location)
    assert followed.status_code == 200
    assert followed.json()["id"] == response.json()["id"]


async def test_create_conflict_becomes_409(courses_client: AsyncClient) -> None:
    payload = {"code": "CS229", "title": "Machine Learning"}
    await courses_client.post("/api/v1/courses", json=payload)

    response = await courses_client.post("/api/v1/courses", json=payload)

    assert response.status_code == 409
    assert "already in use" in response.json()["detail"]


async def test_create_rejects_an_invalid_body_before_the_service_sees_it(
    courses_client: AsyncClient, service: FakeCourseService
) -> None:
    response = await courses_client.post("/api/v1/courses", json={"code": "-nope", "title": "x"})

    assert response.status_code == 422
    assert service.courses == {}


async def test_get_unknown_course_becomes_404(courses_client: AsyncClient) -> None:
    response = await courses_client.get(f"/api/v1/courses/{uuid.uuid4()}")

    assert response.status_code == 404
    assert "no course with id" in response.json()["detail"]


async def test_get_rejects_a_malformed_id(courses_client: AsyncClient) -> None:
    """Path validation, so a bad id never becomes a database round trip."""
    response = await courses_client.get("/api/v1/courses/not-a-uuid")

    assert response.status_code == 422


async def test_list_is_paginated_and_ordered(
    courses_client: AsyncClient, service: FakeCourseService
) -> None:
    service.preload("PHYS101", "CS229", "MATH214")

    first = (await courses_client.get("/api/v1/courses?limit=2&offset=0")).json()
    second = (await courses_client.get("/api/v1/courses?limit=2&offset=2")).json()

    assert [item["code"] for item in first["items"]] == ["CS229", "MATH214"]
    assert [item["code"] for item in second["items"]] == ["PHYS101"]
    # total counts every match, not just this page, so a client can paginate.
    assert first["total"] == second["total"] == 3
    assert (first["limit"], first["offset"]) == (2, 0)


@pytest.mark.parametrize("query", ["limit=0", f"limit={MAX_PAGE_SIZE + 1}", "offset=-1"])
async def test_list_rejects_out_of_range_paging(courses_client: AsyncClient, query: str) -> None:
    """An unbounded limit would let a caller ask for the whole table."""
    assert (await courses_client.get(f"/api/v1/courses?{query}")).status_code == 422


async def test_patch_updates_only_what_was_sent(courses_client: AsyncClient) -> None:
    created = (
        await courses_client.post(
            "/api/v1/courses",
            json={"code": "CS229", "title": "Machine Learning", "description": "Stanford"},
        )
    ).json()

    response = await courses_client.patch(f"/api/v1/courses/{created['id']}", json={"title": "ML"})

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "ML"
    assert body["code"] == created["code"]
    assert body["description"] == created["description"]


async def test_patch_unknown_course_becomes_404(courses_client: AsyncClient) -> None:
    response = await courses_client.patch(f"/api/v1/courses/{uuid.uuid4()}", json={"title": "x"})

    assert response.status_code == 404


async def test_delete_returns_204_with_no_body(courses_client: AsyncClient) -> None:
    created = (
        await courses_client.post("/api/v1/courses", json={"code": "CS229", "title": "ML"})
    ).json()

    response = await courses_client.delete(f"/api/v1/courses/{created['id']}")

    assert response.status_code == 204
    assert not response.content
    assert (await courses_client.get(f"/api/v1/courses/{created['id']}")).status_code == 404


async def test_delete_unknown_course_becomes_404(courses_client: AsyncClient) -> None:
    """Not idempotent-by-silence: the caller is told the id was already gone."""
    response = await courses_client.delete(f"/api/v1/courses/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_course_routes_are_versioned_and_documented(courses_client: AsyncClient) -> None:
    schema = (await courses_client.get("/openapi.json")).json()

    assert "/api/v1/courses" in schema["paths"]
    assert "/api/v1/courses/{course_id}" in schema["paths"]
    # The failure modes a client has to handle should be discoverable in /docs.
    assert "409" in schema["paths"]["/api/v1/courses"]["post"]["responses"]
    assert "404" in schema["paths"]["/api/v1/courses/{course_id}"]["get"]["responses"]
