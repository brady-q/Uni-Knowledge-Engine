"""Course CRUD against a real Postgres.

The unit tests prove the router's contract with a fake service. These prove the
parts a fake cannot: that the SQL is valid, that the constraints in migration
0001 actually fire, and that the ORM writes what we think it writes.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Course, Document, DocumentStatus

pytestmark = pytest.mark.integration


async def test_full_lifecycle(api_client: AsyncClient) -> None:
    created = await api_client.post(
        "/api/v1/courses",
        json={"code": "cs229", "title": "Machine Learning", "description": "Stanford"},
    )
    assert created.status_code == 201, created.text
    course_id = created.json()["id"]

    fetched = await api_client.get(f"/api/v1/courses/{course_id}")
    assert fetched.status_code == 200
    assert fetched.json() == created.json()

    listed = await api_client.get("/api/v1/courses")
    assert [item["id"] for item in listed.json()["items"]] == [course_id]
    assert listed.json()["total"] == 1

    patched = await api_client.patch(f"/api/v1/courses/{course_id}", json={"title": "ML"})
    assert patched.status_code == 200
    assert patched.json()["title"] == "ML"

    assert (await api_client.delete(f"/api/v1/courses/{course_id}")).status_code == 204
    assert (await api_client.get(f"/api/v1/courses/{course_id}")).status_code == 404
    assert (await api_client.get("/api/v1/courses")).json()["total"] == 0


async def test_the_database_populates_timestamps(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """created_at/updated_at come from server defaults, fetched back eagerly.

    If eager_defaults were off, these would come back expired and serialising
    the response would need another SELECT - or fail outright under asyncio.
    """
    body = (await api_client.post("/api/v1/courses", json={"code": "CS229", "title": "ML"})).json()

    assert body["created_at"] is not None
    assert body["created_at"] == body["updated_at"]

    stored = (await db_session.execute(select(Course))).scalar_one()
    assert stored.created_at.tzinfo is not None, "timestamps must be timezone-aware"


async def test_patch_advances_updated_at_but_not_created_at(api_client: AsyncClient) -> None:
    created = (
        await api_client.post("/api/v1/courses", json={"code": "CS229", "title": "ML"})
    ).json()

    patched = (
        await api_client.patch(f"/api/v1/courses/{created['id']}", json={"title": "ML II"})
    ).json()

    assert patched["created_at"] == created["created_at"]
    assert patched["updated_at"] > created["updated_at"]


async def test_normalised_code_is_what_gets_stored(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    await api_client.post("/api/v1/courses", json={"code": "  cs 229 ", "title": "ML"})

    assert (await db_session.execute(select(Course.code))).scalar_one() == "CS 229"


async def test_duplicate_code_is_rejected_by_the_unique_index(api_client: AsyncClient) -> None:
    """Same course code in any casing collides, because the DTO normalises."""
    assert (
        await api_client.post("/api/v1/courses", json={"code": "CS229", "title": "ML"})
    ).status_code == 201

    duplicate = await api_client.post("/api/v1/courses", json={"code": "cs229", "title": "Other"})

    assert duplicate.status_code == 409
    assert "already in use" in duplicate.json()["detail"]


async def test_concurrent_creates_of_the_same_code_produce_exactly_one_course(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """The reason there is no "does this code exist?" pre-check.

    Both requests are in flight before either has inserted, so a check-then-act
    service would let both through. The UNIQUE index is evaluated atomically,
    so one commit wins and the other is translated into a 409.
    """
    payload = {"code": "CS229", "title": "Machine Learning"}
    responses = await asyncio.gather(
        *(api_client.post("/api/v1/courses", json=payload) for _ in range(5))
    )

    statuses = sorted(response.status_code for response in responses)
    assert statuses == [201, 409, 409, 409, 409], statuses

    total = (await db_session.execute(select(func.count()).select_from(Course))).scalar_one()
    assert total == 1


async def test_patching_a_code_onto_an_existing_one_is_a_conflict(
    api_client: AsyncClient,
) -> None:
    await api_client.post("/api/v1/courses", json={"code": "CS229", "title": "ML"})
    other = (
        await api_client.post("/api/v1/courses", json={"code": "CS106B", "title": "Programming"})
    ).json()

    response = await api_client.patch(f"/api/v1/courses/{other['id']}", json={"code": "CS229"})

    assert response.status_code == 409


async def test_deleting_a_course_cascades_to_documents_and_chunks(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """The invariant behind the citation guarantee.

    A chunk whose document is gone is a page reference that can never be
    resolved. ON DELETE CASCADE in migration 0001 is what makes that
    unreachable, rather than a cleanup step someone has to remember.
    """
    course_id = (
        await api_client.post("/api/v1/courses", json={"code": "CS229", "title": "ML"})
    ).json()["id"]

    document = Document(
        course_id=course_id,
        filename="lecture3.pdf",
        content_hash="a" * 64,
        byte_size=12345,
        page_count=40,
        status=DocumentStatus.READY,
    )
    db_session.add(document)
    await db_session.flush()
    db_session.add(
        Chunk(
            document_id=document.id,
            ordinal=0,
            page_start=12,
            page_end=13,
            text="a passage about gradient descent",
            embedding=[0.1] * 384,
        )
    )
    await db_session.commit()

    assert (await api_client.delete(f"/api/v1/courses/{course_id}")).status_code == 204

    async def count(model: type[Course] | type[Document] | type[Chunk]) -> int:
        result = await db_session.execute(select(func.count()).select_from(model))
        return int(result.scalar_one())

    assert await count(Course) == 0
    assert await count(Document) == 0, "documents should have gone with the course"
    assert await count(Chunk) == 0, "chunks should have gone with the document"


async def test_list_is_ordered_by_code_across_pages(api_client: AsyncClient) -> None:
    for code in ("PHYS101", "CS229", "MATH214", "CS106B"):
        await api_client.post("/api/v1/courses", json={"code": code, "title": f"{code} course"})

    first = (await api_client.get("/api/v1/courses?limit=2&offset=0")).json()
    second = (await api_client.get("/api/v1/courses?limit=2&offset=2")).json()

    assert [item["code"] for item in first["items"]] == ["CS106B", "CS229"]
    assert [item["code"] for item in second["items"]] == ["MATH214", "PHYS101"]
    assert first["total"] == 4
