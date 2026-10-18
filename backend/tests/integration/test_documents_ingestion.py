"""Upload, extraction, chunking and embedding against a real Postgres.

The unit tests prove each stage in isolation with a fake either side. These
prove the parts a fake cannot: that the chunks land in the database, that the
constraints in migration 0001 actually fire on them, that the bytes and the row
stay in step, and that a failing PDF ends up with a readable reason on the row
instead of an exception nobody sees.

Ingestion is scheduled as a FastAPI background task, and Starlette runs those
before the ASGI call completes - so by the time ``await client.post(...)``
returns, the pipeline has already finished. That is a property of the in-process
transport, not of production, where the client would poll.

Vectors here come from the fake embedder the ``api_client`` fixture injects, so
these tests stay about upload, extraction, chunking and the two-store dance.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import Chunk, Document, DocumentStatus
from app.services.chunking import CHUNK_SIZE
from app.storage import DocumentStorage
from tests.pdfs import make_empty_pdf, make_encrypted_pdf, make_pdf

pytestmark = pytest.mark.integration

PARAGRAPH = (
    "Gradient descent iteratively adjusts parameters in the direction that "
    "most reduces the loss. The learning rate controls the size of each step. "
)


def paragraphs(count: int, marker: str) -> list[str]:
    return [f"{marker} paragraph {index}. {PARAGRAPH}" for index in range(count)]


async def create_course(client: AsyncClient, code: str = "CS229") -> str:
    response = await client.post(
        "/api/v1/courses", json={"code": code, "title": "Machine Learning"}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def upload(
    client: AsyncClient, course_id: str, data: bytes, name: str = "lecture.pdf"
) -> dict[str, Any]:
    response = await client.post(
        f"/api/v1/courses/{course_id}/documents",
        files={"file": (name, data, "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


def storage_for(settings: Settings) -> DocumentStorage:
    return DocumentStorage(settings.storage_dir)


# --- the happy path -----------------------------------------------------------


async def test_upload_extracts_and_chunks_the_document(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    course_id = await create_course(api_client)
    pdf = make_pdf([paragraphs(4, "Alpha"), paragraphs(4, "Beta"), paragraphs(4, "Gamma")])

    created = await upload(api_client, course_id, pdf)

    # Returned before ingestion; the response describes the upload, not the result.
    assert created["status"] == "pending"
    assert created["page_count"] is None

    document = (
        await db_session.execute(select(Document).where(Document.id == uuid.UUID(created["id"])))
    ).scalar_one()
    assert document.status is DocumentStatus.READY
    assert document.page_count == 3
    assert document.error is None

    chunks = (
        (
            await db_session.execute(
                select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.ordinal)
            )
        )
        .scalars()
        .all()
    )
    assert chunks
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(1 <= chunk.page_start <= chunk.page_end <= 3 for chunk in chunks)
    assert all(len(chunk.text) <= CHUNK_SIZE for chunk in chunks)
    # Embedded in the same pass, which is what makes "ready" mean searchable.
    # The vectors themselves are the subject of test_search_retrieval.py.
    assert all(chunk.embedding is not None for chunk in chunks)


async def test_every_page_is_represented_by_a_chunk_that_cites_it(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """The property the citation guarantee rests on, end to end.

    Text from page N has to appear in a chunk whose recorded range covers N.
    If chunking and page attribution ever drift apart, a citation resolves to a
    page the quoted passage is not on - which is exactly the failure this
    project exists to make impossible.
    """
    course_id = await create_course(api_client)
    pdf = make_pdf([[f"Distinctive marker {number}.", *paragraphs(2, "F")] for number in (1, 2, 3)])

    created = await upload(api_client, course_id, pdf)

    chunks = (
        (
            await db_session.execute(
                select(Chunk).where(Chunk.document_id == uuid.UUID(created["id"]))
            )
        )
        .scalars()
        .all()
    )
    for number in (1, 2, 3):
        citing = [
            chunk
            for chunk in chunks
            if f"Distinctive marker {number}." in chunk.text
            and chunk.page_start <= number <= chunk.page_end
        ]
        assert citing, f"page {number} is cited by no chunk containing its text"


async def test_the_status_endpoint_reports_the_finished_document(
    api_client: AsyncClient,
) -> None:
    course_id = await create_course(api_client)
    created = await upload(api_client, course_id, make_pdf([paragraphs(3, "Alpha")]))

    fetched = await api_client.get(f"/api/v1/courses/{course_id}/documents/{created['id']}")

    assert fetched.status_code == 200
    assert fetched.json()["status"] == "ready"
    assert fetched.json()["page_count"] == 1


async def test_chunks_are_listed_in_reading_order(api_client: AsyncClient) -> None:
    course_id = await create_course(api_client)
    created = await upload(api_client, course_id, make_pdf([paragraphs(12, "Alpha")]))

    body = (
        await api_client.get(
            f"/api/v1/courses/{course_id}/documents/{created['id']}/chunks",
            params={"limit": 100},
        )
    ).json()

    assert body["total"] == len(body["items"]) > 1
    assert [item["ordinal"] for item in body["items"]] == list(range(body["total"]))


async def test_the_stored_pdf_can_be_served_back(
    api_client: AsyncClient, isolated_settings: Settings
) -> None:
    course_id = await create_course(api_client)
    pdf = make_pdf([paragraphs(2, "Alpha")])
    created = await upload(api_client, course_id, pdf, name="week 1 notes.pdf")

    response = await api_client.get(f"/api/v1/courses/{course_id}/documents/{created['id']}/file")

    assert response.status_code == 200
    assert response.content == pdf
    assert response.headers["content-type"] == "application/pdf"
    # Inline, so the cited-page viewer can render it rather than download it.
    assert response.headers["content-disposition"].startswith("inline")
    assert storage_for(isolated_settings).path_for(uuid.UUID(created["id"])).is_file()


async def test_the_recorded_hash_and_size_describe_the_uploaded_bytes(
    api_client: AsyncClient,
) -> None:
    course_id = await create_course(api_client)
    pdf = make_pdf([paragraphs(2, "Alpha")])

    created = await upload(api_client, course_id, pdf)

    assert created["byte_size"] == len(pdf)
    assert created["content_hash"] == hashlib.sha256(pdf).hexdigest()


# --- rejected uploads ---------------------------------------------------------


async def test_the_same_bytes_cannot_be_uploaded_twice_to_one_course(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """Enforced by uq_documents(course_id, content_hash), not by a pre-check."""
    course_id = await create_course(api_client)
    pdf = make_pdf([paragraphs(2, "Alpha")])
    await upload(api_client, course_id, pdf)

    repeat = await api_client.post(
        f"/api/v1/courses/{course_id}/documents",
        files={"file": ("renamed.pdf", pdf, "application/pdf")},
    )

    assert repeat.status_code == 409
    assert (await db_session.execute(select(func.count()).select_from(Document))).scalar_one() == 1


async def test_the_same_bytes_may_be_uploaded_to_a_different_course(
    api_client: AsyncClient,
) -> None:
    """The constraint is per course: two courses can legitimately share a text."""
    pdf = make_pdf([paragraphs(2, "Alpha")])
    first = await create_course(api_client, "CS229")
    second = await create_course(api_client, "CS230")

    await upload(api_client, first, pdf)
    await upload(api_client, second, pdf)


async def test_a_rejected_duplicate_leaves_no_file_behind(
    api_client: AsyncClient, isolated_settings: Settings
) -> None:
    """Staged bytes with no row would accumulate forever."""
    course_id = await create_course(api_client)
    pdf = make_pdf([paragraphs(2, "Alpha")])
    await upload(api_client, course_id, pdf)

    await api_client.post(
        f"/api/v1/courses/{course_id}/documents",
        files={"file": ("again.pdf", pdf, "application/pdf")},
    )

    staging = isolated_settings.storage_dir / "staging"
    assert list(staging.iterdir()) == []


async def test_a_non_pdf_is_415_and_stores_nothing(
    api_client: AsyncClient, db_session: AsyncSession, isolated_settings: Settings
) -> None:
    course_id = await create_course(api_client)

    response = await api_client.post(
        f"/api/v1/courses/{course_id}/documents",
        # Correct extension and Content-Type, wrong bytes: the magic number is
        # the only part of the claim a client cannot get wrong for free.
        files={"file": ("notes.pdf", b"PK\x03\x04 a zip file", "application/pdf")},
    )

    assert response.status_code == 415
    assert (await db_session.execute(select(func.count()).select_from(Document))).scalar_one() == 0
    assert list((isolated_settings.storage_dir / "staging").iterdir()) == []


async def test_an_empty_file_is_415(api_client: AsyncClient) -> None:
    """byte_size > 0 is a CHECK constraint; this fails before reaching it."""
    course_id = await create_course(api_client)

    response = await api_client.post(
        f"/api/v1/courses/{course_id}/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 415


async def test_an_oversized_upload_is_413(
    api_client: AsyncClient, isolated_settings: Settings
) -> None:
    course_id = await create_course(api_client)
    isolated_settings.max_upload_bytes = 2048

    response = await api_client.post(
        f"/api/v1/courses/{course_id}/documents",
        files={"file": ("big.pdf", b"%PDF-1.7" + b"x" * 20_000, "application/pdf")},
    )

    assert response.status_code == 413


async def test_uploading_to_a_missing_course_is_404(api_client: AsyncClient) -> None:
    response = await api_client.post(
        f"/api/v1/courses/{uuid.uuid4()}/documents",
        files={"file": ("notes.pdf", make_pdf([["Text."]]), "application/pdf")},
    )

    assert response.status_code == 404


# --- failed ingestion ---------------------------------------------------------


async def test_a_scanned_pdf_fails_with_a_reason_on_the_row(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """Ingestion has no caller to raise to, so the reason has to be persisted."""
    course_id = await create_course(api_client)

    created = await upload(api_client, course_id, make_empty_pdf(2), name="scan.pdf")

    document = (
        await db_session.execute(select(Document).where(Document.id == uuid.UUID(created["id"])))
    ).scalar_one()
    assert document.status is DocumentStatus.FAILED
    assert document.error is not None
    assert "scanned" in document.error
    assert (await db_session.execute(select(func.count()).select_from(Chunk))).scalar_one() == 0


async def test_an_encrypted_pdf_fails_with_its_own_reason(api_client: AsyncClient) -> None:
    course_id = await create_course(api_client)

    created = await upload(api_client, course_id, make_encrypted_pdf(), name="locked.pdf")

    body = (await api_client.get(f"/api/v1/courses/{course_id}/documents/{created['id']}")).json()
    assert body["status"] == "failed"
    assert "password" in body["error"]


async def test_a_failed_document_is_still_a_document(api_client: AsyncClient) -> None:
    """It is listed and deletable, not a half-existing row."""
    course_id = await create_course(api_client)
    created = await upload(api_client, course_id, make_empty_pdf(1), name="scan.pdf")

    listed = (await api_client.get(f"/api/v1/courses/{course_id}/documents")).json()
    assert [item["id"] for item in listed["items"]] == [created["id"]]

    deleted = await api_client.delete(f"/api/v1/courses/{course_id}/documents/{created['id']}")
    assert deleted.status_code == 204


# --- deletion -----------------------------------------------------------------


async def test_deleting_a_document_removes_its_chunks_and_its_bytes(
    api_client: AsyncClient, db_session: AsyncSession, isolated_settings: Settings
) -> None:
    course_id = await create_course(api_client)
    created = await upload(api_client, course_id, make_pdf([paragraphs(6, "Alpha")]))
    path = storage_for(isolated_settings).path_for(uuid.UUID(created["id"]))
    assert path.is_file()

    response = await api_client.delete(f"/api/v1/courses/{course_id}/documents/{created['id']}")

    assert response.status_code == 204
    assert not path.exists()
    assert (await db_session.execute(select(func.count()).select_from(Chunk))).scalar_one() == 0


async def test_deleting_the_course_cascades_to_documents_chunks_and_files(
    api_client: AsyncClient, db_session: AsyncSession, isolated_settings: Settings
) -> None:
    """Rows go by ON DELETE CASCADE - but nothing cascades to the filesystem.

    The stored PDFs have to be removed by the service, from ids it read before
    the delete. Without that, every file uploaded to the course leaks: the rows
    naming them are gone, so nothing can ever find them again.
    """
    course_id = await create_course(api_client)
    first = await upload(api_client, course_id, make_pdf([paragraphs(6, "Alpha")]), "a.pdf")
    second = await upload(api_client, course_id, make_pdf([paragraphs(6, "Beta")]), "b.pdf")
    storage = storage_for(isolated_settings)
    paths = [storage.path_for(uuid.UUID(document["id"])) for document in (first, second)]
    assert all(path.is_file() for path in paths)

    assert (await api_client.delete(f"/api/v1/courses/{course_id}")).status_code == 204

    for model in (Document, Chunk):
        assert (await db_session.execute(select(func.count()).select_from(model))).scalar_one() == 0
    assert not any(path.exists() for path in paths)


async def test_a_document_cannot_be_deleted_through_another_course(
    api_client: AsyncClient,
) -> None:
    first = await create_course(api_client, "CS229")
    second = await create_course(api_client, "CS230")
    created = await upload(api_client, first, make_pdf([paragraphs(2, "Alpha")]))

    response = await api_client.delete(f"/api/v1/courses/{second}/documents/{created['id']}")

    assert response.status_code == 404


async def test_a_deleted_file_is_reported_as_missing_rather_than_a_500(
    api_client: AsyncClient, isolated_settings: Settings
) -> None:
    """The row can outlive its bytes; from the caller's side it is simply gone."""
    course_id = await create_course(api_client)
    created = await upload(api_client, course_id, make_pdf([paragraphs(2, "Alpha")]))
    storage_for(isolated_settings).path_for(uuid.UUID(created["id"])).unlink()

    response = await api_client.get(f"/api/v1/courses/{course_id}/documents/{created['id']}/file")

    assert response.status_code == 404


# --- listing ------------------------------------------------------------------


async def test_documents_are_listed_newest_first_and_scoped_to_their_course(
    api_client: AsyncClient,
) -> None:
    first = await create_course(api_client, "CS229")
    second = await create_course(api_client, "CS230")
    older = await upload(api_client, first, make_pdf([["Older.", *paragraphs(2, "A")]]))
    newer = await upload(api_client, first, make_pdf([["Newer.", *paragraphs(2, "B")]]))
    await upload(api_client, second, make_pdf([["Elsewhere.", *paragraphs(2, "C")]]))

    body = (await api_client.get(f"/api/v1/courses/{first}/documents")).json()

    assert body["total"] == 2
    assert [item["id"] for item in body["items"]] == [newer["id"], older["id"]]
