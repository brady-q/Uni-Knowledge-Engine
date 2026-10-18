"""HTTP behaviour of the document routes, with no database and no disk.

The service is replaced with an in-memory stand-in and ingestion scheduling
with a recorder, the same way the course tests replace ``get_course_service``.
What is under test is strictly the router's job: status codes, the Location
header, the mapping from domain errors onto responses, and the fact that a
successful upload actually queues ingestion rather than quietly not doing so.

The app fixture still points at an unreachable database, so a route that
reached for a real session would fail loudly rather than pass by luck.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.errors import (
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaError,
)
from app.models import Chunk, Document, DocumentStatus
from app.routers.documents import (
    MULTIPART_OVERHEAD_ALLOWANCE,
    get_document_service,
    get_ingestion_scheduler,
)
from app.schemas import MAX_CHUNK_PAGE_SIZE, MAX_PAGE_SIZE
from app.storage import UploadStream

CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)
COURSE_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")

PDF_BYTES = b"%PDF-1.7\nfake but plausible"


def make_document(course_id: uuid.UUID, filename: str = "notes.pdf") -> Document:
    """A detached Document; column defaults only exist after an INSERT."""
    return Document(
        id=uuid.uuid4(),
        course_id=course_id,
        filename=filename,
        content_hash="0" * 64,
        byte_size=len(PDF_BYTES),
        page_count=None,
        status=DocumentStatus.PENDING,
        error=None,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )


class FakeDocumentService:
    """Same surface as DocumentService, backed by dicts."""

    def __init__(self) -> None:
        self.documents: dict[uuid.UUID, Document] = {}
        self.chunks: dict[uuid.UUID, list[Chunk]] = {}
        self.courses: set[uuid.UUID] = {COURSE_ID}
        self.raises: Exception | None = None

    def preload(self, count: int = 1) -> list[Document]:
        created = [make_document(COURSE_ID, f"doc{index}.pdf") for index in range(count)]
        for document in created:
            self.documents[document.id] = document
        return created

    async def upload(self, course_id: uuid.UUID, upload: UploadStream) -> Document:
        if self.raises is not None:
            raise self.raises
        self._require_course(course_id)
        document = make_document(course_id, upload.filename or "document.pdf")
        self.documents[document.id] = document
        return document

    async def get(self, course_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        document = self.documents.get(document_id)
        if document is None or document.course_id != course_id:
            raise NotFoundError(f"no document with id {document_id} in course {course_id}")
        return document

    async def list(
        self, course_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[Sequence[Document], int]:
        self._require_course(course_id)
        matching = [
            document for document in self.documents.values() if document.course_id == course_id
        ]
        return matching[offset : offset + limit], len(matching)

    async def list_chunks(
        self, course_id: uuid.UUID, document_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[Sequence[Chunk], int]:
        await self.get(course_id, document_id)
        matching = self.chunks.get(document_id, [])
        return matching[offset : offset + limit], len(matching)

    async def delete(self, course_id: uuid.UUID, document_id: uuid.UUID) -> None:
        document = await self.get(course_id, document_id)
        del self.documents[document.id]

    def _require_course(self, course_id: uuid.UUID) -> None:
        if course_id not in self.courses:
            raise NotFoundError(f"no course with id {course_id}")


class RecordingScheduler:
    def __init__(self) -> None:
        self.scheduled: list[uuid.UUID] = []

    def __call__(self, document_id: uuid.UUID) -> None:
        self.scheduled.append(document_id)


@pytest.fixture
def service() -> FakeDocumentService:
    return FakeDocumentService()


@pytest.fixture
def scheduler() -> RecordingScheduler:
    return RecordingScheduler()


@pytest.fixture
async def client(
    app: FastAPI, service: FakeDocumentService, scheduler: RecordingScheduler
) -> AsyncIterator[AsyncClient]:
    app.dependency_overrides[get_document_service] = lambda: service
    app.dependency_overrides[get_ingestion_scheduler] = lambda: scheduler
    try:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as http:
                yield http
    finally:
        app.dependency_overrides.clear()


def upload_body(
    data: bytes = PDF_BYTES, name: str = "notes.pdf"
) -> dict[str, tuple[str, bytes, str]]:
    return {"file": (name, data, "application/pdf")}


def documents_url(course_id: uuid.UUID = COURSE_ID) -> str:
    return f"/api/v1/courses/{course_id}/documents"


# --- upload ------------------------------------------------------------------


async def test_upload_returns_201_with_a_location_header(
    client: AsyncClient, service: FakeDocumentService
) -> None:
    response = await client.post(documents_url(), files=upload_body())

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["filename"] == "notes.pdf"
    assert body["page_count"] is None
    assert response.headers["Location"].endswith(f"{documents_url()}/{body['id']}")


async def test_upload_schedules_ingestion(
    client: AsyncClient, scheduler: RecordingScheduler
) -> None:
    """The upload is only half the job; without this the document never ingests."""
    response = await client.post(documents_url(), files=upload_body())

    assert scheduler.scheduled == [uuid.UUID(response.json()["id"])]


async def test_a_rejected_upload_does_not_schedule_ingestion(
    client: AsyncClient, service: FakeDocumentService, scheduler: RecordingScheduler
) -> None:
    service.raises = UnsupportedMediaError("only PDF files can be uploaded")

    await client.post(documents_url(), files=upload_body(b"not a pdf"))

    assert scheduler.scheduled == []


async def test_upload_without_a_file_is_422(client: AsyncClient) -> None:
    assert (await client.post(documents_url())).status_code == 422


async def test_the_embedding_is_never_exposed(
    client: AsyncClient, service: FakeDocumentService
) -> None:
    """384 floats per chunk is not part of the public contract."""
    document = service.preload()[0]
    service.chunks[document.id] = [
        Chunk(
            id=uuid.uuid4(),
            document_id=document.id,
            ordinal=0,
            page_start=1,
            page_end=1,
            text="A passage.",
            embedding=[0.0] * 8,
        )
    ]

    body = (await client.get(f"{documents_url()}/{document.id}/chunks")).json()

    assert body["items"][0]["text"] == "A passage."
    assert "embedding" not in body["items"][0]


# --- domain errors become statuses -------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (NotFoundError("no such course"), 404),
        (ConflictError("already uploaded"), 409),
        (PayloadTooLargeError("too big"), 413),
        (UnsupportedMediaError("not a pdf"), 415),
    ],
)
async def test_domain_errors_map_to_statuses(
    client: AsyncClient, service: FakeDocumentService, error: Exception, expected: int
) -> None:
    service.raises = error

    response = await client.post(documents_url(), files=upload_body())

    assert response.status_code == expected
    assert response.json()["detail"] == str(error)


async def test_an_oversized_body_is_refused_before_it_is_read(
    client: AsyncClient, app: FastAPI, service: FakeDocumentService
) -> None:
    """The Content-Length check fires without the service being consulted."""
    app.state.settings.max_upload_bytes = 1024
    # Comfortably past the limit plus the multipart allowance, which is what
    # the header is actually compared against.
    oversized = b"%PDF-1.7" + b"x" * (1024 + MULTIPART_OVERHEAD_ALLOWANCE)

    response = await client.post(documents_url(), files=upload_body(oversized))

    assert response.status_code == 413
    assert service.documents == {}


async def test_a_body_within_the_limit_is_not_refused_by_the_header_check(
    client: AsyncClient, app: FastAPI
) -> None:
    """The multipart allowance exists so boundaries do not fail a legal file."""
    app.state.settings.max_upload_bytes = 4096

    response = await client.post(documents_url(), files=upload_body(b"%PDF-1.7" + b"x" * 4000))

    assert response.status_code == 201


# --- reads --------------------------------------------------------------------


async def test_listing_a_missing_course_is_404(client: AsyncClient) -> None:
    assert (await client.get(documents_url(uuid.uuid4()))).status_code == 404


async def test_a_document_in_another_course_is_404(
    client: AsyncClient, service: FakeDocumentService
) -> None:
    """Scoped by URL: answering otherwise would confirm it exists elsewhere."""
    other = make_document(uuid.uuid4())
    service.documents[other.id] = other

    assert (await client.get(f"{documents_url()}/{other.id}")).status_code == 404


async def test_paging_is_reported_back(client: AsyncClient, service: FakeDocumentService) -> None:
    service.preload(5)

    body = (await client.get(documents_url(), params={"limit": 2, "offset": 1})).json()

    assert len(body["items"]) == 2
    assert (body["total"], body["limit"], body["offset"]) == (5, 2, 1)


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("", {"limit": MAX_PAGE_SIZE + 1}),
        ("", {"limit": 0}),
        ("", {"offset": -1}),
    ],
)
async def test_out_of_range_paging_is_422(
    client: AsyncClient, path: str, params: dict[str, int]
) -> None:
    assert (await client.get(documents_url() + path, params=params)).status_code == 422


async def test_chunk_paging_has_its_own_lower_ceiling(
    client: AsyncClient, service: FakeDocumentService
) -> None:
    """Chunk text is long; a course-sized page of it would be megabytes."""
    document = service.preload()[0]

    response = await client.get(
        f"{documents_url()}/{document.id}/chunks", params={"limit": MAX_CHUNK_PAGE_SIZE + 1}
    )

    assert response.status_code == 422
    assert MAX_CHUNK_PAGE_SIZE < MAX_PAGE_SIZE


async def test_a_malformed_id_is_422_not_404(client: AsyncClient) -> None:
    assert (await client.get(f"{documents_url()}/not-a-uuid")).status_code == 422


async def test_delete_returns_204(client: AsyncClient, service: FakeDocumentService) -> None:
    document = service.preload()[0]

    assert (await client.delete(f"{documents_url()}/{document.id}")).status_code == 204
    assert service.documents == {}


async def test_deleting_a_missing_document_is_404(client: AsyncClient) -> None:
    assert (await client.delete(f"{documents_url()}/{uuid.uuid4()}")).status_code == 404
