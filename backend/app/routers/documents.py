"""Document upload and retrieval, nested under a course.

Documents only exist inside a course, and the URL says so. Scoping every route
this way means the ownership check is structural: there is no
``/documents/{id}`` that could return a document belonging to a course the
caller did not name.

As with courses, nothing here contains a try/except. Services raise domain
errors and the handlers registered in ``app.main`` turn them into statuses.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db import get_session
from app.embedding import Embedder, get_embedder
from app.errors import NotFoundError, PayloadTooLargeError
from app.schemas import (
    DEFAULT_CHUNK_PAGE_SIZE,
    DEFAULT_PAGE_SIZE,
    MAX_CHUNK_PAGE_SIZE,
    MAX_PAGE_SIZE,
    ChunkPage,
    ChunkRead,
    DocumentPage,
    DocumentRead,
)
from app.services.documents import DocumentService
from app.services.ingestion import ingest_document
from app.storage import DocumentStorage, get_storage

router = APIRouter(prefix="/courses/{course_id}/documents", tags=["documents"])

OpenApiResponses = dict[int | str, dict[str, Any]]

NOT_FOUND_RESPONSE: OpenApiResponses = {
    status.HTTP_404_NOT_FOUND: {"description": "No such course, or no such document in it"}
}
UPLOAD_FAILURE_RESPONSES: OpenApiResponses = {
    **NOT_FOUND_RESPONSE,
    status.HTTP_409_CONFLICT: {"description": "These bytes are already in this course"},
    status.HTTP_413_CONTENT_TOO_LARGE: {"description": "File exceeds the upload limit"},
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"description": "Not a PDF"},
}

# A multipart body is the file plus boundaries, headers and field names. This
# slack keeps the early Content-Length check from rejecting a file that is
# itself within the limit; the exact limit is still enforced on the bytes
# actually written to disk.
MULTIPART_OVERHEAD_ALLOWANCE = 8 * 1024


def get_document_service(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    storage: Annotated[DocumentStorage, Depends(get_storage)],
) -> DocumentService:
    settings: Settings = request.app.state.settings
    return DocumentService(session, storage, max_upload_bytes=settings.max_upload_bytes)


type IngestionScheduler = Callable[[uuid.UUID], None]


def get_ingestion_scheduler(
    request: Request,
    background: BackgroundTasks,
    storage: Annotated[DocumentStorage, Depends(get_storage)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> IngestionScheduler:
    """Hand back a way to queue ingestion for a document.

    A dependency rather than an inline ``background.add_task`` for the same
    reason ``get_readiness_report`` is one: it is the seam a test overrides.
    Without it, a unit test of the upload contract would run the real pipeline
    against the unreachable database the test app points at, and "was ingestion
    scheduled?" would not be observable at all.

    The embedder is resolved here, in request scope, and closed over. That is
    what makes overriding ``get_embedder`` enough to keep the whole default
    test suite away from the real model: ingestion reaches it through this
    dependency rather than reading ``app.state`` for itself.
    """

    def schedule(document_id: uuid.UUID) -> None:
        # The task runs once this response has been sent, on its own session:
        # the request's is closed by then.
        background.add_task(
            ingest_document, request.app.state.sessionmaker, storage, embedder, document_id
        )

    return schedule


ServiceDep = Annotated[DocumentService, Depends(get_document_service)]
StorageDep = Annotated[DocumentStorage, Depends(get_storage)]
SchedulerDep = Annotated[IngestionScheduler, Depends(get_ingestion_scheduler)]
CourseId = Annotated[uuid.UUID, Path(description="Course identifier")]
DocumentId = Annotated[uuid.UUID, Path(description="Document identifier")]


def _reject_oversized_body(request: Request) -> None:
    """Refuse an over-limit upload before reading it.

    Starlette spools a multipart body to a temporary file before this handler
    runs, so without this check a client could make the server write an
    arbitrarily large file just to be told 413. The header is a claim and can
    be absent or wrong, which is why ``DocumentStorage.stage`` enforces the
    real limit again as it streams.
    """
    settings: Settings = request.app.state.settings
    declared = request.headers.get("content-length")
    if declared is None or not declared.isdigit():
        return
    if int(declared) > settings.max_upload_bytes + MULTIPART_OVERHEAD_ALLOWANCE:
        raise PayloadTooLargeError(
            f"file exceeds the {settings.max_upload_bytes} byte upload limit"
        )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF",
    responses=UPLOAD_FAILURE_RESPONSES,
)
async def upload_document(
    course_id: CourseId,
    service: ServiceDep,
    schedule_ingestion: SchedulerDep,
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File(description="The PDF to ingest")],
) -> DocumentRead:
    """Store a PDF and begin extracting and chunking it.

    Returns as soon as the file is durable, with status ``pending``. Ingestion
    is scheduled rather than awaited - extraction takes seconds and a client
    should not hold a connection open for it - so poll the document to watch it
    progress to ``ready``, or to ``failed`` with a reason.
    """
    _reject_oversized_body(request)
    document = await service.upload(course_id, file)
    schedule_ingestion(document.id)

    response.headers["Location"] = str(
        request.url_for("get_document", course_id=course_id, document_id=document.id)
    )
    return DocumentRead.model_validate(document)


@router.get("", summary="List a course's documents", responses=NOT_FOUND_RESPONSE)
async def list_documents(
    course_id: CourseId,
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentPage:
    documents, total = await service.list(course_id, limit=limit, offset=offset)
    return DocumentPage(
        items=[DocumentRead.model_validate(document) for document in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", summary="Fetch a document", responses=NOT_FOUND_RESPONSE)
async def get_document(
    course_id: CourseId, document_id: DocumentId, service: ServiceDep
) -> DocumentRead:
    """Also the way to follow ingestion: ``status`` advances, ``error`` explains."""
    return DocumentRead.model_validate(await service.get(course_id, document_id))


@router.get(
    "/{document_id}/chunks",
    summary="List a document's chunks",
    responses=NOT_FOUND_RESPONSE,
)
async def list_document_chunks(
    course_id: CourseId,
    document_id: DocumentId,
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=MAX_CHUNK_PAGE_SIZE)] = DEFAULT_CHUNK_PAGE_SIZE,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChunkPage:
    """The chunker's output, in reading order, with the page range of each.

    Empty until ingestion reaches ``chunking``.
    """
    chunks, total = await service.list_chunks(course_id, document_id, limit=limit, offset=offset)
    return ChunkPage(
        items=[ChunkRead.model_validate(chunk) for chunk in chunks],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{document_id}/file",
    summary="Download the original PDF",
    responses={
        **NOT_FOUND_RESPONSE,
        status.HTTP_200_OK: {"content": {"application/pdf": {}}},
    },
    response_class=FileResponse,
)
async def download_document(
    course_id: CourseId,
    document_id: DocumentId,
    service: ServiceDep,
    storage: StorageDep,
) -> FileResponse:
    """Serve the stored bytes back.

    Fetched through the service so the course scoping is applied here too: the
    path on disk is derived from the document id alone, and handing that id
    straight to storage would serve any document to anyone who guessed one.
    """
    document = await service.get(course_id, document_id)
    path = storage.path_for(document.id)
    if not path.is_file():
        # The row survived but its bytes did not. Reported honestly rather than
        # as a 500, because from the caller's side the file is simply not there.
        raise NotFoundError(f"the stored file for document {document_id} is missing")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=document.filename,
        # Inline, so a browser or an <iframe> renders it instead of downloading
        # it - which is what the cited-page viewer will need.
        content_disposition_type="inline",
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    responses=NOT_FOUND_RESPONSE,
)
async def delete_document(
    course_id: CourseId, document_id: DocumentId, service: ServiceDep
) -> None:
    """Deleting a document also deletes its chunks and its stored file."""
    await service.delete(course_id, document_id)
