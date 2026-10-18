"""FastAPI application wiring.

The app is exposed only as a factory and served with
``uvicorn app.main:create_app --factory``. There is deliberately no
module-level ``app = create_app()``: that would make importing this module
read the environment, so tests (and Alembic) would need a configured
DATABASE_URL just to import it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import Settings, get_settings
from app.db import create_engine, create_sessionmaker
from app.embedding import SentenceTransformerEmbedder
from app.errors import (
    ConflictError,
    DomainError,
    EmbeddingError,
    GenerationError,
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaError,
    UnusableAnswerError,
)
from app.llm import ClaudeAnswerModel
from app.routers import answers, courses, documents, health, search
from app.storage import DocumentStorage

logger = logging.getLogger(__name__)

API_V1_PREFIX = "/api/v1"


def _domain_error_response(status_code: int, exc: Exception) -> JSONResponse:
    # Starlette types handlers as taking a bare Exception, hence the narrowing.
    detail = exc.detail if isinstance(exc, DomainError) else str(exc)
    # Same {"detail": ...} shape FastAPI uses for its own errors, so a client
    # needs one code path for failures regardless of where they came from.
    return JSONResponse(status_code=status_code, content={"detail": detail})


def _register_domain_errors(app: FastAPI) -> None:
    """Map domain errors onto status codes, once, for every route.

    This is why no route handler contains a try/except: a service raising
    NotFoundError from any depth becomes a 404. Subclasses dispatch to their
    nearest registered ancestor, so new error types get the right status for
    free.
    """

    async def not_found(request: Request, exc: Exception) -> JSONResponse:
        return _domain_error_response(status.HTTP_404_NOT_FOUND, exc)

    async def conflict(request: Request, exc: Exception) -> JSONResponse:
        return _domain_error_response(status.HTTP_409_CONFLICT, exc)

    async def too_large(request: Request, exc: Exception) -> JSONResponse:
        return _domain_error_response(status.HTTP_413_CONTENT_TOO_LARGE, exc)

    async def unsupported_media(request: Request, exc: Exception) -> JSONResponse:
        return _domain_error_response(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, exc)

    async def unavailable(request: Request, exc: Exception) -> JSONResponse:
        return _domain_error_response(status.HTTP_503_SERVICE_UNAVAILABLE, exc)

    async def bad_gateway(request: Request, exc: Exception) -> JSONResponse:
        return _domain_error_response(status.HTTP_502_BAD_GATEWAY, exc)

    app.add_exception_handler(NotFoundError, not_found)
    app.add_exception_handler(ConflictError, conflict)
    app.add_exception_handler(PayloadTooLargeError, too_large)
    app.add_exception_handler(UnsupportedMediaError, unsupported_media)
    # 503 rather than 500: the model loads in the background, so a search that
    # arrives too early has hit a state that fixes itself, and the client
    # should be told to retry rather than that something is broken.
    app.add_exception_handler(EmbeddingError, unavailable)
    # Same bargain for generation: an unset or rejected key, a rate limit and
    # an unreachable API all mean "this server cannot answer right now".
    app.add_exception_handler(GenerationError, unavailable)
    # A subclass of the above, registered separately because it is the one case
    # where the model *did* reply - with a refusal, or an answer cut off at the
    # token limit - so the upstream is not unavailable, its response is unusable.
    app.add_exception_handler(UnusableAnswerError, bad_gateway)
    # IngestionError is deliberately unmapped: it is only ever raised after the
    # response has been sent, and is recorded on the document row instead.


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        force=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the process-wide resources: the engine and the two models.

    Startup deliberately does not require any dependency to be *usable*. It
    creates the engine without connecting, and it starts the embedding model
    loading without waiting for it. A booting API that reports its own
    unreadiness is more useful than one that crash-loops while Postgres comes
    up, or one that answers nothing at all - not even `/health` - for the
    minutes a first-run model download takes. `/health/ready` is the honest
    signal for both.
    """
    settings: Settings = app.state.settings

    storage = DocumentStorage(settings.storage_dir)
    storage.prepare()
    app.state.storage = storage

    engine = create_engine(settings.database_url)
    app.state.engine = engine
    app.state.sessionmaker = create_sessionmaker(engine)

    # One model per process, shared by every request. It is hundreds of
    # megabytes of weights and thread-safe to call, so a copy per request or
    # per task would be pure waste.
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model, batch_size=settings.embedding_batch_size
    )
    app.state.embedder = embedder
    # Fire and forget: `load` records its own failure and never raises, and the
    # task is awaited on the way out so a shutdown mid-download is clean rather
    # than a warning about a task nobody retrieved.
    preload = asyncio.create_task(embedder.load()) if settings.embedding_preload else None

    # No preload equivalent and nothing to warm: the SDK client is a few
    # objects and an idle connection pool, built on the first question rather
    # than here so that a deployment with no key still starts cleanly.
    answer_model = ClaudeAnswerModel(
        api_key=settings.anthropic_api_key,
        model=settings.answer_model,
        max_tokens=settings.answer_max_tokens,
        effort=settings.answer_effort,
        timeout=settings.answer_timeout_seconds,
    )
    app.state.answer_model = answer_model

    logger.info(
        "startup complete: env=%s storage=%s embedder=%s answers=%s",
        settings.environment,
        settings.storage_dir,
        settings.embedding_model,
        settings.answer_model if settings.anthropic_api_key else "disabled (no api key)",
    )
    try:
        yield
    finally:
        if preload is not None:
            preload.cancel()
            await asyncio.gather(preload, return_exceptions=True)
        await answer_model.close()
        await engine.dispose()
        logger.info("shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings.log_level)

    app = FastAPI(
        title="Uni Knowledge Engine",
        summary="Ask questions about university course PDFs and get answers cited to page.",
        version=__version__,
        lifespan=lifespan,
    )
    # Set before the lifespan runs, which is where it gets read.
    app.state.settings = settings

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    _register_domain_errors(app)

    # Health is intentionally unversioned: probes should not have to track the
    # API version. Feature routers mount under API_V1_PREFIX.
    app.include_router(health.router)
    app.include_router(courses.router, prefix=API_V1_PREFIX)
    app.include_router(documents.router, prefix=API_V1_PREFIX)
    app.include_router(search.router, prefix=API_V1_PREFIX)
    app.include_router(answers.router, prefix=API_V1_PREFIX)

    return app
