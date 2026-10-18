"""FastAPI application wiring.

The app is exposed only as a factory and served with
``uvicorn app.main:create_app --factory``. There is deliberately no
module-level ``app = create_app()``: that would make importing this module
read the environment, so tests (and Alembic) would need a configured
DATABASE_URL just to import it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import Settings, get_settings
from app.db import create_engine, create_sessionmaker
from app.errors import ConflictError, DomainError, NotFoundError
from app.routers import courses, health

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

    app.add_exception_handler(NotFoundError, not_found)
    app.add_exception_handler(ConflictError, conflict)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        force=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own process-wide resources: the DB engine now, the embedding model later.

    Startup deliberately does not require the database to be reachable. A
    booting API that reports its own unreadiness is more useful than one that
    crash-loops while Postgres comes up; `/health/ready` is the honest signal.
    """
    settings: Settings = app.state.settings
    settings.storage_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(settings.database_url)
    app.state.engine = engine
    app.state.sessionmaker = create_sessionmaker(engine)
    logger.info(
        "startup complete: env=%s storage=%s",
        settings.environment,
        settings.storage_dir,
    )
    try:
        yield
    finally:
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

    return app
