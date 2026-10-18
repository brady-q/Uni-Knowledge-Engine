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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import Settings, get_settings
from app.db import create_engine, create_sessionmaker
from app.routers import health

logger = logging.getLogger(__name__)

API_V1_PREFIX = "/api/v1"


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

    # Health is intentionally unversioned: probes should not have to track the
    # API version. Feature routers mount under API_V1_PREFIX as they are added.
    app.include_router(health.router)

    return app
