"""Async engine, session factory, and the request-scoped session dependency.

The engine is created in the application lifespan and stored on ``app.state``,
not at import time: importing this module must never open a socket, or tests
and Alembic would need a live database just to load the app.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(
        database_url,
        # Recycles connections a proxy or restarted Postgres has dropped,
        # which turns a confusing 500 into a transparent reconnect.
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
    )


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a session bound to the app's engine, closing it after the request."""
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with sessionmaker() as session:
        yield session


def violated_constraint(error: IntegrityError) -> str | None:
    """Name the constraint Postgres rejected a statement on, if it said.

    asyncpg records the constraint on the driver exception, which SQLAlchemy
    then wraps twice: ``error.orig`` is the DBAPI shim and its ``__cause__`` is
    the asyncpg error carrying ``constraint_name``. Matching on that name
    rather than on the message text keeps the mapping exact - a table with two
    unique constraints must not report either violation as the other.
    """
    cause = getattr(error.orig, "__cause__", None)
    name = getattr(cause, "constraint_name", None)
    return name if isinstance(name, str) else None
