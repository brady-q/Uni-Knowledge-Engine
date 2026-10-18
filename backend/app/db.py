"""Async engine, session factory, and the request-scoped session dependency.

The engine is created in the application lifespan and stored on ``app.state``,
not at import time: importing this module must never open a socket, or tests
and Alembic would need a live database just to load the app.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
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
