"""Alembic environment, async-aware.

The URL comes from application Settings rather than alembic.ini, so migrations
and the running app can never disagree about which database they mean.
``target_metadata`` is the ORM metadata, which is what autogenerate diffs
against the live database.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig
from typing import Any, Literal

from alembic import context
from alembic.autogenerate.api import AutogenContext
from pgvector.sqlalchemy import Vector
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config import get_settings

# Imported for the side effect of registering every table on Base.metadata.
# A model that is not reachable from this import is invisible to autogenerate,
# which silently produces an empty migration rather than an error.
from app.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# '%' is the ConfigParser interpolation character, so a URL-encoded password
# would otherwise blow up here.
config.set_main_option("sqlalchemy.url", get_settings().database_url.replace("%", "%%"))

target_metadata = Base.metadata


def render_item(type_: str, obj: Any, autogen_context: AutogenContext) -> str | Literal[False]:
    """Render pgvector column types with an import autogenerate can resolve.

    Alembic only knows how to emit ``sa.``-prefixed types, so without this a
    generated migration references an undefined ``sa.VECTOR`` and fails at
    import. Returning False falls back to the default rendering.
    """
    if type_ == "type" and isinstance(obj, Vector):
        autogen_context.imports.add("import pgvector.sqlalchemy")
        return f"pgvector.sqlalchemy.Vector({obj.dim})"
    return False


# Shared by the offline and online paths so a `--sql` dry run and a real
# migration can never be generated with different comparison rules.
COMPARE_OPTIONS: dict[str, Any] = {
    "target_metadata": target_metadata,
    # Catch a column whose type drifted from the models, not just added and
    # dropped columns.
    "compare_type": True,
    "compare_server_default": True,
    "render_item": render_item,
}


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a DB connection (`alembic upgrade head --sql`)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **COMPARE_OPTIONS,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, **COMPARE_OPTIONS)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
