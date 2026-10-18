"""What migration 0001 actually built.

The models describe a schema; these tests check that the migration produced it
and that the constraints are real. Everything here queries the catalogue or
tries an illegal write rather than trusting the ORM's own picture of the world.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EMBEDDING_DIMENSIONS
from tests.integration.conftest import AlembicRunner

pytestmark = pytest.mark.integration


def test_models_and_migrations_do_not_drift(alembic: AlembicRunner) -> None:
    """The highest-value test in this file.

    `alembic check` autogenerates against the live database and fails if it
    finds anything. A green run means the models, the migration and the real
    schema all agree - which is what makes `make revision` trustworthy for the
    next change instead of a source of surprise diffs.
    """
    result = alembic("check")

    assert result.returncode == 0, f"schema drift detected:\n{result.stdout}\n{result.stderr}"


async def test_pgvector_is_installed_by_the_migration(db_session: AsyncSession) -> None:
    """The readiness probe's pgvector check should now report "installed"."""
    version = await db_session.scalar(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    )

    assert version is not None


async def test_embedding_column_has_the_expected_dimensions(db_session: AsyncSession) -> None:
    """A mismatch here would only surface when the first vector is inserted."""
    declared = await db_session.scalar(
        text("""
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            WHERE a.attrelid = 'chunks'::regclass AND a.attname = 'embedding'
        """)
    )

    assert declared == f"vector({EMBEDDING_DIMENSIONS})"


async def test_chunks_have_an_hnsw_index_using_cosine_distance(db_session: AsyncSession) -> None:
    """The opclass has to match the distance operator retrieval will use.

    An index built with vector_l2_ops is simply not used by a `<=>` query, and
    the only symptom is a sequential scan once the table is large.
    """
    definition = await db_session.scalar(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunks_embedding_hnsw'")
    )

    assert definition is not None
    assert "USING hnsw" in definition
    assert "vector_cosine_ops" in definition


@pytest.mark.parametrize(
    ("constraint", "referenced"),
    [
        ("fk_documents_course_id_courses", "courses"),
        ("fk_chunks_document_id_documents", "documents"),
    ],
)
async def test_foreign_keys_cascade_on_delete(
    db_session: AsyncSession, constraint: str, referenced: str
) -> None:
    """'c' is CASCADE. 'a' (no action) would leave orphaned rows behind.

    Cast to text because confdeltype is Postgres's internal "char" type, which
    asyncpg hands back as bytes.
    """
    action = await db_session.scalar(
        text("SELECT confdeltype::text FROM pg_constraint WHERE conname = :name"),
        {"name": constraint},
    )

    assert action == "c", f"{constraint} -> {referenced} is not ON DELETE CASCADE"


async def test_constraints_use_the_naming_convention(db_session: AsyncSession) -> None:
    """Deterministic names are what a later ALTER or downgrade has to reference.

    Left to itself Postgres would invent names like "courses_code_key", which
    are invisible in the models and easy to get wrong in a migration.
    """
    names = (
        await db_session.scalars(
            text("""
                SELECT conname FROM pg_constraint
                WHERE conrelid IN ('courses'::regclass, 'documents'::regclass,
                                   'chunks'::regclass)
            """)
        )
    ).all()

    assert names
    assert all(name.startswith(("pk_", "uq_", "fk_", "ck_")) for name in names), names


async def test_a_backwards_page_range_is_rejected(db_session: AsyncSession) -> None:
    """Checked in the database, because citations depend on these numbers."""
    with pytest.raises(IntegrityError, match="ck_chunks_page_range_ordered"):
        await db_session.execute(
            text("""
                INSERT INTO chunks (document_id, ordinal, page_start, page_end, text)
                VALUES (gen_random_uuid(), 0, 9, 3, 'backwards')
            """)
        )


async def test_an_unknown_document_status_is_rejected(db_session: AsyncSession) -> None:
    """What native_enum=False plus create_constraint=True buys.

    Without the CHECK constraint the column would be a plain VARCHAR and would
    accept any string at all.
    """
    with pytest.raises(IntegrityError, match="ck_documents_document_status"):
        await db_session.execute(
            text("""
                INSERT INTO documents (course_id, filename, content_hash, byte_size, status)
                VALUES (gen_random_uuid(), 'f.pdf', repeat('a', 64), 1, 'not_a_status')
            """)
        )


async def test_a_document_cannot_reference_a_missing_course(db_session: AsyncSession) -> None:
    with pytest.raises(IntegrityError, match="fk_documents_course_id_courses"):
        await db_session.execute(
            text("""
                INSERT INTO documents (course_id, filename, content_hash, byte_size)
                VALUES (gen_random_uuid(), 'f.pdf', repeat('a', 64), 1)
            """)
        )
