"""SQLAlchemy ORM models: the one in-Python description of the schema.

``Base.metadata`` is what Alembic diffs against the live database, so this
module is the source of truth for all DDL and every table must be reachable
from here (see ``migrations/env.py``).

Invariants are declared as database constraints rather than checked in service
code, because a UNIQUE index is atomic while a "SELECT then INSERT" check in
Python is a race two concurrent requests can both win. Services translate the
*violation* into an HTTP status; Postgres enforces the rule.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# all-MiniLM-L6-v2 and bge-small-en-v1.5 both emit 384 dimensions. The width is
# baked into the DDL, so switching embedding models is a migration and a
# re-embed, never a config change - hence one named constant to grep for.
EMBEDDING_DIMENSIONS = 384

# Postgres will happily invent names like "courses_code_key" for constraints it
# creates implicitly, and those names are what a later ALTER or a downgrade has
# to reference. Naming them deterministically here means migrations can always
# address a constraint by a name that is visible in this file.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Referenced by the service that turns a duplicate insert into a 409, so the
# error mapping breaks loudly here if the constraint is ever renamed.
COURSE_CODE_UNIQUE = "uq_courses_code"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    # Fetch server-generated values (the timestamps below) back with RETURNING
    # instead of leaving them expired. Together with expire_on_commit=False in
    # app/db.py this is what lets a router serialise a just-committed row
    # without a second SELECT.
    # RUF012 wants ClassVar here, but SQLAlchemy requires this exact attribute
    # name as a plain dict, and annotating it as ClassVar collides with
    # DeclarativeBase's own declaration under mypy --strict.
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012


class UuidPrimaryKey:
    """Opaque UUID identity.

    Chosen over a bigserial because these ids appear in frontend routes and in
    citation payloads, where a sequential integer would leak how many rows
    exist and invite enumeration. The cost is 16 bytes and unordered index
    inserts, which is irrelevant at this scale.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        # Mixin columns are appended after the model's own by default; these
        # keep the generated DDL readable: id first, timestamps last.
        sort_order=-100,
        # Declared server-side as well, so data migrations and hand-written SQL
        # do not have to invent ids. The Python default wins for ORM inserts,
        # which lets a service know the id before the row is even flushed.
        server_default=func.gen_random_uuid(),
    )


class Timestamped:
    """Row lifecycle, kept in the database's clock rather than the app's."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), sort_order=100
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        sort_order=100,
    )


class DocumentStatus(StrEnum):
    """Where a document is in the ingestion pipeline.

    Stored as a checked VARCHAR rather than a native Postgres ENUM: adding a
    stage later is then a CHECK constraint swap inside a normal transaction,
    instead of an ALTER TYPE that cannot always run in one.
    """

    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class Course(UuidPrimaryKey, Timestamped, Base):
    """A unit of study. Documents, and later questions, are scoped to one."""

    __tablename__ = "courses"

    # Normalised to upper case at the API boundary (schemas.CourseCreate), so a
    # plain UNIQUE index is enough and the citext extension is not needed.
    # UNIQUE here rather than a primary key because course codes get renamed
    # and reused, and identity must never change.
    code: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    documents: Mapped[list[Document]] = relationship(
        back_populates="course",
        cascade="all, delete-orphan",
        # Trust the database's ON DELETE CASCADE instead of loading every child
        # to delete it individually: one DELETE rather than N+1.
        passive_deletes=True,
    )


class Document(UuidPrimaryKey, Timestamped, Base):
    """An uploaded PDF and its ingestion state."""

    __tablename__ = "documents"
    __table_args__ = (
        # The same bytes may only be uploaded once per course. This also
        # supplies the index Postgres does *not* create automatically for a
        # foreign key: course_id leads, so "documents in this course" is
        # covered by the same index.
        UniqueConstraint("course_id", "content_hash"),
        CheckConstraint("byte_size > 0", name="byte_size_positive"),
        CheckConstraint("page_count IS NULL OR page_count > 0", name="page_count_positive"),
    )

    course_id: Mapped[uuid.UUID] = mapped_column(
        # ON DELETE CASCADE is declared in the database, not just in the ORM
        # relationship: a chunk whose document is gone is a citation that can
        # never be resolved, and no amount of application code can be trusted
        # to prevent that as reliably as the constraint does.
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False,
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # SHA-256 hex, so re-uploading an identical file is detectable without
    # reading the stored bytes back.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    # Unknown until extraction has run, hence nullable.
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(
            DocumentStatus,
            name="document_status",
            native_enum=False,
            # Off by default since SQLAlchemy 1.4: without it the column is an
            # unchecked VARCHAR and any string at all would be accepted.
            create_constraint=True,
            length=16,
            # Without this SQLAlchemy persists the member *name* ("PENDING");
            # the lower-case value is what the API exposes.
            values_callable=lambda enum: [member.value for member in enum],
        ),
        nullable=False,
        default=DocumentStatus.PENDING,
        server_default=DocumentStatus.PENDING.value,
        index=True,
    )
    # Why ingestion failed, surfaced to the user rather than buried in logs.
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    course: Mapped[Course] = relationship(back_populates="documents")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Chunk(UuidPrimaryKey, Base):
    """A retrievable passage, carrying the page range every citation resolves to.

    No timestamps: chunks are derived data, fully regenerable from their
    document, and there will be orders of magnitude more of them than of
    anything else.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        # Position within the document. Doubles as the index covering "every
        # chunk of this document, in reading order".
        UniqueConstraint("document_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_non_negative"),
        # Page numbers are 1-based and a range cannot run backwards. These are
        # the values the citation guarantee rests on, so they are checked by
        # the database rather than assumed from the chunker.
        CheckConstraint("page_start >= 1", name="page_start_positive"),
        CheckConstraint("page_end >= page_start", name="page_range_ordered"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            # Cosine distance, to match the normalised vectors the
            # sentence-transformers models produce.
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Nullable because chunks are written during extraction and embedded in a
    # later pass: a chunk without a vector is a valid intermediate state, not
    # corruption. Retrieval filters these out.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
