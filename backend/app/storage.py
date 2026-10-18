"""On-disk store for uploaded PDFs.

The database owns metadata; this owns bytes. Keeping them apart means a row is
never large, and it means the blob for a document can be found without a query:
the path is derived from the document id alone.

Deliberately *not* content-addressed. Hashing the bytes into the path would let
the same PDF uploaded to two courses share one file, but then deleting a
document has to prove no other row still points at it, and getting that
reference count wrong either leaks files forever or deletes one that is still
cited. One file per document is a little more disk and no bookkeeping.

Writes are staged into a sibling directory and moved into place with
``os.replace``, which is atomic within a filesystem. A crash mid-upload
therefore leaves a ``.part`` file behind, never a truncated PDF that extraction
would read as real.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import anyio
from fastapi import Request

from app.errors import PayloadTooLargeError

logger = logging.getLogger(__name__)

# 1 MiB per read: large enough that a 50 MiB upload is 50 iterations rather
# than thousands, small enough that memory use does not scale with file size.
READ_CHUNK_BYTES = 1024 * 1024

# Enough to identify the format. PDFs start with "%PDF-" and a version.
HEADER_BYTES = 8

_DOCUMENTS_DIR = "documents"
_STAGING_DIR = "staging"


class UploadStream(Protocol):
    """The part of Starlette's ``UploadFile`` this module actually needs.

    Narrow on purpose: it is what keeps the storage and service layers free of
    a FastAPI import, and it is trivial for a test to satisfy with a wrapper
    around ``BytesIO``.
    """

    filename: str | None

    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True, slots=True)
class StagedUpload:
    """A fully received upload that is not yet part of any document."""

    path: Path
    byte_size: int
    # SHA-256 hex, which is what documents.content_hash stores, so a re-upload
    # of identical bytes is detectable without reading the stored file back.
    content_hash: str
    # The first few bytes, kept from the streaming pass so format validation
    # does not need to reopen the file.
    header: bytes


class DocumentStorage:
    """Blob storage rooted at one directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, document_id: uuid.UUID) -> Path:
        """Where a document's bytes live.

        Sharded on the first two hex characters of the id so no single
        directory accumulates every PDF ever uploaded; ext4 copes, but ``ls``
        and backup tools stop being usable somewhere around six figures.
        """
        name = document_id.hex
        return self._root / _DOCUMENTS_DIR / name[:2] / f"{name}.pdf"

    def prepare(self) -> None:
        """Create the directory tree. Called once, from the app lifespan."""
        (self._root / _DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)
        (self._root / _STAGING_DIR).mkdir(parents=True, exist_ok=True)

    async def stage(self, upload: UploadStream, *, max_bytes: int) -> StagedUpload:
        """Stream an upload to a temporary file, hashing and measuring as it goes.

        The size limit is enforced here rather than after the write, so a
        client that lies about Content-Length fills at most ``max_bytes`` plus
        one read chunk of disk before being cut off.
        """
        staging = self._root / _STAGING_DIR
        staging.mkdir(parents=True, exist_ok=True)
        # uuid4 rather than the document id: no row exists yet, and an upload
        # that is rejected must never collide with a concurrent one.
        path = staging / f"{uuid.uuid4().hex}.part"

        digest = hashlib.sha256()
        byte_size = 0
        header = b""
        try:
            async with await anyio.open_file(path, "wb") as handle:
                while data := await upload.read(READ_CHUNK_BYTES):
                    byte_size += len(data)
                    if byte_size > max_bytes:
                        raise PayloadTooLargeError(
                            f"file exceeds the {max_bytes} byte upload limit"
                        )
                    if len(header) < HEADER_BYTES:
                        header = (header + data)[:HEADER_BYTES]
                    digest.update(data)
                    await handle.write(data)
        except BaseException:
            # Includes PayloadTooLargeError and cancellation: a staged file
            # whose upload did not finish has no owner and must not survive.
            self.discard(path)
            raise

        return StagedUpload(
            path=path, byte_size=byte_size, content_hash=digest.hexdigest(), header=header
        )

    def commit(self, staged: StagedUpload, document_id: uuid.UUID) -> None:
        """Move a staged file to its final path, atomically.

        Staging and documents share a root, hence a filesystem, which is what
        makes ``os.replace`` a rename rather than a copy.
        """
        destination = self.path_for(document_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged.path, destination)

    def discard(self, path: Path) -> None:
        """Remove a staged file. Never raises: the caller is already failing."""
        path.unlink(missing_ok=True)

    def delete(self, document_id: uuid.UUID) -> None:
        """Remove a document's bytes, tolerating their absence.

        ``missing_ok`` because this runs after the row is already gone. A file
        that vanished some other way must not turn a successful delete into a
        500 that suggests the document still exists.
        """
        self.path_for(document_id).unlink(missing_ok=True)


def get_storage(request: Request) -> DocumentStorage:
    """The app's storage, as a FastAPI dependency.

    Lives here rather than in a router for the same reason ``get_session``
    lives in ``app.db``: it belongs to the resource it hands out, and more than
    one router needs it. Importing Request at this level is consistent with
    ``app.db`` - the rule that keeps FastAPI out applies to services,
    repositories and models, not to the infrastructure they are handed.
    """
    storage: DocumentStorage = request.app.state.storage
    return storage
