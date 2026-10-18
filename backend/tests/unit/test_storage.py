"""The blob store, against a real temporary directory.

Real files rather than a mocked filesystem: what is being checked is that the
staging-then-rename dance actually leaves the right things on disk, and a mock
would only confirm that the calls were made in the order the test expected.
"""

from __future__ import annotations

import hashlib
import uuid
from io import BytesIO
from pathlib import Path

import pytest

from app.errors import PayloadTooLargeError
from app.storage import DocumentStorage


class FakeUpload:
    """The narrow slice of Starlette's UploadFile that DocumentStorage uses."""

    def __init__(self, data: bytes, filename: str | None = "notes.pdf") -> None:
        self.filename = filename
        self._buffer = BytesIO(data)

    async def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)


@pytest.fixture
def storage(tmp_path: Path) -> DocumentStorage:
    store = DocumentStorage(tmp_path / "storage")
    store.prepare()
    return store


async def test_staging_records_size_hash_and_header(storage: DocumentStorage) -> None:
    data = b"%PDF-1.7\nbody bytes here"

    staged = await storage.stage(FakeUpload(data), max_bytes=1024)

    assert staged.byte_size == len(data)
    assert staged.content_hash == hashlib.sha256(data).hexdigest()
    assert staged.header == b"%PDF-1.7"
    assert staged.path.read_bytes() == data


async def test_the_header_is_captured_across_read_boundaries(
    storage: DocumentStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reads are chunked, and the magic number can straddle two of them."""
    monkeypatch.setattr("app.storage.READ_CHUNK_BYTES", 3)
    data = b"%PDF-1.7 and then some more content"

    staged = await storage.stage(FakeUpload(data), max_bytes=1024)

    assert staged.header == b"%PDF-1.7"
    assert staged.content_hash == hashlib.sha256(data).hexdigest()


async def test_an_empty_upload_stages_cleanly(storage: DocumentStorage) -> None:
    """Rejecting it is the service's call; storage just reports what it got."""
    staged = await storage.stage(FakeUpload(b""), max_bytes=1024)

    assert staged.byte_size == 0
    assert staged.header == b""


async def test_an_oversized_upload_is_cut_off_and_leaves_nothing_behind(
    storage: DocumentStorage,
) -> None:
    """The limit is enforced while streaming, not after the whole file lands."""
    with pytest.raises(PayloadTooLargeError):
        await storage.stage(FakeUpload(b"x" * 5000), max_bytes=100)

    assert list((storage.root / "staging").iterdir()) == []


async def test_a_failed_read_leaves_nothing_behind(storage: DocumentStorage) -> None:
    class Exploding:
        filename: str | None = "boom.pdf"

        async def read(self, size: int = -1) -> bytes:
            raise OSError("connection reset")

    with pytest.raises(OSError, match="connection reset"):
        await storage.stage(Exploding(), max_bytes=1024)

    assert list((storage.root / "staging").iterdir()) == []


async def test_commit_moves_the_staged_file_to_the_document_path(
    storage: DocumentStorage,
) -> None:
    document_id = uuid.uuid4()
    staged = await storage.stage(FakeUpload(b"%PDF-1.7 x"), max_bytes=1024)

    storage.commit(staged, document_id)

    assert storage.path_for(document_id).read_bytes() == b"%PDF-1.7 x"
    assert not staged.path.exists()
    assert list((storage.root / "staging").iterdir()) == []


def test_paths_are_sharded_and_derived_from_the_id_alone(storage: DocumentStorage) -> None:
    """No query is needed to find a document's bytes, and no directory grows unbounded."""
    document_id = uuid.UUID("0123abcd-0000-0000-0000-000000000000")

    path = storage.path_for(document_id)

    assert path.parent.name == "01"
    assert path.name == f"{document_id.hex}.pdf"
    assert storage.path_for(document_id) == path


def test_two_documents_never_share_a_path(storage: DocumentStorage) -> None:
    ids = [uuid.uuid4() for _ in range(200)]

    assert len({storage.path_for(one) for one in ids}) == len(ids)


async def test_delete_removes_the_bytes(storage: DocumentStorage) -> None:
    document_id = uuid.uuid4()
    staged = await storage.stage(FakeUpload(b"%PDF-1.7"), max_bytes=1024)
    storage.commit(staged, document_id)

    storage.delete(document_id)

    assert not storage.path_for(document_id).exists()


def test_deleting_absent_bytes_is_not_an_error(storage: DocumentStorage) -> None:
    """It runs after the row is already gone; it must not resurrect a 500."""
    storage.delete(uuid.uuid4())


async def test_staging_names_do_not_collide(storage: DocumentStorage) -> None:
    """Two concurrent uploads of identical bytes must not overwrite each other."""
    first = await storage.stage(FakeUpload(b"%PDF-1.7 same"), max_bytes=1024)
    second = await storage.stage(FakeUpload(b"%PDF-1.7 same"), max_bytes=1024)

    assert first.path != second.path
    assert first.content_hash == second.content_hash
