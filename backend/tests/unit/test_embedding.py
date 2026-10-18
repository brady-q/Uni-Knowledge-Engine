"""The embedder's contract, with a stub model in place of the real one.

``_construct`` is the single seam: overriding it means these tests cover the
loading, locking, validation and conversion this class actually implements,
without importing torch or downloading anything. Whether
``all-MiniLM-L6-v2`` produces good vectors is not this file's question - that is
``tests/slow/test_real_embedding_model.py``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from app.embedding import SentenceTransformerEmbedder
from app.errors import EmbeddingError
from app.models import EMBEDDING_DIMENSIONS


class StubModel:
    """The three methods ``SentenceTransformerEmbedder`` calls on a model."""

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self._dimensions = dimensions
        self.document_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.query_calls: list[tuple[list[str], dict[str, Any]]] = []

    def get_embedding_dimension(self) -> int:
        return self._dimensions

    def encode_document(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.document_calls.append((texts, kwargs))
        return [self._row(index, text) for index, text in enumerate(texts)]

    def encode_query(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.query_calls.append((texts, kwargs))
        return [self._row(index, text) for index, text in enumerate(texts)]

    def _row(self, index: int, text: str) -> list[float]:
        """A vector that identifies which text it came from and in what order."""
        row = [0.0] * self._dimensions
        row[0] = float(index)
        row[1] = float(len(text))
        return row


class StubEmbedder(SentenceTransformerEmbedder):
    """Loads a StubModel instead of reading weights off disk."""

    def __init__(
        self,
        *,
        model: StubModel | None = None,
        fail_times: int = 0,
        load_delay: float = 0.0,
        batch_size: int = 4,
    ) -> None:
        super().__init__("stub/model", batch_size=batch_size)
        self.model = model or StubModel()
        self.constructed = 0
        self._fail_times = fail_times
        self._load_delay = load_delay

    def _construct(self) -> Any:
        # Runs in a worker thread, so a blocking sleep here is exactly what a
        # real load looks like from the event loop's point of view.
        self.constructed += 1
        if self._load_delay:
            time.sleep(self._load_delay)
        if self.constructed <= self._fail_times:
            raise RuntimeError("no route to huggingface.co")
        return self.model


# --- embedding ----------------------------------------------------------------


async def test_documents_are_embedded_in_the_order_given() -> None:
    """Chunk N's vector has to be chunk N's, or every citation points elsewhere."""
    embedder = StubEmbedder()

    vectors = await embedder.embed_documents(["first", "second", "third"])

    assert [vector[0] for vector in vectors] == [0.0, 1.0, 2.0]
    assert [vector[1] for vector in vectors] == [5.0, 6.0, 5.0]
    assert all(len(vector) == EMBEDDING_DIMENSIONS for vector in vectors)


async def test_vectors_are_plain_floats() -> None:
    """asyncpg and pgvector take Python floats; numpy scalars stop here."""
    vectors = await StubEmbedder().embed_documents(["text"])

    assert all(type(value) is float for value in vectors[0])


async def test_documents_are_normalised_and_batched_as_configured() -> None:
    """Unit vectors are what make cosine distance and the dot product agree.

    ``ix_chunks_embedding_hnsw`` is built with ``vector_cosine_ops``; asking the
    model for un-normalised output would leave magnitude in the comparison and
    silently change what "closest" means.
    """
    embedder = StubEmbedder(batch_size=8)

    await embedder.embed_documents(["a", "b"])

    texts, kwargs = embedder.model.document_calls[0]
    assert texts == ["a", "b"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 8
    assert kwargs["show_progress_bar"] is False


async def test_a_query_uses_the_query_side_of_the_model() -> None:
    """Asymmetric models prepend an instruction to queries and not to passages."""
    embedder = StubEmbedder()

    vector = await embedder.embed_query("what is a learning rate")

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert embedder.model.query_calls[0][0] == ["what is a learning rate"]
    assert embedder.model.query_calls[0][1]["normalize_embeddings"] is True
    assert embedder.model.document_calls == []


async def test_embedding_nothing_does_not_load_the_model() -> None:
    """A document that chunked to nothing must not be what pulls in the weights."""
    embedder = StubEmbedder()

    assert await embedder.embed_documents([]) == []
    assert embedder.constructed == 0
    assert embedder.is_loaded is False


# --- loading ------------------------------------------------------------------


async def test_the_model_is_loaded_once_and_reused() -> None:
    embedder = StubEmbedder()

    await embedder.embed_documents(["one"])
    await embedder.embed_query("two")

    assert embedder.constructed == 1
    assert embedder.is_loaded is True
    assert embedder.load_error is None


async def test_concurrent_first_calls_load_the_model_once() -> None:
    """Without the lock, N simultaneous requests each read their own copy.

    That is the same seconds of work and the same hundreds of megabytes, N
    times over, for a result they are all going to share.
    """
    embedder = StubEmbedder(load_delay=0.05)

    await asyncio.gather(*(embedder.embed_query(f"query {index}") for index in range(5)))

    assert embedder.constructed == 1


async def test_a_model_of_the_wrong_width_is_refused() -> None:
    """Otherwise every INSERT fails, and the reason is nowhere near the cause."""
    embedder = StubEmbedder(model=StubModel(dimensions=768))

    with pytest.raises(EmbeddingError):
        await embedder.embed_documents(["text"])

    assert embedder.is_loaded is False
    assert embedder.load_error is not None
    assert "768" in embedder.load_error
    assert str(EMBEDDING_DIMENSIONS) in embedder.load_error


async def test_a_failed_load_is_reported_and_then_retried() -> None:
    """A torn download is transient; refusing to try again would need a restart."""
    embedder = StubEmbedder(fail_times=1)

    with pytest.raises(EmbeddingError):
        await embedder.embed_query("first attempt")
    assert embedder.load_error is not None
    assert "RuntimeError" in embedder.load_error

    vector = await embedder.embed_query("second attempt")

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert embedder.constructed == 2
    assert embedder.load_error is None


async def test_preloading_never_raises() -> None:
    """It is called from the app lifespan, where raising takes down startup."""
    embedder = StubEmbedder(fail_times=1)

    await embedder.load()

    assert embedder.is_loaded is False
    assert embedder.load_error is not None


async def test_preloading_leaves_the_model_ready() -> None:
    embedder = StubEmbedder()

    await embedder.load()

    assert embedder.is_loaded is True
    assert embedder.constructed == 1
    # And the first real call does not load it again.
    await embedder.embed_query("query")
    assert embedder.constructed == 1
