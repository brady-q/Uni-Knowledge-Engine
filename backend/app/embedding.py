"""The embedding model, behind a narrow interface.

Text becomes a vector in exactly two places - the last stage of ingestion and
the first stage of retrieval - and both go through ``Embedder``. The protocol is
three methods wide on purpose: it is what lets every test in the default suite
substitute a deterministic fake instead of downloading half a gigabyte of
weights, and it is the seam a different model swaps in behind.

Lives at the top level rather than under ``services/`` for the same reason
``storage.py`` does: it is infrastructure the services are *handed*, and it owns
the FastAPI dependency that hands it over. The rule that keeps FastAPI out
applies to services, repositories and models.

Two properties of the real implementation shape everything here:

* **The model is enormous and slow to load.** It is loaded once per process,
  lazily, in a worker thread, and shared by every request. Startup kicks that
  off in the background rather than waiting for it, so the API answers
  ``/health`` while the weights are still being read.
* **Encoding is CPU-bound and synchronous.** Every call into the model runs in
  a worker thread. Inline, it would block the event loop - and therefore every
  other request this worker is serving - for the length of the batch.

Vectors are normalised at the point of creation, which is what makes cosine
distance and the dot product the same number and lets ``ix_chunks_embedding_hnsw``
(built with ``vector_cosine_ops``) serve the query.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from typing import Any, Protocol

import anyio
import anyio.to_thread
from fastapi import Request

from app.errors import EmbeddingError
from app.models import EMBEDDING_DIMENSIONS

logger = logging.getLogger(__name__)

# What a caller is told when the model is not usable. Deliberately vague about
# the cause - a missing HuggingFace cache and a torn download are the same
# problem from the outside - and deliberately explicit that retrying is the
# right response, because it usually is.
MODEL_UNAVAILABLE_DETAIL = "the embedding model is not available yet; try again shortly"


class Embedder(Protocol):
    """What the pipeline needs from an embedding model, and nothing more.

    Documents and queries are embedded through separate methods even though
    MiniLM treats them identically. Asymmetric models are the norm - E5 and BGE
    prepend an instruction to one side and not the other - and a single
    ``embed()`` would either silently drop that distinction or push it onto
    every caller. Sentence-Transformers reads the per-model prompts from the
    model's own config, so the split is free here and correct later.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def is_loaded(self) -> bool:
        """True once the model can answer without blocking on a load."""

    @property
    def load_error(self) -> str | None:
        """Why the last load attempt failed, for readiness to report."""

    async def load(self) -> None:
        """Load the model if it is not loaded. Never raises."""

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """One unit vector per text, in the order given."""

    async def embed_query(self, text: str) -> list[float]:
        """One unit vector for a search query."""


class SentenceTransformerEmbedder:
    """A Sentence-Transformers model, loaded once and shared by the process."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        dimensions: int = EMBEDDING_DIMENSIONS,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._dimensions = dimensions
        self._model: Any | None = None
        self._load_error: str | None = None
        # Concurrent first requests must not each load their own copy of the
        # weights: that is the same seconds of work and the same hundreds of
        # megabytes, N times over, for one shared result.
        self._lock = anyio.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    async def load(self) -> None:
        """Preload the model, swallowing failure.

        Called from the app lifespan, where raising would take down a startup
        that is otherwise fine. The reason is kept on ``load_error`` for
        ``/health/ready`` to report, and the next call that needs the model
        tries again - a failed download is usually transient.
        """
        with contextlib.suppress(EmbeddingError):
            await self._ensure_model()

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        # Short-circuited before the model is touched: a document that chunked
        # to nothing should not be what triggers a load.
        if not texts:
            return []
        model = await self._ensure_model()
        return await anyio.to_thread.run_sync(self._encode_documents, model, list(texts))

    async def embed_query(self, text: str) -> list[float]:
        model = await self._ensure_model()
        vectors = await anyio.to_thread.run_sync(self._encode_query, model, text)
        return vectors[0]

    async def _ensure_model(self) -> Any:
        """The loaded model, loading it first if necessary.

        The check outside the lock is not redundant with the one inside it: the
        fast path must not serialise every embed call behind a lock that exists
        only for the one-time load, and the second check is what stops the
        callers that queued up behind that load from repeating it.
        """
        loaded = self._model
        if loaded is not None:
            return loaded

        async with self._lock:
            if self._model is None:
                self._model = await self._load_checked()
                self._load_error = None
                logger.info("embedding model %s loaded", self._model_name)
            return self._model

    async def _load_checked(self) -> Any:
        """Construct the model and verify it fits the column it will be stored in."""
        try:
            # abandon_on_cancel, uniquely on this call: by default anyio waits
            # for the worker thread before letting a cancellation propagate,
            # which would make shutting down during a first-run download block
            # until the whole 90 MB had arrived. The result of an abandoned
            # load is discarded, and huggingface_hub writes to a temporary file
            # and renames, so an interrupted download leaves no partial file
            # behind. Encoding is deliberately *not* abandonable: those calls
            # are short and somebody is waiting on the answer.
            model = await anyio.to_thread.run_sync(self._construct, abandon_on_cancel=True)
            dimensions = int(model.get_embedding_dimension())
        # A failed torch import, a torn download and an unwritable cache all
        # surface differently and all mean the same thing to a caller.
        except Exception as exc:
            logger.exception("could not load embedding model %s", self._model_name)
            raise self._failed(f"{type(exc).__name__}: {exc}") from exc

        if dimensions != self._dimensions:
            # A silent mismatch would be an INSERT rejected by Postgres for
            # every chunk of every document, reported as an unexpected
            # ingestion failure. Caught once, at the source, and named.
            raise self._failed(
                f"{self._model_name} emits {dimensions} dimensions, "
                f"but chunks.embedding is vector({self._dimensions}); "
                f"changing the model needs a migration and a re-embed"
            )
        return model

    def _failed(self, reason: str) -> EmbeddingError:
        """Record why the model is unusable and build the error to raise.

        The reason goes to readiness, which is a diagnostic for whoever is
        running this; the exception's own detail is what a client sees.
        """
        self._load_error = reason[:300]
        return EmbeddingError(MODEL_UNAVAILABLE_DETAIL)

    def _construct(self) -> Any:
        """Read the weights off disk, downloading them if this is the first run.

        ``sentence_transformers`` is imported here rather than at module scope
        because importing it imports torch, which costs seconds of startup and
        is pure waste in a process that never embeds anything - every unit test
        in the suite, for instance. The dependency is not optional; the cost of
        pulling it in is just deferred to the process that actually pays for it.
        """
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self._model_name)

    def _encode_documents(self, model: Any, texts: list[str]) -> list[list[float]]:
        matrix = model.encode_document(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return _as_vectors(matrix)

    def _encode_query(self, model: Any, text: str) -> list[list[float]]:
        matrix = model.encode_query(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return _as_vectors(matrix)


def _as_vectors(matrix: Any) -> list[list[float]]:
    """Convert the model's numpy output into plain lists of floats.

    asyncpg and pgvector both want a sequence of Python floats, and float32
    numpy scalars are not that. Doing it here keeps numpy out of every layer
    above: nothing downstream of this module knows the model uses it.
    """
    return [[float(value) for value in row] for row in matrix]


def get_embedder(request: Request) -> Embedder:
    """The app's embedder, as a FastAPI dependency.

    The single override point for tests: replacing this replaces the model for
    the search path *and* for ingestion, which reaches it through the scheduler
    dependency in ``app.routers.documents``.
    """
    embedder: Embedder = request.app.state.embedder
    return embedder
