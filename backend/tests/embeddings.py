"""A deterministic stand-in for the embedding model.

The real model is 90 MB of weights, a torch import and several seconds of load
time. Paying that in the default suite would mean tests that need a network on
their first run, take minutes, and produce vectors nobody can predict - so the
one test that exercises the real thing is marked ``slow`` and lives in
``tests/slow/``, and everything else uses this.

It is not a mock: it returns real, normalised 384-dimension vectors, and it
places texts that share words close together. That is enough for a ranking
assertion to mean something - a query about "gradient descent" really does come
back with the chunk about gradient descent first - while staying completely
predictable. What it deliberately cannot do is match paraphrases, which is the
one property only the real model has and the only thing ``tests/slow`` claims.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from app.models import EMBEDDING_DIMENSIONS

_TOKEN = re.compile(r"[a-z0-9]+")


def fake_vector(text: str, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """A hashed bag of words, normalised to unit length.

    ``hashlib`` rather than ``hash()``: the builtin is salted per process, so
    the same text would embed differently in two runs and a stored vector could
    never be compared against a freshly computed one.
    """
    vector = [0.0] * dimensions
    for token in _TOKEN.findall(text.lower()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        vector[int.from_bytes(digest, "big") % dimensions] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        # No words at all. Any fixed unit vector will do; a zero vector would
        # make cosine distance undefined and pgvector would reject the maths.
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class FakeEmbedder:
    """Implements ``app.embedding.Embedder`` with no model behind it."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self.model_name = "fake-bag-of-words"
        self.is_loaded = True
        self.load_error: str | None = None
        # What to raise instead of embedding, for the paths that only exist
        # because the real model can be unavailable.
        self.raises = raises
        self.embedded_documents: list[list[str]] = []
        self.embedded_queries: list[str] = []

    async def load(self) -> None:
        return None

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self.raises is not None:
            raise self.raises
        self.embedded_documents.append(list(texts))
        return [fake_vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        if self.raises is not None:
            raise self.raises
        self.embedded_queries.append(text)
        return fake_vector(text)
