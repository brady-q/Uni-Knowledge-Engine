"""Embedding and retrieval against a real Postgres with real pgvector.

The unit tests prove the embedder's contract and the router's contract with a
fake on the other side. These prove the parts a fake cannot: that a 384-float
vector survives a round trip through the ``vector`` column, that the
``<=>`` ordering and the course join actually produce the ranking the API
claims, that a chunk with no vector is excluded rather than ranked last, and
that ``ready`` now means searchable.

Vectors come from ``tests.embeddings.FakeEmbedder``: hashed bags of words, so
texts sharing terms with the query really are closer and the expected ordering
is something a reader of this file can work out. The paraphrase matching only
the real model does is covered in ``tests/slow``.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document, DocumentStatus
from app.schemas import DEFAULT_SEARCH_LIMIT
from tests.embeddings import FakeEmbedder
from tests.pdfs import make_empty_pdf, make_pdf

pytestmark = pytest.mark.integration

# Three topics, worded so that a bag-of-words query hits one of them and misses
# the others. Each page is padded past CHUNK_SIZE so that a topic is worth
# several chunks: three short pages would collapse into a single chunk covering
# all of them, and every ranking assertion below would be vacuous.
GRADIENT = [
    "Gradient descent iteratively adjusts parameters in the direction that most "
    "reduces the loss, and the learning rate controls the size of each step.",
    "Setting the learning rate too large makes gradient descent diverge instead "
    "of converging on a minimum of the loss surface.",
]
KERNEL = [
    "A kernel computes an inner product in a feature space without ever forming "
    "the coordinates of that space explicitly.",
    "The radial basis kernel corresponds to an infinite dimensional feature map, "
    "which is why it cannot be written down directly.",
]
BAYES = [
    "The posterior distribution is proportional to the likelihood multiplied by "
    "the prior, which is the entire content of Bayes rule.",
    "A conjugate prior keeps the posterior in the same family as the prior, so "
    "updating is arithmetic rather than integration.",
]


def page(sentences: list[str], repeats: int = 5) -> list[str]:
    """One page of paragraphs, all on the same topic and long enough to chunk."""
    return [f"Section {index}. {' '.join(sentences)}" for index in range(repeats)]


GRADIENT_PAGE = page(GRADIENT)
KERNEL_PAGE = page(KERNEL)
BAYES_PAGE = page(BAYES)


async def create_course(client: AsyncClient, code: str = "CS229") -> str:
    response = await client.post(
        "/api/v1/courses", json={"code": code, "title": "Machine Learning"}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def upload(
    client: AsyncClient,
    course_id: str,
    pages: list[list[str]],
    name: str = "lecture.pdf",
) -> dict[str, Any]:
    response = await client.post(
        f"/api/v1/courses/{course_id}/documents",
        files={"file": (name, make_pdf(pages), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


async def search(client: AsyncClient, course_id: str, query: str, **params: Any) -> dict[str, Any]:
    response = await client.get(
        f"/api/v1/courses/{course_id}/search", params={"q": query, **params}
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


# --- ingestion now embeds -----------------------------------------------------


async def test_every_chunk_of_a_ready_document_has_a_vector(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """``ready`` has to mean searchable, or a client cannot tell when to search."""
    course_id = await create_course(api_client)

    created = await upload(api_client, course_id, [GRADIENT_PAGE, KERNEL_PAGE])

    document = (
        await db_session.execute(select(Document).where(Document.id == uuid.UUID(created["id"])))
    ).scalar_one()
    assert document.status is DocumentStatus.READY
    assert document.error is None

    chunks = (
        (await db_session.execute(select(Chunk).where(Chunk.document_id == document.id)))
        .scalars()
        .all()
    )
    assert chunks
    for chunk in chunks:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 384
        # Normalised on the way in. If this drifts, cosine distance stops
        # agreeing with the dot product and the HNSW opclass is wrong.
        assert sum(float(value) ** 2 for value in chunk.embedding) == pytest.approx(1.0)


async def test_the_document_is_embedded_exactly_once(
    api_client: AsyncClient, embedder: FakeEmbedder, db_session: AsyncSession
) -> None:
    """One batch per document, not one call per chunk: the model is the
    expensive part of ingestion and it is cheapest per chunk in bulk."""
    course_id = await create_course(api_client)

    created = await upload(api_client, course_id, [GRADIENT_PAGE, KERNEL_PAGE, BAYES_PAGE])

    total = (
        await db_session.execute(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_id == uuid.UUID(created["id"]))
        )
    ).scalar_one()
    assert len(embedder.embedded_documents) == 1
    assert len(embedder.embedded_documents[0]) == total


async def test_a_failed_document_never_reaches_the_embedder(
    api_client: AsyncClient, embedder: FakeEmbedder
) -> None:
    """Extraction failed, so there is nothing to embed and no cost to pay."""
    course_id = await create_course(api_client)

    response = await api_client.post(
        f"/api/v1/courses/{course_id}/documents",
        files={"file": ("scan.pdf", make_empty_pdf(2), "application/pdf")},
    )

    assert response.status_code == 201
    assert embedder.embedded_documents == []


# --- retrieval ----------------------------------------------------------------


async def test_the_closest_passage_comes_first(api_client: AsyncClient) -> None:
    course_id = await create_course(api_client)
    await upload(api_client, course_id, [GRADIENT_PAGE, KERNEL_PAGE, BAYES_PAGE])

    body = await search(api_client, course_id, "learning rate gradient descent")

    assert body["items"], "a course with three pages of text returned nothing"
    top = body["items"][0]
    assert "gradient descent" in top["text"].lower()
    # Scores are similarities: monotonically decreasing, and never above 1.
    scores = [item["score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= score <= 1.0 + 1e-6 for score in scores)


async def test_a_result_cites_a_page_the_text_is_actually_on(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """The property the whole project exists to guarantee, on the read path.

    The page range in a result is read out of the chunk row, so it has to be
    the range recorded for the chunk whose text was returned. Anything else
    means a citation pointing at a page that does not contain the quote.
    """
    course_id = await create_course(api_client)
    await upload(api_client, course_id, [GRADIENT_PAGE, KERNEL_PAGE, BAYES_PAGE])

    body = await search(api_client, course_id, "conjugate prior posterior")

    for item in body["items"]:
        chunk = (
            await db_session.execute(select(Chunk).where(Chunk.id == uuid.UUID(item["chunk_id"])))
        ).scalar_one()
        assert item["text"] == chunk.text
        assert (item["page_start"], item["page_end"]) == (chunk.page_start, chunk.page_end)
        assert 1 <= item["page_start"] <= item["page_end"] <= 3


async def test_results_are_scoped_to_the_course(api_client: AsyncClient) -> None:
    """A chunk from an unrelated unit in an answer's citations is a correctness
    failure, not a ranking one - so the scope is a join, not a filter a caller
    can forget to send."""
    mine = await create_course(api_client, "CS229")
    theirs = await create_course(api_client, "CS230")
    await upload(api_client, mine, [KERNEL_PAGE])
    await upload(api_client, theirs, [GRADIENT_PAGE])

    body = await search(api_client, mine, "learning rate gradient descent")

    assert body["items"]
    assert all("gradient descent" not in item["text"].lower() for item in body["items"])


async def test_results_span_every_document_in_the_course(api_client: AsyncClient) -> None:
    """Retrieval is over the course, not over one file at a time."""
    course_id = await create_course(api_client)
    first = await upload(api_client, course_id, [GRADIENT_PAGE], name="week1.pdf")
    second = await upload(api_client, course_id, [BAYES_PAGE], name="week2.pdf")

    gradient = await search(api_client, course_id, "learning rate gradient descent")
    bayes = await search(api_client, course_id, "conjugate prior posterior")

    assert gradient["items"][0]["document_id"] == first["id"]
    assert gradient["items"][0]["filename"] == "week1.pdf"
    assert bayes["items"][0]["document_id"] == second["id"]
    assert bayes["items"][0]["filename"] == "week2.pdf"


async def test_the_limit_bounds_the_result_set(api_client: AsyncClient) -> None:
    course_id = await create_course(api_client)
    await upload(api_client, course_id, [GRADIENT_PAGE, KERNEL_PAGE, BAYES_PAGE])

    body = await search(api_client, course_id, "kernel feature space", limit=2)

    assert body["limit"] == 2
    assert len(body["items"]) == 2


async def test_a_chunk_without_a_vector_is_excluded(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """NULL means "not embedded yet", not "distant". Ranking it at the bottom
    of a result set that claims to be sorted by relevance would be a lie, and
    with a small limit it would push out a chunk that is actually relevant."""
    course_id = await create_course(api_client)
    created = await upload(api_client, course_id, [GRADIENT_PAGE])
    document_id = uuid.UUID(created["id"])

    unembedded = Chunk(
        document_id=document_id,
        # Past every ordinal the chunker produced, to stay clear of
        # uq_chunks(document_id, ordinal).
        ordinal=10_000,
        page_start=1,
        page_end=1,
        text="The learning rate gradient descent loss parameters direction.",
        embedding=None,
    )
    db_session.add(unembedded)
    await db_session.commit()

    body = await search(api_client, course_id, "learning rate gradient descent")

    assert body["items"]
    assert all(item["chunk_id"] != str(unembedded.id) for item in body["items"])


async def test_a_course_with_no_documents_returns_nothing(api_client: AsyncClient) -> None:
    course_id = await create_course(api_client)

    body = await search(api_client, course_id, "anything at all")

    assert body["items"] == []
    assert body["limit"] == DEFAULT_SEARCH_LIMIT


async def test_searching_a_missing_course_is_404(api_client: AsyncClient) -> None:
    response = await api_client.get(
        f"/api/v1/courses/{uuid.uuid4()}/search", params={"q": "gradient descent"}
    )

    assert response.status_code == 404


async def test_a_deleted_document_leaves_the_index(api_client: AsyncClient) -> None:
    """Its chunks go with it by ON DELETE CASCADE, so they cannot be cited."""
    course_id = await create_course(api_client)
    kernels = await upload(api_client, course_id, [KERNEL_PAGE], name="kernels.pdf")
    await upload(api_client, course_id, [BAYES_PAGE], name="bayes.pdf")
    assert (await search(api_client, course_id, "kernel inner product"))["items"]

    deleted = await api_client.delete(f"/api/v1/courses/{course_id}/documents/{kernels['id']}")
    assert deleted.status_code == 204

    body = await search(api_client, course_id, "kernel inner product")
    assert all(item["document_id"] != kernels["id"] for item in body["items"])
