"""The real embedding model, loaded once. Excluded from the default suite.

Everything else in the suite substitutes a deterministic fake, which is what
keeps `make test` fast, offline and predictable. That trade has one hole in it,
and this file is the patch: a fake cannot tell us whether
``all-MiniLM-L6-v2`` actually places a paraphrase closer than an unrelated
sentence, and *that* is the entire premise of retrieving by vector rather than
by keyword. If this file fails, the fakes are still green and the product does
not work.

The other thing only the real model can confirm is its own width. ``384`` is
compiled into the DDL of ``chunks.embedding``, so a model swap that changes it
is a migration and a re-embed. The assertion below is what turns that from a
comment into something CI can catch.

Run with:

    make test-slow      # or: pytest -m slow

The first run downloads roughly 90 MB into the HuggingFace cache and takes a
minute; later runs read it off disk. It needs a network exactly once, which is
why it is not in the default suite.
"""

from __future__ import annotations

import pytest

from app.config import DEFAULT_EMBEDDING_MODEL
from app.embedding import SentenceTransformerEmbedder
from app.models import EMBEDDING_DIMENSIONS

pytestmark = pytest.mark.slow

# One sentence, said three ways, plus one about something else entirely. No
# term overlap between the paraphrase and the original beyond stop words, so a
# keyword index would score them at zero and only a semantic one recovers them.
PASSAGE = "The learning rate controls how large a step gradient descent takes."
PARAPHRASE = "Step size determines how far each optimisation update moves the weights."
RELATED = "Choosing too large a step makes the optimiser diverge instead of converge."
UNRELATED = "The library on the north side of campus closes at eight on weekdays."


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    """One embedder for the whole module, so the weights are read once.

    Deliberately synchronous and un-loaded: the embedder loads itself on first
    use, so the first test in the module pays for it and the rest do not. A
    module-scoped *async* fixture would need a module-scoped event loop, and
    widening the loop scope for the whole suite to save one model load is a bad
    trade.
    """
    return SentenceTransformerEmbedder(DEFAULT_EMBEDDING_MODEL)


def similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity, which for unit vectors is the dot product.

    Deliberately not importing numpy: this is the arithmetic Postgres will do
    with ``<=>``, written out, so what the assertions below compare is what
    retrieval will actually rank on.
    """
    return sum(a * b for a, b in zip(left, right, strict=True))


async def test_the_model_matches_the_width_of_the_column(
    embedder: SentenceTransformerEmbedder,
) -> None:
    """The one assertion that guards a schema constant against a model swap.

    ``SentenceTransformerEmbedder`` also checks this at load time and refuses a
    mismatch; this is the test that notices when ``DEFAULT_EMBEDDING_MODEL`` is
    changed to something the column cannot hold.
    """
    vector = await embedder.embed_query("how wide is this model")

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert embedder.is_loaded is True
    assert embedder.load_error is None


async def test_vectors_come_back_normalised(embedder: SentenceTransformerEmbedder) -> None:
    """Unit length is what makes cosine distance and the dot product agree, and
    what ``vector_cosine_ops`` on the HNSW index assumes."""
    vectors = await embedder.embed_documents([PASSAGE, UNRELATED])

    for vector in vectors:
        assert similarity(vector, vector) == pytest.approx(1.0, abs=1e-5)


async def test_a_paraphrase_beats_an_unrelated_sentence(
    embedder: SentenceTransformerEmbedder,
) -> None:
    """The premise of the whole retrieval design, stated as an inequality.

    "Step size" and "learning rate" share no words. If a paraphrase does not
    land closer than a sentence about the library, vector search is buying
    nothing that a LIKE query would not, and the cost of the model is waste.
    """
    query = await embedder.embed_query("what does the learning rate do")
    paraphrase, unrelated = await embedder.embed_documents([PARAPHRASE, UNRELATED])

    assert similarity(query, paraphrase) > similarity(query, unrelated)
    # Not just ordered but clearly apart, so the test fails on a regression
    # rather than on a rounding difference between torch versions.
    assert similarity(query, paraphrase) - similarity(query, unrelated) > 0.2


async def test_the_ranking_is_by_topic_not_by_wording(
    embedder: SentenceTransformerEmbedder,
) -> None:
    """The ordering a search over these four sentences has to produce."""
    query = await embedder.embed_query("how big a step should the optimiser take")
    vectors = await embedder.embed_documents([PASSAGE, PARAPHRASE, RELATED, UNRELATED])

    ranked = sorted(
        zip([PASSAGE, PARAPHRASE, RELATED, UNRELATED], vectors, strict=True),
        key=lambda pair: similarity(query, pair[1]),
        reverse=True,
    )

    assert [text for text, _ in ranked][-1] == UNRELATED


async def test_the_same_text_always_embeds_the_same_way(
    embedder: SentenceTransformerEmbedder,
) -> None:
    """Stored vectors are compared against ones computed later, in another
    process. If encoding were not deterministic, a chunk's position in the
    index would drift away from what its text means."""
    first = await embedder.embed_documents([PASSAGE])
    second = await embedder.embed_documents([PASSAGE])

    assert first[0] == pytest.approx(second[0])


async def test_a_query_and_a_document_are_encoded_consistently(
    embedder: SentenceTransformerEmbedder,
) -> None:
    """MiniLM is symmetric - it declares no separate query prompt - so the two
    methods have to agree on identical text. An asymmetric model would not
    satisfy this, which is precisely why the two calls are kept distinct
    instead of collapsed into one ``embed()``."""
    query = await embedder.embed_query(PASSAGE)
    document = await embedder.embed_documents([PASSAGE])

    assert similarity(query, document[0]) == pytest.approx(1.0, abs=1e-4)
