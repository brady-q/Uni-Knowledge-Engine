"""The chunker, as a pure function.

No PDF, no database, no file system: every case here is a literal list of pages
in and a literal list of chunks out. That is the point of keeping chunking free
of I/O - the component most likely to be retuned is the one whose behaviour can
be pinned down exactly.

The properties worth protecting are the ones the citation guarantee rests on:
page ranges are derived from the source and never invented, and no extracted
text is silently dropped.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from app.services.chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MIN_CHUNK_CHARS,
    chunk_pages,
)
from app.services.extraction import PageText


def page(number: int, *paragraphs: str) -> PageText:
    return PageText(number=number, text="\n\n".join(paragraphs))


def sentences(count: int, *, marker: str = "x") -> str:
    """A paragraph of ``count`` short sentences, each individually identifiable."""
    return " ".join(f"Sentence {index} about {marker}." for index in range(count))


def test_no_pages_yields_no_chunks() -> None:
    assert chunk_pages([]) == []


def test_blank_pages_yield_no_chunks() -> None:
    assert chunk_pages([page(1, "   "), page(2, "\n\n")]) == []


def test_short_document_is_one_chunk_on_its_own_page() -> None:
    chunks = chunk_pages([page(1, "Gradient descent minimises a loss function.")])

    assert len(chunks) == 1
    assert chunks[0].ordinal == 0
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].text == "Gradient descent minimises a loss function."


def test_ordinals_are_dense_and_start_at_zero() -> None:
    chunks = chunk_pages([page(number, sentences(30)) for number in range(1, 6)])

    assert len(chunks) > 1
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_no_chunk_exceeds_the_size_limit() -> None:
    chunks = chunk_pages([page(number, sentences(40)) for number in range(1, 4)])

    assert chunks
    assert all(len(chunk.text) <= CHUNK_SIZE for chunk in chunks)


def test_page_numbers_never_run_backwards() -> None:
    """Chunks are emitted in reading order, so their pages must be monotonic.

    A citation resolved from a chunk whose range went backwards would point at
    a page the passage does not appear on.
    """
    chunks = chunk_pages([page(number, sentences(25)) for number in range(1, 8)])

    for chunk in chunks:
        assert 1 <= chunk.page_start <= chunk.page_end
    assert [chunk.page_start for chunk in chunks] == sorted(chunk.page_start for chunk in chunks)


def test_page_range_is_taken_from_the_source_not_guessed() -> None:
    """Every page's text ends up in a chunk that claims that page."""
    pages = [page(number, f"Unique marker {number}. {sentences(3)}") for number in range(1, 5)]

    chunks = chunk_pages(pages)

    for number in range(1, 5):
        owners = [
            chunk
            for chunk in chunks
            if f"Unique marker {number}." in chunk.text
            and chunk.page_start <= number <= chunk.page_end
        ]
        assert owners, f"page {number}'s text is in no chunk claiming page {number}"


def test_a_chunk_may_span_a_page_break() -> None:
    """A paragraph continued across pages belongs in one passage, citing both.

    Two half-size pages fit in a single chunk, and that chunk has to say so.
    """
    filler = sentences(15)
    chunks = chunk_pages([page(1, filler), page(2, filler)])

    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_consecutive_chunks_overlap() -> None:
    """Otherwise a passage straddling a boundary is retrievable by neither half.

    Each chunk has to begin with text the previous one ended with, and that
    repeated run has to be near the configured overlap - not one stray word.
    """
    chunks = chunk_pages([page(1, sentences(60))])

    assert len(chunks) > 1
    for earlier, later in pairwise(chunks):
        shared = _shared_run(earlier.text, later.text)
        assert shared, "no part of the previous chunk's tail was carried forward"
        assert len(shared) <= CHUNK_OVERLAP
        # Sentence-sized granularity, so the overlap lands near the target
        # rather than exactly on it. A tenth of it would mean the carry is
        # effectively not happening.
        assert len(shared) > CHUNK_OVERLAP // 2


def _shared_run(earlier: str, later: str) -> str:
    """The longest prefix of ``later`` that ends ``earlier``."""
    for length in range(min(len(earlier), len(later)), 0, -1):
        if earlier.endswith(later[:length]):
            return later[:length]
    return ""


def test_no_text_is_lost() -> None:
    """Every sentence in the input appears in at least one chunk."""
    pages = [page(number, sentences(20, marker=f"p{number}")) for number in range(1, 4)]

    combined = " ".join(chunk.text for chunk in chunk_pages(pages))

    for number in range(1, 4):
        for index in range(20):
            assert f"Sentence {index} about p{number}." in combined


def test_a_paragraph_larger_than_a_chunk_is_split_on_sentences() -> None:
    chunks = chunk_pages([page(1, sentences(80))])

    assert len(chunks) > 1
    assert all(len(chunk.text) <= CHUNK_SIZE for chunk in chunks)
    # Sentence boundaries were respected, so no chunk begins mid-word.
    assert all(chunk.text.startswith("Sentence") for chunk in chunks)


def test_text_with_no_sentence_or_word_boundaries_is_hard_split() -> None:
    """A base64 blob or a de-spaced table still has to fit in the embedder."""
    blob = "A" * (CHUNK_SIZE * 3 + 17)

    chunks = chunk_pages([page(1, blob)])

    assert all(len(chunk.text) <= CHUNK_SIZE for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks).count("A") >= len(blob)


def test_hard_wrapped_lines_are_reflowed() -> None:
    """PDF extraction breaks every visual line; a chunk should read as prose."""
    wrapped = PageText(number=1, text="The gradient\nof the loss\nis computed\nby backprop.")

    chunks = chunk_pages([wrapped])

    assert chunks[0].text == "The gradient of the loss is computed by backprop."


def test_paragraph_boundaries_survive_as_line_breaks() -> None:
    chunks = chunk_pages([page(1, "First paragraph.", "Second paragraph.")])

    assert chunks[0].text == "First paragraph.\nSecond paragraph."


def test_a_runt_final_chunk_is_folded_into_its_predecessor() -> None:
    """A trailing heading must not become its own low-value row.

    It is merged rather than dropped, because dropping it would lose text, and
    the merged chunk's page range widens to cover the page it came from.
    """
    tail = "Appendix"
    chunks = chunk_pages([page(1, sentences(60)), page(2, tail)])

    assert len(tail) < MIN_CHUNK_CHARS
    assert chunks[-1].text.endswith(tail)
    assert chunks[-1].page_end == 2
    assert not any(len(chunk.text) < MIN_CHUNK_CHARS for chunk in chunks)
    assert all(len(chunk.text) <= CHUNK_SIZE for chunk in chunks)


def test_a_short_document_is_still_allowed_to_be_one_short_chunk() -> None:
    """The runt rule folds a tail into a predecessor; with none, text survives."""
    chunks = chunk_pages([page(1, "Short.")])

    assert [chunk.text for chunk in chunks] == ["Short."]


@pytest.mark.parametrize("paragraph_count", [1, 2, 5, 20, 100])
def test_chunking_terminates_for_any_shape(paragraph_count: int) -> None:
    """Overlap carries text forward, so a bad floor could loop or duplicate.

    ``_overlap_tail`` always leaves the first segment behind for exactly this
    reason; the assertion is that output stays proportional to input.
    """
    pages = [page(1, *[sentences(4)] * paragraph_count)]

    chunks = chunk_pages(pages)

    total = sum(len(chunk.text) for chunk in chunks)
    assert total <= len(pages[0].text) * 2 + CHUNK_SIZE


def test_the_runt_rule_never_produces_an_oversized_chunk() -> None:
    """Merging is skipped when it would breach the embedder's window.

    A full chunk plus a runt does not fit, so the runt is left alone. Losing
    the tail of a chunk to truncation at embedding time would be worse than one
    short row.
    """
    chunks = chunk_pages([page(1, "A" * CHUNK_SIZE), page(2, "tail")])

    assert all(len(chunk.text) <= CHUNK_SIZE for chunk in chunks)
    assert chunks[-1].text == "tail"
