"""Splitting extracted pages into retrievable passages.

A pure function over data: no database, no file system, no configuration
lookup. That is deliberate - chunking is the part of the pipeline most likely
to be tuned, and the only way to tune it confidently is for every case to be
reproducible from a literal list of pages in a test.

Two properties matter more than the exact boundaries chosen:

* Every chunk knows the page range it came from, and that range is derived from
  the source text rather than estimated. This is the foundation the citation
  guarantee is built on: the answer path never computes a page number, it reads
  one that was recorded here.
* Chunks overlap. A passage that straddles a boundary would otherwise be
  retrievable by neither of the two chunks that each hold half of it.

The shape of the algorithm follows from the second property. Text is broken
down to *sentences* before anything is assembled, not to paragraphs, because
overlap is carried as whole units: if a unit were as large as a chunk, chunks
would be single-unit and there would be nothing to carry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.extraction import PageText

# Sizes are in characters, not tokens. The embedding model's window is 256
# word-piece tokens, and English prose runs about four characters per token, so
# 1000 characters lands just under it with room for the model's own special
# tokens. Measuring in real tokens would mean loading a tokenizer here, which
# would couple the chunker to a specific model and make these tests slow for no
# gain in a bound that is approximate either way.
CHUNK_SIZE = 1000

# Carried from the end of each chunk into the start of the next. Roughly one or
# two sentences: enough that a passage split across a boundary survives intact
# in one of the two, without inflating the index by much.
CHUNK_OVERLAP = 150

# A chunk shorter than this is not independently retrievable - it is a heading
# or a stray line, and as its own row it mostly generates false matches. Only
# the final chunk can be this short, and it is folded into its predecessor
# where that fits rather than dropped, which would lose text.
MIN_CHUNK_CHARS = 100

# One or more blank lines: the paragraph boundaries extraction preserved.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")

# After ., ! or ? plus whitespace. Crude - it splits "Fig. 3" - but the cost of
# a wrong boundary is only that two sentences swap chunks, and every chunk
# overlaps its neighbour anyway.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")

_WHITESPACE = re.compile(r"\s+")

# How a piece of text attaches to the one before it. A sentence continues its
# paragraph; a new paragraph starts a new line. Both are one character, which
# is what lets length accounting just add one.
_SENTENCE_JOIN = " "
_PARAGRAPH_JOIN = "\n"


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One row-to-be of ``chunks``, before it has an id or an embedding."""

    ordinal: int
    page_start: int
    page_end: int
    text: str


@dataclass(frozen=True, slots=True)
class _Segment:
    """An indivisible piece of text, the page it came from, and how it joins.

    ``join`` is the separator to put *before* this segment, and is ignored when
    the segment happens to start a chunk. Carrying it here rather than deciding
    at assembly time is what keeps a chunk reading as prose: sentences that
    were one paragraph are rejoined with a space, not with a line break.
    """

    page: int
    text: str
    join: str


def chunk_pages(pages: list[PageText]) -> list[TextChunk]:
    """Turn extracted pages into ordered, overlapping chunks.

    Chunks may span pages: a paragraph continuing across a page break belongs
    in one passage, and splitting it there would be an artefact of the PDF's
    layout rather than of the text's meaning. When that happens the chunk
    records the full range, so a citation points at both pages.
    """
    segments = _segment(pages)
    if not segments:
        return []

    chunks: list[TextChunk] = []
    current: list[_Segment] = []
    length = 0

    for segment in segments:
        addition = len(segment.text) + (len(segment.join) if current else 0)
        if current and length + addition > CHUNK_SIZE:
            chunks.append(_build(current, ordinal=len(chunks)))
            current = _overlap_tail(current)
            length = _joined_length(current)
            addition = len(segment.text) + (len(segment.join) if current else 0)
        current.append(segment)
        length += addition

    if current:
        chunks.append(_build(current, ordinal=len(chunks)))
    return _absorb_short_tail(chunks)


def _segment(pages: list[PageText]) -> list[_Segment]:
    """Flatten pages into sentences, each tagged with its page and its joiner.

    Pieces stay in reading order, which is what lets a chunk's page range be
    read off its first and last segment instead of scanning all of them.
    """
    segments: list[_Segment] = []
    for page in pages:
        for paragraph in _PARAGRAPH_BREAK.split(page.text):
            # PDF extraction hard-wraps every visual line. Collapsing those
            # newlines restores sentences the layout had broken up, which both
            # embeds better and makes a quoted chunk readable.
            collapsed = _WHITESPACE.sub(" ", paragraph).strip()
            if not collapsed:
                continue
            pieces = _split(collapsed)
            segments.extend(
                _Segment(
                    page=page.number,
                    text=piece,
                    # Only the first piece of a paragraph starts a new line;
                    # the rest are its own sentences, put back with a space.
                    join=_PARAGRAPH_JOIN if index == 0 else _SENTENCE_JOIN,
                )
                for index, piece in enumerate(pieces)
            )
    return segments


def _split(paragraph: str) -> list[str]:
    """Break a paragraph into pieces no larger than a chunk.

    Sentences first. A sentence that is still too large - a table dumped as one
    line, a URL, a base64 blob - falls back to words and then to a hard cut.
    Each fallback is worse than the last and is only reached when the one
    before it cannot make the piece fit.
    """
    pieces: list[str] = []
    for sentence in _SENTENCE_BREAK.split(paragraph):
        stripped = sentence.strip()
        if not stripped:
            continue
        if len(stripped) <= CHUNK_SIZE:
            pieces.append(stripped)
        else:
            pieces.extend(_split_long_sentence(stripped))
    return pieces


def _split_long_sentence(sentence: str) -> list[str]:
    """Pack a too-long sentence into word-bounded pieces, cutting if it must."""
    pieces: list[str] = []
    current = ""
    for word in sentence.split(" "):
        if not word:
            continue
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > CHUNK_SIZE:
            pieces.append(current)
            current = word
        else:
            current = candidate
        # A single "word" longer than a chunk has no boundary left to respect.
        while len(current) > CHUNK_SIZE:
            pieces.append(current[:CHUNK_SIZE])
            current = current[CHUNK_SIZE:]
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(segments: list[_Segment]) -> list[_Segment]:
    """The trailing segments to repeat at the start of the next chunk.

    Never the whole chunk: ``segments[0]`` is always left behind. Without that
    floor, a chunk made of one segment would be carried forward in its entirety
    and the next chunk would strictly contain it, which at best duplicates text
    in the index and at worst never terminates.
    """
    tail: list[_Segment] = []
    length = 0
    for segment in reversed(segments[1:]):
        addition = len(segment.text) + (len(segment.join) if tail else 0)
        if length + addition > CHUNK_OVERLAP:
            break
        tail.insert(0, segment)
        length += addition
    return tail


def _joined_length(segments: list[_Segment]) -> int:
    if not segments:
        return 0
    return sum(len(segment.text) for segment in segments) + sum(
        len(segment.join) for segment in segments[1:]
    )


def _build(segments: list[_Segment], *, ordinal: int) -> TextChunk:
    text = segments[0].text + "".join(segment.join + segment.text for segment in segments[1:])
    return TextChunk(
        ordinal=ordinal,
        # Segments are in reading order, so the first and last carry the range.
        page_start=segments[0].page,
        page_end=segments[-1].page,
        text=text,
    )


def _absorb_short_tail(chunks: list[TextChunk]) -> list[TextChunk]:
    """Fold a runt final chunk into the one before it, where it fits.

    Only the last chunk can be short, because every other one was emitted for
    being full. Merging rather than dropping keeps the invariant that no
    extracted text is missing from the index, and the merged chunk's page range
    widens to cover both.

    Skipped when the merge would breach CHUNK_SIZE, because a chunk too large
    for the embedder's window would be silently truncated at embedding time -
    losing text, which is the exact outcome this is trying to avoid. A short
    final chunk is the lesser problem.
    """
    if len(chunks) < 2 or len(chunks[-1].text) >= MIN_CHUNK_CHARS:
        return chunks

    previous, last = chunks[-2], chunks[-1]
    merged_text = f"{previous.text}\n{last.text}"
    if len(merged_text) > CHUNK_SIZE:
        return chunks

    merged = TextChunk(
        ordinal=previous.ordinal,
        page_start=previous.page_start,
        page_end=max(previous.page_end, last.page_end),
        text=merged_text,
    )
    return [*chunks[:-2], merged]
