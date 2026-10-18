"""PDF text extraction, page by page.

Page numbers are the whole point of this module. Every chunk carries the pages
it came from, and those page numbers are what a citation ultimately resolves
to, so text is never concatenated into one blob and re-split later - the page
each fragment came from is attached at the moment it is read and travels with
it from here on.

Everything here is synchronous and CPU-bound. PyMuPDF releases nothing useful
to an event loop, so callers run ``extract_pdf`` in a worker thread; doing it
inline would stall every other request for the length of a large PDF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from app.errors import IngestionError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PageText:
    """Text of one page, with the 1-based number a reader would see."""

    number: int
    text: str


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    page_count: int
    pages: list[PageText]


def extract_pdf(path: Path) -> ExtractedDocument:
    """Read every page's text.

    Failures are raised as ``IngestionError`` with a message written for the
    uploader, because that message is what ends up in ``documents.error`` and
    on their screen. "This PDF is a scan" is something they can act on;
    "RuntimeError in mupdf" is not.
    """
    # Checked up front rather than caught: PyMuPDF reports a missing file as
    # the same generic error as a corrupt one, and those are different
    # problems. A missing file is a storage bug on our side; a corrupt one is
    # something the uploader can fix.
    if not path.is_file():
        raise IngestionError("the stored file for this document is missing")

    try:
        document = pymupdf.open(path)
    except Exception as exc:
        logger.warning("could not open %s: %s", path, exc)
        raise IngestionError("the file could not be read as a PDF") from exc

    with document:
        # An encrypted PDF opens fine and then yields empty pages, which would
        # otherwise be reported as "no extractable text" and send the uploader
        # looking for an OCR problem they do not have.
        if document.needs_pass:
            raise IngestionError("the PDF is password protected")

        pages = [
            PageText(number=index + 1, text=_page_text(page)) for index, page in enumerate(document)
        ]
        page_count = document.page_count

    if page_count == 0:
        raise IngestionError("the PDF has no pages")
    if not any(page.text for page in pages):
        raise IngestionError(
            "no extractable text found; this looks like a scanned PDF, and OCR is not supported"
        )
    return ExtractedDocument(page_count=page_count, pages=pages)


# Index of the text and of the block type within a "blocks" tuple, which is
# (x0, y0, x1, y1, text, block_number, block_type).
_BLOCK_TEXT = 4
_BLOCK_TYPE = 6
_TEXT_BLOCK = 0


def _page_text(page: Any) -> str:
    """Read one page as paragraphs separated by blank lines.

    ``page`` is ``Any`` because PyMuPDF is deliberately untyped at this
    boundary; see the mypy override in pyproject.toml.

    Blocks rather than ``get_text("text")``, which returns one newline per
    *visual* line and no blank lines at all - under that mode a page is a
    single undifferentiated run, and the chunker has no paragraph boundaries to
    prefer when it splits. Blocks are MuPDF's own paragraph segmentation, and
    joining them with a blank line hands that structure downstream.

    ``sort=True`` orders blocks top-to-bottom and left-to-right instead of in
    the order the content stream happened to draw them, which real PDFs do not
    reliably keep in reading order. Image blocks are skipped: their "text" is
    raw image metadata.
    """
    blocks = page.get_text("blocks", sort=True)
    paragraphs = (
        _clean(block[_BLOCK_TEXT]) for block in blocks if block[_BLOCK_TYPE] == _TEXT_BLOCK
    )
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _clean(text: str) -> str:
    """Normalise a block's text without destroying its line structure.

    Two things are worth removing and one is worth keeping. Removed: the NUL
    bytes some producers emit, which Postgres rejects outright in a text
    column, and trailing whitespace on every line, which is noise. Kept: the
    line breaks themselves - the chunker collapses them when it reflows a
    paragraph, and doing it here would lose the boundary between blocks.
    """
    without_nulls = text.replace("\x00", "")
    return "\n".join(line.rstrip() for line in without_nulls.splitlines()).strip()
