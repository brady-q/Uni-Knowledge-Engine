"""Synthesised PDFs for tests.

Built with PyMuPDF rather than committed as binary fixtures. A checked-in file
is opaque - nobody can tell from the repository what text a test expects to get
back out - whereas these are readable, parameterisable, and cannot drift from
the assertions that use them.

No network and no measurable time, so tests that use this stay in the default
suite rather than behind the ``slow`` marker.
"""

from __future__ import annotations

import pymupdf

# Left margin and first baseline, in points. Well inside any page box.
_MARGIN = 72.0
# Vertical gap between paragraphs. Large enough that MuPDF segments them as
# separate blocks, which is the paragraph signal extraction relies on.
_PARAGRAPH_GAP = 40.0
_LINE_HEIGHT = 14.0


def make_pdf(pages: list[list[str]]) -> bytes:
    """A PDF where ``pages[i]`` is the list of paragraphs on page ``i + 1``.

    Paragraphs are laid out with enough vertical separation to survive as
    distinct blocks, so a test can assert on page ranges and on paragraph
    boundaries rather than on whatever the layout engine happened to do.
    """
    document = pymupdf.open()
    try:
        for paragraphs in pages:
            page = document.new_page()
            baseline = _MARGIN
            for paragraph in paragraphs:
                page.insert_text((_MARGIN, baseline), paragraph)
                baseline += _PARAGRAPH_GAP + _LINE_HEIGHT * paragraph.count("\n")
        return bytes(document.tobytes())
    finally:
        document.close()


def make_empty_pdf(page_count: int = 1) -> bytes:
    """A PDF with pages but no text on any of them - what a scan looks like."""
    document = pymupdf.open()
    try:
        for _ in range(page_count):
            document.new_page()
        return bytes(document.tobytes())
    finally:
        document.close()


def make_encrypted_pdf() -> bytes:
    """A password-protected PDF."""
    document = pymupdf.open()
    try:
        page = document.new_page()
        page.insert_text((_MARGIN, _MARGIN), "secret")
        return bytes(
            document.tobytes(
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner",
                user_pw="user",
            )
        )
    finally:
        document.close()
