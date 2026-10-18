"""PDF extraction against synthesised PDFs.

No database and no network, so these run in the default suite. The PDFs are
built in-process by ``tests.pdfs``, which means the expected text is visible in
the test rather than buried in a committed binary.

What is under test is mostly the page metadata. The text itself is MuPDF's job;
attaching the right page number to it is ours, and it is what every citation
eventually resolves through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.errors import IngestionError
from app.services.extraction import extract_pdf
from tests.pdfs import make_empty_pdf, make_encrypted_pdf, make_pdf


def write(tmp_path: Path, data: bytes, name: str = "doc.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_pages_are_numbered_from_one_in_reading_order(tmp_path: Path) -> None:
    path = write(tmp_path, make_pdf([["First page."], ["Second page."], ["Third page."]]))

    extracted = extract_pdf(path)

    assert extracted.page_count == 3
    assert [page.number for page in extracted.pages] == [1, 2, 3]
    assert "First page." in extracted.pages[0].text
    assert "Third page." in extracted.pages[2].text


def test_paragraphs_are_separated_by_a_blank_line(tmp_path: Path) -> None:
    """The only structural signal the chunker gets, so it has to survive.

    ``get_text("text")`` would return these as consecutive lines and the
    boundary would be gone; block extraction is what preserves it.
    """
    path = write(tmp_path, make_pdf([["Alpha paragraph.", "Beta paragraph."]]))

    text = extract_pdf(path).pages[0].text

    assert "Alpha paragraph." in text
    assert "Beta paragraph." in text
    assert "\n\n" in text, "paragraph boundary was lost"


def test_a_page_with_no_text_is_kept_as_an_empty_page(tmp_path: Path) -> None:
    """Dropping it would shift every later page number by one."""
    path = write(tmp_path, make_pdf([["Has text."], [], ["Also has text."]]))

    extracted = extract_pdf(path)

    assert [page.number for page in extracted.pages] == [1, 2, 3]
    assert extracted.pages[1].text == ""
    assert "Also has text." in extracted.pages[2].text


def test_a_scanned_pdf_is_rejected_with_an_actionable_message(tmp_path: Path) -> None:
    path = write(tmp_path, make_empty_pdf(3))

    with pytest.raises(IngestionError) as caught:
        extract_pdf(path)

    assert "scanned" in caught.value.detail
    assert "OCR" in caught.value.detail


def test_an_encrypted_pdf_is_named_as_such(tmp_path: Path) -> None:
    """Otherwise it presents as "no text" and sends the uploader after OCR."""
    path = write(tmp_path, make_encrypted_pdf())

    with pytest.raises(IngestionError) as caught:
        extract_pdf(path)

    assert "password" in caught.value.detail


def test_bytes_that_are_not_a_pdf_are_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, b"this is plainly not a pdf")

    with pytest.raises(IngestionError) as caught:
        extract_pdf(path)

    assert "PDF" in caught.value.detail


def test_a_missing_file_is_reported_as_a_storage_problem(tmp_path: Path) -> None:
    """Distinct from a bad upload: the row exists and its bytes do not."""
    with pytest.raises(IngestionError) as caught:
        extract_pdf(tmp_path / "never-written.pdf")

    assert "missing" in caught.value.detail


def test_nul_bytes_are_stripped(tmp_path: Path) -> None:
    """Postgres rejects NUL in a text column outright, so it cannot reach a chunk."""
    path = write(tmp_path, make_pdf([["Before\x00after."]]))

    text = extract_pdf(path).pages[0].text

    assert "\x00" not in text
