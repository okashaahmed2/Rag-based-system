"""Tests for rag_pipeline.extraction."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pymupdf
import pytest

from rag_pipeline.extraction import (
    ExtractedDocument,
    PdfExtractionError,
    compute_doc_id,
    extract_pdf,
)


def _write_pdf(path: Path, page_texts: list[str | None], **save_kwargs: object) -> Path:
    """Write a PDF with one page per entry; ``None`` creates a blank page."""
    with pymupdf.open() as pdf:
        for text in page_texts:
            page = pdf.new_page()
            if text is not None:
                page.insert_text((72, 72), text)
        pdf.save(path, **save_kwargs)
    return path


def test_compute_doc_id_is_sha256_hex_of_bytes() -> None:
    data = b"%PDF-1.7 example bytes"

    doc_id = compute_doc_id(data)

    assert doc_id == hashlib.sha256(data).hexdigest()
    assert len(doc_id) == 64


def test_extract_pdf_returns_text_per_page_with_1_based_numbers(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "report.pdf", ["First page", "Second page"])

    document = extract_pdf(pdf_path)

    assert isinstance(document, ExtractedDocument)
    assert document.filename == "report.pdf"
    assert document.page_count == 2
    assert [page.page_number for page in document.pages] == [1, 2]
    assert "First page" in document.pages[0].text
    assert "Second page" in document.pages[1].text
    assert all(page.has_text for page in document.pages)


def test_doc_id_matches_sha256_of_file_bytes(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "a.pdf", ["Hello"])

    document = extract_pdf(pdf_path)

    assert document.doc_id == hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def test_doc_id_ignores_filename(tmp_path: Path) -> None:
    original = _write_pdf(tmp_path / "original.pdf", ["Same content"])
    copy = tmp_path / "renamed.pdf"
    copy.write_bytes(original.read_bytes())

    assert extract_pdf(original).doc_id == extract_pdf(copy).doc_id


def test_doc_id_differs_for_different_content(tmp_path: Path) -> None:
    first = _write_pdf(tmp_path / "first.pdf", ["Content A"])
    second = _write_pdf(tmp_path / "second.pdf", ["Content B"])

    assert extract_pdf(first).doc_id != extract_pdf(second).doc_id


def test_all_pages_with_text_does_not_need_ocr(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "text.pdf", ["One", "Two"])

    document = extract_pdf(pdf_path)

    assert document.needs_ocr is False
    assert document.pages_without_text == ()
    assert document.pages_with_text_count == 2


def test_some_blank_pages_are_listed_but_document_does_not_need_ocr(
    tmp_path: Path,
) -> None:
    pdf_path = _write_pdf(
        tmp_path / "mixed.pdf", ["Text", None, "More text", None]
    )

    document = extract_pdf(pdf_path)

    assert document.pages[1].has_text is False
    assert document.pages[1].text.strip() == ""
    assert document.pages_without_text == (2, 4)
    assert document.pages_with_text_count == 2
    assert document.needs_ocr is False


def test_all_pages_blank_needs_ocr(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "scan.pdf", [None, None])

    document = extract_pdf(pdf_path)

    assert document.pages_without_text == (1, 2)
    assert document.pages_with_text_count == 0
    assert document.needs_ocr is True


def test_whitespace_only_page_counts_as_without_text(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path / "spaces.pdf", ["   "])

    document = extract_pdf(pdf_path)

    assert document.pages[0].has_text is False
    assert document.needs_ocr is True


def test_partial_blank_pages_are_logged_as_warning_with_page_numbers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf_path = _write_pdf(tmp_path / "mixed.pdf", ["Text", None, None])

    with caplog.at_level(logging.WARNING, logger="rag_pipeline.extraction"):
        extract_pdf(pdf_path)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "without text" in warnings[0].getMessage()
    assert "[2, 3]" in warnings[0].getMessage()


def test_no_warning_when_every_page_has_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf_path = _write_pdf(tmp_path / "text.pdf", ["One", "Two"])

    with caplog.at_level(logging.WARNING, logger="rag_pipeline.extraction"):
        extract_pdf(pdf_path)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_fully_blank_document_is_logged_as_needing_ocr(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf_path = _write_pdf(tmp_path / "scan.pdf", [None])

    with caplog.at_level(logging.WARNING, logger="rag_pipeline.extraction"):
        extract_pdf(pdf_path)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "needs OCR" in warnings[0].getMessage()


def test_missing_file_raises_extraction_error(tmp_path: Path) -> None:
    with pytest.raises(PdfExtractionError, match="Cannot read file"):
        extract_pdf(tmp_path / "missing.pdf")


def test_empty_file_raises_extraction_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")

    with pytest.raises(PdfExtractionError, match="Invalid or empty PDF"):
        extract_pdf(empty)


def test_non_pdf_file_raises_extraction_error(tmp_path: Path) -> None:
    not_pdf = tmp_path / "notes.pdf"
    not_pdf.write_bytes(b"this is plain text, not a PDF")

    with pytest.raises(PdfExtractionError, match="Invalid or empty PDF"):
        extract_pdf(not_pdf)


def test_password_protected_pdf_raises_extraction_error(tmp_path: Path) -> None:
    pdf_path = _write_pdf(
        tmp_path / "locked.pdf",
        ["Secret"],
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )

    with pytest.raises(PdfExtractionError, match="password-protected"):
        extract_pdf(pdf_path)
