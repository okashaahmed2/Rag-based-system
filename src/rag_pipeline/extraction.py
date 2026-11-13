"""Page-by-page PDF text extraction with PyMuPDF.

The public entry point is :func:`extract_pdf`. It reads the file once, derives
the ``doc_id`` (SHA-256 hex digest of the raw bytes) and returns the raw text
of every page together with a flag telling whether the page had any
extractable text. A document needs OCR only when none of its pages has
extractable text; individual blank pages are reported but do not make the
whole document ``needs_ocr``.

Text is returned unmodified; cleaning and chunking happen in later steps.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be read or its text cannot be extracted."""


@dataclass(frozen=True)
class PageText:
    """Raw text of a single page.

    Attributes:
        page_number: 1-based page number, as shown to users in citations.
        text: Raw text extracted by PyMuPDF (may be empty).
        has_text: ``False`` if the page contains no non-whitespace text.
    """

    page_number: int
    text: str
    has_text: bool


@dataclass(frozen=True)
class ExtractedDocument:
    """Result of extracting one PDF."""

    doc_id: str
    filename: str
    pages: tuple[PageText, ...]

    @property
    def page_count(self) -> int:
        """Number of pages in the document."""
        return len(self.pages)

    @property
    def pages_without_text(self) -> tuple[int, ...]:
        """1-based numbers of pages that have no extractable text."""
        return tuple(page.page_number for page in self.pages if not page.has_text)

    @property
    def pages_with_text_count(self) -> int:
        """Number of pages that have extractable text."""
        return sum(1 for page in self.pages if page.has_text)

    @property
    def needs_ocr(self) -> bool:
        """``True`` only if no page has extractable text."""
        return self.pages_with_text_count == 0


def compute_doc_id(data: bytes) -> str:
    """Return the document id: the SHA-256 hex digest of the raw PDF bytes."""
    return hashlib.sha256(data).hexdigest()


def extract_pdf(path: Path) -> ExtractedDocument:
    """Extract the raw text of every page of the PDF at ``path``.

    Raises:
        PdfExtractionError: if the file cannot be read, is empty, is not a
            valid PDF, is password-protected or has no pages.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        logger.error("Cannot read PDF file %s: %s", path, exc)
        raise PdfExtractionError(f"Cannot read file {path}: {exc}") from exc

    doc_id = compute_doc_id(data)
    pages = _extract_pages(data, path)

    document = ExtractedDocument(doc_id=doc_id, filename=path.name, pages=pages)
    if document.needs_ocr:
        logger.warning(
            "PDF %s (doc_id=%s) has no extractable text on any of its %d page(s); "
            "needs OCR",
            document.filename,
            doc_id,
            document.page_count,
        )
    elif document.pages_without_text:
        logger.warning(
            "PDF %s (doc_id=%s) has %d of %d page(s) without text: %s",
            document.filename,
            doc_id,
            len(document.pages_without_text),
            document.page_count,
            list(document.pages_without_text),
        )
    logger.info(
        "Extracted %d page(s) from %s (doc_id=%s, pages_with_text=%d, needs_ocr=%s)",
        document.page_count,
        document.filename,
        doc_id,
        document.pages_with_text_count,
        document.needs_ocr,
    )
    return document


def _extract_pages(data: bytes, path: Path) -> tuple[PageText, ...]:
    """Open the PDF from ``data`` and return the text of each page."""
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except pymupdf.FileDataError as exc:  # also covers EmptyFileError
        logger.error("Invalid or empty PDF %s: %s", path, exc)
        raise PdfExtractionError(f"Invalid or empty PDF {path}: {exc}") from exc

    with pdf:
        if pdf.needs_pass:
            logger.error("PDF %s is password-protected", path)
            raise PdfExtractionError(f"PDF {path} is password-protected")
        if pdf.page_count == 0:
            logger.error("PDF %s has no pages", path)
            raise PdfExtractionError(f"PDF {path} has no pages")

        pages: list[PageText] = []
        for index, page in enumerate(pdf):
            try:
                text = page.get_text("text")
            except RuntimeError as exc:  # MuPDF errors surface as RuntimeError
                logger.error("Failed to extract page %d of %s: %s", index + 1, path, exc)
                raise PdfExtractionError(
                    f"Failed to extract page {index + 1} of {path}: {exc}"
                ) from exc
            pages.append(
                PageText(page_number=index + 1, text=text, has_text=bool(text.strip()))
            )
    return tuple(pages)
