"""Prepare one PDF for indexing: extract, clean, chunk and attach metadata.

:func:`prepare_document` chains the existing steps
(:func:`rag_pipeline.extraction.extract_pdf` ->
:func:`rag_pipeline.cleaning.clean_pages` ->
:func:`rag_pipeline.cleaning.remove_repeated_headers_footers` ->
:func:`rag_pipeline.chunking.chunk_pages`) and wraps every chunk in a
:class:`DocumentChunk` carrying the document's ``doc_id`` and ``filename``.

It does no file moves, no writes and keeps no global state. Embedding and
storage happen in later phases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from rag_pipeline.chunking import TextChunk, chunk_pages
from rag_pipeline.cleaning import clean_pages, remove_repeated_headers_footers
from rag_pipeline.config import Settings, get_settings
from rag_pipeline.extraction import extract_pdf

logger = logging.getLogger(__name__)

#: Number of leading ``doc_id`` characters shown in log lines.
LOG_DOC_ID_CHARS = 12


@dataclass(frozen=True)
class DocumentChunk:
    """A chunk of one document's page, with the metadata needed for citations.

    Attributes:
        chunk_id: ``f"{doc_id}:{chunk_index}"``; unique within and across
            documents, and stable for the same file bytes and settings.
        doc_id: SHA-256 hex digest of the PDF bytes.
        filename: Base name of the PDF file (never a path).
        page_number: 1-based page the chunk comes from.
        chunk_index: Sequential index 0..N-1 within the document.
        text: Chunk text, an exact slice of the cleaned page text.
        char_start: Start offset into the cleaned page text (inclusive).
        char_end: End offset into the cleaned page text (exclusive).
    """

    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int

    def to_metadata(self) -> dict[str, str | int]:
        """Return flat, primitive-only metadata for a vector store (no text)."""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass(frozen=True)
class PreparedDocument:
    """Result of :func:`prepare_document` for one PDF.

    Attributes:
        doc_id: SHA-256 hex digest of the PDF bytes.
        filename: Base name of the PDF file.
        page_count: Number of pages in the PDF.
        needs_ocr: ``True`` if no page has extractable text (then ``chunks``
            is empty).
        pages_without_text: Pages with no extractable text, as reported by
            extraction. A page whose text is removed entirely by cleaning or
            header/footer removal has no chunks but is not listed here.
        chunk_size: Chunk size (characters) used for this document.
        chunk_overlap: Chunk overlap (characters) used for this document.
        chunks: Chunks in ``chunk_index`` order.
        headers_footers_removed: Number of header/footer lines removed.
    """

    doc_id: str
    filename: str
    page_count: int
    needs_ocr: bool
    pages_without_text: tuple[int, ...]
    chunk_size: int
    chunk_overlap: int
    chunks: tuple[DocumentChunk, ...]
    headers_footers_removed: int


def _to_document_chunk(chunk: TextChunk, doc_id: str, filename: str) -> DocumentChunk:
    """Attach document metadata to a page chunk."""
    return DocumentChunk(
        chunk_id=f"{doc_id}:{chunk.chunk_index}",
        doc_id=doc_id,
        filename=filename,
        page_number=chunk.page_number,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
    )


def prepare_document(pdf_path: Path, settings: Settings | None = None) -> PreparedDocument:
    """Extract, clean and chunk one PDF and attach chunk metadata.

    Uses ``settings.chunk_size`` and ``settings.chunk_overlap``; when
    ``settings`` is ``None``, :func:`rag_pipeline.config.get_settings` is used.

    A document that needs OCR is returned normally with ``needs_ocr=True``
    and no chunks. Deterministic for the same file bytes, name and settings.
    No files are moved or written. Logs one INFO summary line with a
    shortened ``doc_id``; neither text nor filename is logged here.

    Raises:
        PdfExtractionError: unchanged from :func:`extract_pdf` (unreadable,
            invalid, password-protected or empty PDF).
    """
    active = settings if settings is not None else get_settings()
    extracted = extract_pdf(pdf_path)
    header_footer = remove_repeated_headers_footers(clean_pages(extracted.pages))
    page_chunks = chunk_pages(
        header_footer.pages, active.chunk_size, active.chunk_overlap
    )
    prepared = PreparedDocument(
        doc_id=extracted.doc_id,
        filename=extracted.filename,
        page_count=extracted.page_count,
        needs_ocr=extracted.needs_ocr,
        pages_without_text=extracted.pages_without_text,
        chunk_size=active.chunk_size,
        chunk_overlap=active.chunk_overlap,
        chunks=tuple(
            _to_document_chunk(chunk, extracted.doc_id, extracted.filename)
            for chunk in page_chunks
        ),
        headers_footers_removed=header_footer.lines_removed,
    )
    logger.info(
        "Prepared document %s: %d page(s), %d chunk(s), needs_ocr=%s, "
        "%d header/footer line(s) removed",
        prepared.doc_id[:LOG_DOC_ID_CHARS],
        prepared.page_count,
        len(prepared.chunks),
        prepared.needs_ocr,
        prepared.headers_footers_removed,
    )
    return prepared
