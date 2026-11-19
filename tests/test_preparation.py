"""Tests for rag_pipeline.preparation."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pymupdf
import pytest

from rag_pipeline.config import Settings
from rag_pipeline.extraction import PdfExtractionError, compute_file_doc_id
from rag_pipeline.preparation import DocumentChunk, PreparedDocument, prepare_document

HEADER = "ACME Course Notes Header"
METADATA_KEYS = {"doc_id", "filename", "page_number", "chunk_index", "char_start", "char_end"}


def _body(page: int, lines: int = 30) -> str:
    return "\n".join(
        f"Page {page} line {i} explains topic {page * 100 + i} in plain words." for i in range(lines)
    )


def _write_pdf(path: Path, page_texts: list[str | None]) -> Path:
    """Write a PDF with one page per entry; ``None`` creates a blank page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as pdf:
        for text in page_texts:
            page = pdf.new_page()
            if text is not None:
                page.insert_text((72, 60), text, fontsize=9)
        pdf.save(path)
    return path


def _settings(chunk_size: int = 800, chunk_overlap: int = 120) -> Settings:
    return Settings(_env_file=None, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    return _write_pdf(
        tmp_path / "docs" / "lecture.pdf",
        [f"{HEADER}\n{_body(page)}" for page in range(1, 5)],
    )


def test_metadata_on_every_chunk(text_pdf: Path) -> None:
    result = prepare_document(text_pdf, _settings())

    assert isinstance(result, PreparedDocument)
    assert len(result.chunks) > result.page_count  # several chunks per page
    for index, chunk in enumerate(result.chunks):
        assert isinstance(chunk, DocumentChunk)
        assert chunk.doc_id == result.doc_id
        assert chunk.filename == "lecture.pdf"
        assert 1 <= chunk.page_number <= result.page_count
        assert chunk.chunk_index == index
        assert chunk.chunk_id == f"{result.doc_id}:{index}"
        assert chunk.text
        assert chunk.char_end - chunk.char_start == len(chunk.text)


def test_chunk_ids_unique_and_doc_id_is_file_hash(text_pdf: Path) -> None:
    result = prepare_document(text_pdf, _settings())

    ids = [chunk.chunk_id for chunk in result.chunks]
    assert len(set(ids)) == len(ids)
    assert result.doc_id == compute_file_doc_id(text_pdf)


def test_filename_is_basename_only(text_pdf: Path) -> None:
    result = prepare_document(text_pdf, _settings())

    assert result.filename == "lecture.pdf"
    for chunk in result.chunks:
        assert "/" not in chunk.filename and "\\" not in chunk.filename
        assert "/" not in str(chunk.to_metadata()["filename"])
        assert "\\" not in str(chunk.to_metadata()["filename"])


def test_to_metadata_has_exact_keys_and_primitive_values(text_pdf: Path) -> None:
    chunk = prepare_document(text_pdf, _settings()).chunks[0]

    metadata = chunk.to_metadata()

    assert set(metadata) == METADATA_KEYS
    assert all(type(value) in (str, int) for value in metadata.values())
    assert chunk.text not in metadata.values()
    assert metadata == {
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }


def test_custom_chunk_settings_are_applied_and_recorded(text_pdf: Path) -> None:
    default = prepare_document(text_pdf, _settings())
    small = prepare_document(text_pdf, _settings(chunk_size=200, chunk_overlap=0))

    assert (small.chunk_size, small.chunk_overlap) == (200, 0)
    assert (default.chunk_size, default.chunk_overlap) == (800, 120)
    assert len(small.chunks) > len(default.chunks)
    assert all(len(chunk.text) <= 200 for chunk in small.chunks)


def test_default_settings_used_when_none(
    text_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = _settings(chunk_size=300, chunk_overlap=30)
    monkeypatch.setattr("rag_pipeline.preparation.get_settings", lambda: custom)

    result = prepare_document(text_pdf)

    assert (result.chunk_size, result.chunk_overlap) == (300, 30)


def test_same_file_twice_gives_identical_result(text_pdf: Path) -> None:
    assert prepare_document(text_pdf, _settings()) == prepare_document(text_pdf, _settings())


def test_renamed_copy_has_same_ids_and_different_filename(
    text_pdf: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "renamed_copy.pdf"
    shutil.copyfile(text_pdf, copy)

    original = prepare_document(text_pdf, _settings())
    renamed = prepare_document(copy, _settings())

    assert renamed.doc_id == original.doc_id
    assert [c.chunk_id for c in renamed.chunks] == [c.chunk_id for c in original.chunks]
    assert [c.text for c in renamed.chunks] == [c.text for c in original.chunks]
    assert renamed.filename == "renamed_copy.pdf" != original.filename
    assert all(c.filename == "renamed_copy.pdf" for c in renamed.chunks)


def test_no_text_pdf_needs_ocr_with_zero_chunks(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "scanned.pdf", [None, None])

    result = prepare_document(pdf, _settings())

    assert result.needs_ocr is True
    assert result.chunks == ()
    assert result.page_count == 2
    assert result.pages_without_text == (1, 2)


def test_partly_blank_pdf_lists_blank_page_without_chunks(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "partly.pdf", [_body(1, 5), None, _body(3, 5)])

    result = prepare_document(pdf, _settings())

    assert result.needs_ocr is False
    assert result.pages_without_text == (2,)
    assert {c.page_number for c in result.chunks} == {1, 3}
    assert [c.chunk_index for c in result.chunks] == list(range(len(result.chunks)))


def test_invalid_pdf_raises_pdf_extraction_error(tmp_path: Path) -> None:
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"this is not a pdf")

    with pytest.raises(PdfExtractionError):
        prepare_document(fake, _settings())


def test_missing_file_raises_pdf_extraction_error(tmp_path: Path) -> None:
    with pytest.raises(PdfExtractionError):
        prepare_document(tmp_path / "missing.pdf", _settings())


def test_repeated_header_removed_before_chunking(text_pdf: Path) -> None:
    result = prepare_document(text_pdf, _settings())

    assert result.headers_footers_removed == 4
    assert all(HEADER not in chunk.text for chunk in result.chunks)
    assert any("Page 1 line 0 explains" in chunk.text for chunk in result.chunks)


def test_prepare_document_writes_no_files(text_pdf: Path, tmp_path: Path) -> None:
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    prepare_document(text_pdf, _settings())

    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_logs_one_summary_line_without_text_or_filename(
    text_pdf: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG, logger="rag_pipeline.preparation"):
        result = prepare_document(text_pdf, _settings())

    records = [r for r in caplog.records if r.name == "rag_pipeline.preparation"]
    assert len(records) == 1
    message = records[0].getMessage()
    assert records[0].levelno == logging.INFO
    assert result.doc_id[:12] in message
    assert result.doc_id not in message
    assert "lecture.pdf" not in message
    assert "explains" not in message
