"""Tests for rag_pipeline.inbox_ingestion.ingest_one_file (tmp folders, tmp stores, stub model)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag_pipeline import inbox_ingestion
from rag_pipeline.extraction import compute_file_doc_id
from rag_pipeline.inbox_ingestion import InboxRunReport, ingest_inbox, ingest_one_file
from rag_pipeline.ingestion import (
    SKIP_ALREADY_INGESTED,
    SKIP_IN_PROGRESS,
    IngestContext,
    IngestOutcome,
)
from rag_pipeline.status_db import DocumentStatus
from rag_pipeline.vector_store import count_document_chunks
from tests.test_inbox_ingestion import (  # noqa: F401 - fixtures are used by name
    MAX_ATTEMPTS,
    body,
    clear_chroma_system_cache,
    context,
    dirs,
    names,
    write_pdf,
)


def one(context: IngestContext, dirs: dict[str, Path], name: str) -> InboxRunReport:
    return ingest_one_file(
        context,
        dirs["inbox"] / name,
        dirs["processed"],
        dirs["failed"],
        dirs["needs_ocr"],
        inbox_dir=dirs["inbox"],
    )


def test_new_text_pdf_is_ingested_and_moved(context: IngestContext, dirs: dict[str, Path]) -> None:
    path = write_pdf(dirs["inbox"] / "lecture.pdf", [body(1), body(2)])
    doc_id = compute_file_doc_id(path)

    report = one(context, dirs, "lecture.pdf")

    assert len(report.ingest_results) == 1
    result = report.ingest_results[0]
    assert (result.outcome, result.filename, result.attempts) == (
        IngestOutcome.DONE, "lecture.pdf", 1
    )
    assert [m.destination for m in report.moves] == [dirs["processed"] / "lecture.pdf"]
    assert report.left_in_inbox == () and report.move_failures == () and report.scan_errors == ()
    assert names(dirs["inbox"]) == [] and names(dirs["processed"]) == ["lecture.pdf"]
    record = context.store.get_document(doc_id)
    assert record is not None and record.status is DocumentStatus.DONE
    assert count_document_chunks(context.collection, doc_id) == result.chunk_count > 0


def test_only_the_named_file_is_touched(context: IngestContext, dirs: dict[str, Path]) -> None:
    write_pdf(dirs["inbox"] / "a.pdf", [body(1)])
    other = write_pdf(dirs["inbox"] / "b.pdf", [body(2)])

    one(context, dirs, "a.pdf")

    assert names(dirs["inbox"]) == ["b.pdf"]
    assert context.store.get_document(compute_file_doc_id(other)) is None


def test_scanned_pdf_goes_to_needs_ocr(context: IngestContext, dirs: dict[str, Path]) -> None:
    write_pdf(dirs["inbox"] / "scan.pdf", [None, None])

    report = one(context, dirs, "scan.pdf")

    assert report.ingest_results[0].outcome is IngestOutcome.NEEDS_OCR
    assert names(dirs["needs_ocr"]) == ["scan.pdf"]


def test_invalid_pdf_stays_until_attempts_are_exhausted(
    context: IngestContext, dirs: dict[str, Path]
) -> None:
    (dirs["inbox"] / "broken.pdf").write_bytes(b"not a pdf")

    for attempt in range(1, MAX_ATTEMPTS):
        report = one(context, dirs, "broken.pdf")
        result = report.ingest_results[0]
        assert (result.outcome, result.attempts) == (IngestOutcome.FAILED, attempt)
        assert [entry.reason for entry in report.left_in_inbox] == [
            f"failed, attempts {attempt} of {MAX_ATTEMPTS}"
        ]
        assert names(dirs["inbox"]) == ["broken.pdf"]

    last = one(context, dirs, "broken.pdf")

    assert last.ingest_results[0].attempts == MAX_ATTEMPTS
    assert names(dirs["failed"]) == ["broken.pdf"] and names(dirs["inbox"]) == []


def test_copy_of_a_done_document_is_skipped_and_moved(
    context: IngestContext, dirs: dict[str, Path]
) -> None:
    original = write_pdf(dirs["inbox"] / "a.pdf", [body(1), body(2)])
    doc_id = compute_file_doc_id(original)
    copy = dirs["inbox"] / "copy of a.pdf"
    shutil.copyfile(original, copy)
    first = one(context, dirs, "a.pdf")
    chunks = count_document_chunks(context.collection, doc_id)

    report = one(context, dirs, "copy of a.pdf")

    result = report.ingest_results[0]
    assert (result.outcome, result.skip_reason) == (IngestOutcome.SKIPPED, SKIP_ALREADY_INGESTED)
    assert result.chunk_count == first.ingest_results[0].chunk_count == chunks
    assert names(dirs["processed"]) == ["a.pdf", "copy of a.pdf"]
    record = context.store.get_document(doc_id)
    assert record is not None and record.filename == "a.pdf" and record.attempts == 1


def test_processing_document_is_skipped_and_stays(
    context: IngestContext, dirs: dict[str, Path]
) -> None:
    path = write_pdf(dirs["inbox"] / "busy.pdf", [body(1)])
    doc_id = compute_file_doc_id(path)
    context.store.register_document(doc_id, "busy.pdf")
    context.store.mark_processing(doc_id)

    report = one(context, dirs, "busy.pdf")

    assert report.ingest_results[0].skip_reason == SKIP_IN_PROGRESS
    assert [entry.reason for entry in report.left_in_inbox] == ["processing"]
    assert names(dirs["inbox"]) == ["busy.pdf"]


def test_missing_file_is_a_failed_result_not_an_exception(
    context: IngestContext, dirs: dict[str, Path]
) -> None:
    report = one(context, dirs, "gone.pdf")

    result = report.ingest_results[0]
    assert (result.outcome, result.doc_id, result.attempts) == (IngestOutcome.FAILED, None, None)
    assert result.error_message is not None and result.error_message.startswith("FileNotFoundError")
    assert report.moves == () and report.left_in_inbox == () and report.move_failures == ()
    assert context.store.list_documents() == []


def test_move_failure_is_reported_and_the_file_stays(
    context: IngestContext, dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pdf(dirs["inbox"] / "open.pdf", [body(1)])

    def locked(source: Path, dest_dir: Path) -> Path:
        raise PermissionError("file is open in another program")

    monkeypatch.setattr(inbox_ingestion, "safe_move", locked)

    report = one(context, dirs, "open.pdf")

    assert report.ingest_results[0].outcome is IngestOutcome.DONE
    assert len(report.move_failures) == 1
    failure = report.move_failures[0]
    assert failure.destination_dir == dirs["processed"]
    assert failure.error == "PermissionError: file is open in another program"
    assert report.moves == () and names(dirs["inbox"]) == ["open.pdf"]

    monkeypatch.undo()
    healed = ingest_inbox(
        context, dirs["inbox"], dirs["processed"], dirs["failed"], dirs["needs_ocr"]
    )
    assert healed.ingest_results == () and names(dirs["processed"]) == ["open.pdf"]


def test_path_outside_the_inbox_is_rejected(context: IngestContext, dirs: dict[str, Path]) -> None:
    outside = write_pdf(dirs["inbox"].parent / "elsewhere.pdf", [body(1)])

    with pytest.raises(ValueError, match="not directly in the inbox"):
        ingest_one_file(
            context,
            outside,
            dirs["processed"],
            dirs["failed"],
            dirs["needs_ocr"],
            inbox_dir=dirs["inbox"],
        )

    assert context.store.list_documents() == []


def test_destination_equal_to_inbox_is_rejected(
    context: IngestContext, dirs: dict[str, Path]
) -> None:
    write_pdf(dirs["inbox"] / "a.pdf", [body(1)])

    with pytest.raises(ValueError, match="must not be the inbox"):
        ingest_one_file(
            context,
            dirs["inbox"] / "a.pdf",
            dirs["inbox"],
            dirs["failed"],
            dirs["needs_ocr"],
            inbox_dir=dirs["inbox"],
        )
