"""Tests for rag_pipeline.status_db (real SQLite file in tmp_path, fake clock)."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rag_pipeline.extraction import compute_file_doc_id
from rag_pipeline.inbox import ScanStatus, scan_inbox
from rag_pipeline.status_db import (
    INTERRUPTED_MESSAGE,
    MAX_ERROR_MESSAGE_CHARS,
    SCHEMA_VERSION,
    DocumentRecord,
    DocumentStatus,
    StatusStore,
    StatusStoreError,
)

START = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)
DOC = "d" * 64
OTHER = "e" * 64


class FakeClock:
    """Deterministic clock: every call returns a time one second later."""

    def __init__(self, start: datetime = START) -> None:
        self.current = start

    def __call__(self) -> datetime:
        value = self.current
        self.current += timedelta(seconds=1)
        return value


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "db" / "status.db"


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(db_path: Path, clock: FakeClock) -> StatusStore:
    status_store = StatusStore(db_path, clock=clock)
    status_store.initialize()
    return status_store


def raw(path: Path, sql: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    """Run one statement on a separate, always-closed connection."""
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            rows = conn.execute(sql, params).fetchall()
    return rows


# Ways to bring a fresh document into each status.
def _to_pending(s: StatusStore, d: str) -> None:
    s.register_document(d, "doc.pdf")


def _to_processing(s: StatusStore, d: str) -> None:
    _to_pending(s, d)
    s.mark_processing(d)


def _to_done(s: StatusStore, d: str) -> None:
    _to_processing(s, d)
    s.mark_done(d, 7)


def _to_needs_ocr(s: StatusStore, d: str) -> None:
    _to_processing(s, d)
    s.mark_needs_ocr(d)


def _to_failed(s: StatusStore, d: str) -> None:
    _to_processing(s, d)
    s.mark_failed(d, "boom")


SETUP: dict[DocumentStatus, Callable[[StatusStore, str], None]] = {
    DocumentStatus.PENDING: _to_pending,
    DocumentStatus.PROCESSING: _to_processing,
    DocumentStatus.DONE: _to_done,
    DocumentStatus.NEEDS_OCR: _to_needs_ocr,
    DocumentStatus.FAILED: _to_failed,
}
ACTIONS: dict[str, Callable[[StatusStore, str], DocumentRecord]] = {
    "processing": lambda s, d: s.mark_processing(d),
    "done": lambda s, d: s.mark_done(d, 3),
    "needs_ocr": lambda s, d: s.mark_needs_ocr(d),
    "failed": lambda s, d: s.mark_failed(d, "again"),
}
ALLOWED_FROM: dict[str, set[DocumentStatus]] = {
    "processing": {DocumentStatus.PENDING, DocumentStatus.FAILED},
    "done": {DocumentStatus.PROCESSING},
    "needs_ocr": {DocumentStatus.PROCESSING},
    "failed": {DocumentStatus.PROCESSING},
}
INVALID = [
    (state, action)
    for action, allowed in ALLOWED_FROM.items()
    for state in DocumentStatus
    if state not in allowed
]


# --- schema --------------------------------------------------------------------


def test_initialize_creates_folder_table_index_and_version(
    store: StatusStore, db_path: Path
) -> None:
    assert db_path.is_file()
    assert raw(db_path, "PRAGMA user_version") == [(SCHEMA_VERSION,)] == [(2,)]
    names = {row[0] for row in raw(db_path, "SELECT name FROM sqlite_master")}
    assert {
        "documents", "idx_documents_status", "jobs", "idx_jobs_status", "idx_jobs_one_active"
    } <= names


def test_initialize_twice_is_harmless(store: StatusStore, db_path: Path) -> None:
    store.register_document(DOC, "a.pdf")

    store.initialize()

    assert raw(db_path, "PRAGMA user_version") == [(2,)]
    assert store.get_document(DOC) is not None


def test_newer_schema_version_raises(store: StatusStore, db_path: Path) -> None:
    raw(db_path, "PRAGMA user_version = 3")

    with pytest.raises(StatusStoreError, match="schema version 3"):
        StatusStore(db_path).initialize()


def test_constructor_touches_nothing(db_path: Path) -> None:
    StatusStore(db_path)

    assert not db_path.parent.exists()


def test_use_before_initialize_raises_and_creates_no_file(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True)

    with pytest.raises(StatusStoreError, match="not initialized"):
        StatusStore(db_path).get_document(DOC)

    assert not db_path.exists()


def test_schema_check_matches_enum(store: StatusStore, db_path: Path) -> None:
    for status in DocumentStatus:
        raw(
            db_path,
            "INSERT INTO documents (doc_id, filename, status, created_at, updated_at) "
            "VALUES (?, 'f.pdf', ?, 'x', 'x')",
            (f"id-{status.value}", status.value),
        )
    assert len(raw(db_path, "SELECT doc_id FROM documents")) == len(DocumentStatus)


@pytest.mark.parametrize(
    ("column_values", "label"),
    [
        (("bad", "bogus", 0, None), "status"),
        (("neg", "pending", -1, None), "attempts"),
        (("chunks", "done", 1, -5), "chunk_count"),
    ],
)
def test_check_constraints_reject_bad_rows(
    store: StatusStore, db_path: Path, column_values: tuple[object, ...], label: str
) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        raw(
            db_path,
            "INSERT INTO documents (doc_id, filename, status, attempts, chunk_count, "
            "created_at, updated_at) VALUES (?, 'f.pdf', ?, ?, ?, 'x', 'x')",
            column_values,
        )
    assert raw(db_path, "SELECT COUNT(*) FROM documents") == [(0,)]


# --- lifecycle -----------------------------------------------------------------


def test_register_creates_pending(store: StatusStore) -> None:
    record = store.register_document(DOC, "a.pdf")

    assert record == DocumentRecord(
        doc_id=DOC,
        filename="a.pdf",
        status=DocumentStatus.PENDING,
        error_message=None,
        attempts=0,
        chunk_count=None,
        created_at=START,
        updated_at=START,
    )
    assert record.created_at.tzinfo is not None
    assert record.created_at.utcoffset() == timedelta(0)


def test_happy_path_pending_processing_done(store: StatusStore) -> None:
    registered = store.register_document(DOC, "a.pdf")
    processing = store.mark_processing(DOC)
    done = store.mark_done(DOC, 17)

    assert (processing.status, processing.attempts) == (DocumentStatus.PROCESSING, 1)
    assert (done.status, done.chunk_count, done.attempts, done.error_message) == (
        DocumentStatus.DONE,
        17,
        1,
        None,
    )
    assert registered.updated_at < processing.updated_at < done.updated_at
    assert done.created_at == registered.created_at == START
    assert store.get_document(DOC) == done


def test_needs_ocr_path(store: StatusStore) -> None:
    _to_processing(store, DOC)

    record = store.mark_needs_ocr(DOC)

    assert (record.status, record.chunk_count, record.attempts) == (
        DocumentStatus.NEEDS_OCR,
        0,
        1,
    )


def test_failed_path(store: StatusStore) -> None:
    _to_processing(store, DOC)

    record = store.mark_failed(DOC, "  PdfExtractionError: broken xref  ")

    assert record.status is DocumentStatus.FAILED
    assert record.error_message == "PdfExtractionError: broken xref"
    assert (record.attempts, record.chunk_count) == (1, None)


def test_retry_clears_error_and_counts_attempts(store: StatusStore) -> None:
    _to_failed(store, DOC)

    record = store.mark_processing(DOC)

    assert (record.status, record.error_message, record.attempts) == (
        DocumentStatus.PROCESSING,
        None,
        2,
    )
    assert store.mark_done(DOC, 4).attempts == 2


@pytest.mark.parametrize(
    ("state", "action"),
    INVALID,
    ids=[f"{state.value}->{action}" for state, action in INVALID],
)
def test_invalid_transition_raises_and_changes_nothing(
    store: StatusStore, state: DocumentStatus, action: str
) -> None:
    SETUP[state](store, DOC)
    before = store.get_document(DOC)

    with pytest.raises(StatusStoreError, match=f"status {state.value}"):
        ACTIONS[action](store, DOC)

    assert store.get_document(DOC) == before


@pytest.mark.parametrize("action", sorted(ACTIONS))
def test_transition_of_unknown_doc_raises(store: StatusStore, action: str) -> None:
    with pytest.raises(StatusStoreError, match="unknown doc_id"):
        ACTIONS[action](store, DOC)

    assert store.list_documents() == []


def test_register_existing_keeps_status_attempts_and_filename(store: StatusStore) -> None:
    _to_failed(store, DOC)
    before = store.get_document(DOC)

    again = store.register_document(DOC, "renamed_copy.pdf")

    assert again == before
    assert (again.status, again.attempts, again.filename) == (
        DocumentStatus.FAILED,
        1,
        "doc.pdf",
    )


@pytest.mark.parametrize(("doc_id", "filename"), [("", "a.pdf"), (DOC, ""), (DOC, "   ")])
def test_register_rejects_empty_values(store: StatusStore, doc_id: str, filename: str) -> None:
    with pytest.raises(StatusStoreError, match="must not be empty"):
        store.register_document(doc_id, filename)

    assert store.list_documents() == []


@pytest.mark.parametrize("bad", [-1, True, 2.5])
def test_mark_done_rejects_bad_chunk_count(store: StatusStore, bad: object) -> None:
    _to_processing(store, DOC)
    before = store.get_document(DOC)

    with pytest.raises(StatusStoreError, match="chunk_count"):
        store.mark_done(DOC, bad)  # type: ignore[arg-type]

    assert store.get_document(DOC) == before


def test_long_error_message_is_trimmed(store: StatusStore) -> None:
    _to_processing(store, DOC)

    record = store.mark_failed(DOC, "x" * (MAX_ERROR_MESSAGE_CHARS + 500))

    assert record.error_message == "x" * MAX_ERROR_MESSAGE_CHARS


@pytest.mark.parametrize("message", ["", "   \n"])
def test_empty_error_message_rejected(store: StatusStore, message: str) -> None:
    _to_processing(store, DOC)

    with pytest.raises(StatusStoreError, match="error_message must not be empty"):
        store.mark_failed(DOC, message)

    assert store.get_document(DOC).status is DocumentStatus.PROCESSING  # type: ignore[union-attr]


def test_naive_clock_rejected(db_path: Path) -> None:
    status_store = StatusStore(db_path, clock=lambda: datetime(2026, 1, 1))

    with pytest.raises(StatusStoreError, match="naive"):
        status_store.initialize()
        status_store.register_document(DOC, "a.pdf")


def test_transitions_are_logged_with_short_doc_id(
    store: StatusStore, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="rag_pipeline.status_db"):
        _to_done(store, DOC)

    assert f"{DOC[:12]} (doc.pdf): new -> pending" in caplog.text
    assert "pending -> processing" in caplog.text
    assert "processing -> done" in caplog.text
    assert DOC not in caplog.text


# --- reads -----------------------------------------------------------------------


def test_get_unknown_document_is_none(store: StatusStore) -> None:
    assert store.get_document(DOC) is None


def test_list_documents_ordered_and_filtered(store: StatusStore) -> None:
    ids = ["c" * 64, "a" * 64, "b" * 64]
    for doc_id in ids:
        store.register_document(doc_id, f"{doc_id[0]}.pdf")
    store.mark_processing(ids[1])

    assert [r.doc_id for r in store.list_documents()] == ids
    assert [r.doc_id for r in store.list_documents(DocumentStatus.PENDING)] == [ids[0], ids[2]]
    assert [r.doc_id for r in store.list_documents(DocumentStatus.PROCESSING)] == [ids[1]]
    assert store.list_documents(DocumentStatus.DONE) == []


def _failed_with_attempts(store: StatusStore, doc_id: str, attempts: int) -> None:
    store.register_document(doc_id, "f.pdf")
    for _ in range(attempts):
        store.mark_processing(doc_id)
        store.mark_failed(doc_id, "boom")


@pytest.mark.parametrize("max_attempts", [1, 2, 3])
def test_known_doc_ids_matrix(store: StatusStore, max_attempts: int) -> None:
    SETUP[DocumentStatus.PENDING](store, "pending")
    SETUP[DocumentStatus.PROCESSING](store, "processing")
    SETUP[DocumentStatus.DONE](store, "done")
    SETUP[DocumentStatus.NEEDS_OCR](store, "needs_ocr")
    for attempts in (1, 2, 3):
        _failed_with_attempts(store, f"failed{attempts}", attempts)

    known = store.known_doc_ids(max_attempts)

    expected = {"processing", "done", "needs_ocr"} | {
        f"failed{a}" for a in (1, 2, 3) if a >= max_attempts
    }
    assert known == expected
    assert "pending" not in known


@pytest.mark.parametrize("bad", [0, -1, True])
def test_known_doc_ids_rejects_bad_max_attempts(store: StatusStore, bad: object) -> None:
    with pytest.raises(StatusStoreError, match="max_attempts"):
        store.known_doc_ids(bad)  # type: ignore[arg-type]


# --- recovery ---------------------------------------------------------------------


def test_recover_interrupted_changes_only_processing(store: StatusStore) -> None:
    for status, setup in SETUP.items():
        setup(store, status.value)
    SETUP[DocumentStatus.PROCESSING](store, "processing2")
    store.mark_processing("failed")  # failed -> processing, attempts 2
    others_before = {
        r.doc_id: r
        for r in store.list_documents()
        if r.status is not DocumentStatus.PROCESSING
    }

    changed = store.recover_interrupted()

    assert changed == 3
    for doc_id, attempts in (("processing", 1), ("processing2", 1), ("failed", 2)):
        record = store.get_document(doc_id)
        assert record is not None
        assert (record.status, record.error_message, record.attempts) == (
            DocumentStatus.FAILED,
            INTERRUPTED_MESSAGE,
            attempts,
        )
    for doc_id, before in others_before.items():
        assert store.get_document(doc_id) == before
    assert store.recover_interrupted() == 0


def test_recovered_doc_is_retryable_or_known_by_attempts(store: StatusStore) -> None:
    _to_processing(store, "once")  # attempts 1
    _failed_with_attempts(store, "thrice", 2)
    store.mark_processing("thrice")  # attempts 3

    store.recover_interrupted()

    known = store.known_doc_ids(3)
    assert "once" not in known
    assert "thrice" in known
    assert store.mark_processing("once").attempts == 2


# --- persistence, safety, files ------------------------------------------------------


def test_new_store_on_same_file_sees_data(db_path: Path, store: StatusStore) -> None:
    _to_done(store, DOC)

    reopened = StatusStore(db_path)
    reopened.initialize()

    record = reopened.get_document(DOC)
    assert record is not None
    assert (record.status, record.chunk_count, record.created_at) == (
        DocumentStatus.DONE,
        7,
        START,
    )


@pytest.mark.parametrize(
    "filename",
    [
        "O'Brien \"notes\"; DROP TABLE documents; --.pdf",
        "Lecture 3 — kya haal hai, dost? \U0001F600.pdf",
        "مشین لرننگ.pdf",
    ],
)
def test_filename_with_quotes_and_unicode_round_trips(
    store: StatusStore, db_path: Path, filename: str
) -> None:
    doc_id = "x'); DELETE FROM documents; --"

    store.register_document(doc_id, filename)

    record = store.get_document(doc_id)
    assert record is not None and record.filename == filename
    assert raw(db_path, "SELECT COUNT(*) FROM documents") == [(1,)]


def test_timestamps_stored_as_utc_iso_with_offset(store: StatusStore, db_path: Path) -> None:
    store.register_document(DOC, "a.pdf")

    (created, updated), = raw(
        db_path, "SELECT created_at, updated_at FROM documents WHERE doc_id = ?", (DOC,)
    )

    assert created == "2026-11-20T12:00:00.000000+00:00"
    assert datetime.fromisoformat(str(updated)).utcoffset() == timedelta(0)


def test_non_utc_clock_is_stored_as_utc(db_path: Path) -> None:
    plus5 = timezone(timedelta(hours=5))
    status_store = StatusStore(db_path, clock=lambda: datetime(2026, 11, 20, 17, 0, tzinfo=plus5))
    status_store.initialize()

    record = status_store.register_document(DOC, "a.pdf")

    assert record.created_at == START
    assert record.created_at.tzinfo == timezone.utc


def test_db_file_can_be_deleted_after_use(store: StatusStore, db_path: Path) -> None:
    _to_done(store, DOC)
    _to_failed(store, OTHER)
    store.list_documents()
    store.known_doc_ids(3)
    store.recover_interrupted()
    with pytest.raises(StatusStoreError):
        store.mark_processing(DOC)  # failing transition must close its connection too

    db_path.unlink()

    assert not db_path.exists()
    assert list(db_path.parent.iterdir()) == []


# --- integration with scan_inbox ---------------------------------------------------


def test_scan_inbox_uses_known_doc_ids(tmp_path: Path, store: StatusStore) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    done_pdf = inbox / "done.pdf"
    failed_pdf = inbox / "failed.pdf"
    exhausted_pdf = inbox / "exhausted.pdf"
    fresh_pdf = inbox / "fresh.pdf"
    for i, path in enumerate((done_pdf, failed_pdf, exhausted_pdf, fresh_pdf)):
        path.write_bytes(b"%PDF-1.4 fake content " + bytes([i]))
    done_id = compute_file_doc_id(done_pdf)
    failed_id = compute_file_doc_id(failed_pdf)
    exhausted_id = compute_file_doc_id(exhausted_pdf)
    store.register_document(done_id, done_pdf.name)
    store.mark_processing(done_id)
    store.mark_done(done_id, 5)
    _failed_with_attempts(store, failed_id, 2)
    _failed_with_attempts(store, exhausted_id, 3)

    result = scan_inbox(inbox, known_doc_ids=store.known_doc_ids(3))

    status_by_name = {f.filename: f.status for f in result.files}
    assert status_by_name == {
        "done.pdf": ScanStatus.ALREADY_KNOWN,
        "exhausted.pdf": ScanStatus.ALREADY_KNOWN,
        "failed.pdf": ScanStatus.NEW,
        "fresh.pdf": ScanStatus.NEW,
    }
