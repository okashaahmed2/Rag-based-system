"""Tests for the jobs table and schema v2 of rag_pipeline.status_db.

Real SQLite file in tmp_path, fake clock, fake job id factory.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from rag_pipeline.status_db import (
    JOB_INTERRUPTED_MESSAGE,
    MAX_ERROR_MESSAGE_CHARS,
    ONE_ACTIVE_JOB_INDEX,
    SCHEMA_VERSION,
    DocumentStatus,
    JobKind,
    JobRecord,
    JobStatus,
    StatusStore,
    StatusStoreError,
)

START = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)
DOC = "d" * 64

# Exact schema of version 1 (status_db.py at commit e4f7ec3, before jobs).
V1_DDL = (
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        status TEXT NOT NULL
            CHECK (status IN ('pending', 'processing', 'done', 'failed', 'needs_ocr')),
        error_message TEXT,
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        chunk_count INTEGER CHECK (chunk_count IS NULL OR chunk_count >= 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status)",
    "PRAGMA user_version = 1",
)
V1_ROWS = [
    ("a" * 64, "a.pdf", "done", None, 1, 12, "2026-11-01T10:00:00.000000+00:00",
     "2026-11-01T10:00:05.123456+00:00"),
    ("b" * 64, "Lecture 3 — kya haal hai \U0001F600.pdf", "failed",
     "PdfExtractionError: broken xref", 3, None, "2026-11-02T10:00:00.000000+00:00",
     "2026-11-02T10:01:00.000000+00:00"),
    ("c" * 64, "مشین.pdf", "needs_ocr", None, 1, 0,
     "2026-11-03T10:00:00.000000+00:00", "2026-11-03T10:00:01.000000+00:00"),
    ("e" * 64, "pending.pdf", "pending", None, 0, None, "2026-11-04T10:00:00.000000+00:00",
     "2026-11-04T10:00:00.000000+00:00"),
]


class FakeClock:
    """Deterministic clock: every call returns a time one second later."""

    def __init__(self, start: datetime = START) -> None:
        self.current = start

    def __call__(self) -> datetime:
        value = self.current
        self.current += timedelta(seconds=1)
        return value


class FakeIds:
    """Job ids job-0001, job-0002, ..."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> str:
        self.count += 1
        return f"job-{self.count:04d}"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "db" / "status.db"


@pytest.fixture
def store(db_path: Path) -> StatusStore:
    status_store = StatusStore(db_path, clock=FakeClock(), job_id_factory=FakeIds())
    status_store.initialize()
    return status_store


def raw(path: Path, sql: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    """Run one statement on a separate, always-closed connection."""
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            rows = conn.execute(sql, params).fetchall()
    return rows


def build_v1_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            for statement in V1_DDL:
                conn.execute(statement)
            conn.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)", V1_ROWS)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def documents_snapshot(path: Path) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    """All documents rows (with SQLite storage types) and the documents schema SQL."""
    rows = raw(
        path,
        "SELECT doc_id, filename, status, error_message, attempts, chunk_count, created_at, "
        "updated_at, typeof(error_message), typeof(chunk_count), hex(filename) "
        "FROM documents ORDER BY doc_id",
    )
    schema = raw(
        path,
        "SELECT type, name, sql FROM sqlite_master WHERE tbl_name = 'documents' ORDER BY name",
    )
    return rows, schema


# Ways to bring a fresh job into each status (returns its id).
def _queued(s: StatusStore) -> str:
    job, created = s.create_job_or_get_active(JobKind.SCAN)
    assert created
    return job.job_id


def _running(s: StatusStore) -> str:
    job_id = _queued(s)
    s.mark_job_running(job_id)
    return job_id


def _succeeded(s: StatusStore) -> str:
    job_id = _running(s)
    s.mark_job_succeeded(job_id, {"ingested": 1})
    return job_id


def _failed(s: StatusStore) -> str:
    job_id = _running(s)
    s.mark_job_failed(job_id, "crash")
    return job_id


SETUP: dict[JobStatus, Callable[[StatusStore], str]] = {
    JobStatus.QUEUED: _queued,
    JobStatus.RUNNING: _running,
    JobStatus.SUCCEEDED: _succeeded,
    JobStatus.FAILED: _failed,
}
ACTIONS: dict[str, Callable[[StatusStore, str], JobRecord]] = {
    "running": lambda s, j: s.mark_job_running(j),
    "succeeded": lambda s, j: s.mark_job_succeeded(j, {"x": 1}),
    "failed": lambda s, j: s.mark_job_failed(j, "again"),
}
ALLOWED_FROM: dict[str, set[JobStatus]] = {
    "running": {JobStatus.QUEUED},
    "succeeded": {JobStatus.RUNNING},
    "failed": {JobStatus.RUNNING},
}
INVALID = [
    (state, action)
    for action, allowed in ALLOWED_FROM.items()
    for state in JobStatus
    if state not in allowed
]


# --- schema v2 ---------------------------------------------------------------------


def test_fresh_db_has_version_2_and_both_tables(store: StatusStore, db_path: Path) -> None:
    assert SCHEMA_VERSION == 2
    assert raw(db_path, "PRAGMA user_version") == [(2,)]
    names = {row[0] for row in raw(db_path, "SELECT name FROM sqlite_master")}
    assert {
        "documents", "idx_documents_status", "jobs", "idx_jobs_status", ONE_ACTIVE_JOB_INDEX
    } <= names
    assert raw(db_path, "SELECT COUNT(*) FROM jobs") == [(0,)]


ACTIVE_STATUSES = (JobStatus.QUEUED, JobStatus.RUNNING)
FINISHED_STATUSES = (JobStatus.SUCCEEDED, JobStatus.FAILED)
RAW_INSERT_JOB = (
    "INSERT INTO jobs (job_id, kind, target, status, created_at) VALUES (?, ?, ?, ?, 'x')"
)


def test_jobs_check_accepts_valid_rows(store: StatusStore, db_path: Path) -> None:
    # Every (kind, status) pair is valid; an active row is deleted right away so
    # the one-active index does not reject the next one.
    for i, status in enumerate(JobStatus):
        for kind, target in (("scan", None), ("ingest", "a.pdf")):
            job_id = f"{kind}{i}"
            raw(db_path, RAW_INSERT_JOB, (job_id, kind, target, status.value))
            if status in ACTIVE_STATUSES:
                raw(db_path, "DELETE FROM jobs WHERE job_id = ?", (job_id,))

    assert raw(db_path, "SELECT COUNT(*) FROM jobs") == [(2 * len(FINISHED_STATUSES),)]


@pytest.mark.parametrize("first", ACTIVE_STATUSES, ids=lambda s: s.value)
@pytest.mark.parametrize("second", ACTIVE_STATUSES, ids=lambda s: s.value)
@pytest.mark.parametrize(
    "second_kind", [("scan", None), ("ingest", "b.pdf")], ids=["scan", "ingest"]
)
def test_unique_index_rejects_a_second_active_row(
    store: StatusStore,
    db_path: Path,
    first: JobStatus,
    second: JobStatus,
    second_kind: tuple[str, str | None],
) -> None:
    raw(db_path, RAW_INSERT_JOB, ("one", "scan", None, first.value))

    with pytest.raises(sqlite3.IntegrityError, match=ONE_ACTIVE_JOB_INDEX):
        raw(db_path, RAW_INSERT_JOB, ("two", *second_kind, second.value))

    assert raw(db_path, "SELECT job_id FROM jobs") == [("one",)]


def test_unique_index_rejects_reactivating_a_finished_row(
    store: StatusStore, db_path: Path
) -> None:
    finished = _succeeded(store)
    _queued(store)

    with pytest.raises(sqlite3.IntegrityError, match=ONE_ACTIVE_JOB_INDEX):
        raw(db_path, "UPDATE jobs SET status = 'running' WHERE job_id = ?", (finished,))


def test_unique_index_allows_any_number_of_finished_rows(
    store: StatusStore, db_path: Path
) -> None:
    for i in range(40):
        status = FINISHED_STATUSES[i % 2]
        raw(db_path, RAW_INSERT_JOB, (f"f{i}", "scan", None, status.value))
    raw(db_path, RAW_INSERT_JOB, ("active", "ingest", "a.pdf", "running"))

    assert raw(db_path, "SELECT status, COUNT(*) FROM jobs GROUP BY status ORDER BY status") == [
        ("failed", 20), ("running", 1), ("succeeded", 20)
    ]


def test_index_violation_inside_create_becomes_status_store_error(
    store: StatusStore, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active_id = _running(store)
    # Simulate the in-code check missing the active job: only the index stops it.
    monkeypatch.setattr(StatusStore, "_fetch_active_job", staticmethod(lambda conn: None))

    with pytest.raises(StatusStoreError, match="another job is already active") as info:
        store.create_job_or_get_active(JobKind.SCAN)

    assert isinstance(info.value.__cause__, sqlite3.IntegrityError)
    assert raw(db_path, "SELECT job_id, status FROM jobs") == [(active_id, "running")]


def test_duplicate_job_id_becomes_status_store_error(db_path: Path) -> None:
    store = StatusStore(db_path, clock=FakeClock(), job_id_factory=lambda: "same-id")
    store.initialize()
    store.mark_job_running(_queued(store))
    store.mark_job_succeeded("same-id", {})

    with pytest.raises(StatusStoreError, match="constraint violated"):
        store.create_job_or_get_active(JobKind.SCAN)

    assert raw(db_path, "SELECT COUNT(*) FROM jobs") == [(1,)]


@pytest.mark.parametrize(
    ("kind", "target", "status"),
    [
        ("export", None, "queued"),
        ("scan", None, "done"),
        ("scan", "a.pdf", "queued"),
        ("ingest", None, "queued"),
        ("ingest", "", "queued"),
        ("ingest", "   ", "queued"),
    ],
    ids=["bad-kind", "bad-status", "scan-with-target", "ingest-null", "ingest-empty",
         "ingest-blank"],
)
def test_jobs_check_constraints_reject_bad_rows(
    store: StatusStore, db_path: Path, kind: str, target: str | None, status: str
) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        raw(
            db_path,
            "INSERT INTO jobs (job_id, kind, target, status, created_at) VALUES ('j', ?, ?, ?, 'x')",
            (kind, target, status),
        )
    assert raw(db_path, "SELECT COUNT(*) FROM jobs") == [(0,)]


@pytest.mark.parametrize("column", ["kind", "status", "created_at"])
def test_jobs_not_null_columns(store: StatusStore, db_path: Path, column: str) -> None:
    values = {"kind": "scan", "status": "queued", "created_at": "x"}
    values[column] = None  # type: ignore[assignment]

    with pytest.raises(sqlite3.IntegrityError, match="NOT NULL|CHECK"):
        raw(
            db_path,
            "INSERT INTO jobs (job_id, kind, status, created_at) VALUES ('j', ?, ?, ?)",
            (values["kind"], values["status"], values["created_at"]),
        )


# --- migration -------------------------------------------------------------------------


def test_v1_db_migrates_to_v2_and_keeps_documents_identical(db_path: Path) -> None:
    build_v1_db(db_path)
    before = documents_snapshot(db_path)
    assert raw(db_path, "PRAGMA user_version") == [(1,)]

    store = StatusStore(db_path, clock=FakeClock())
    store.initialize()

    assert raw(db_path, "PRAGMA user_version") == [(2,)]
    names = {row[0] for row in raw(db_path, "SELECT name FROM sqlite_master")}
    assert {"jobs", "idx_jobs_status", ONE_ACTIVE_JOB_INDEX} <= names
    assert raw(db_path, "SELECT COUNT(*) FROM jobs") == [(0,)]
    assert documents_snapshot(db_path) == before
    assert len(before[0]) == len(V1_ROWS)
    record = store.get_document("b" * 64)
    assert record is not None
    assert (record.status, record.attempts, record.filename) == (
        DocumentStatus.FAILED,
        3,
        V1_ROWS[1][1],
    )


def test_migrated_db_initialized_twice_is_harmless(db_path: Path) -> None:
    build_v1_db(db_path)
    before = documents_snapshot(db_path)
    store = StatusStore(db_path, clock=FakeClock(), job_id_factory=FakeIds())
    store.initialize()
    job_id = _queued(store)

    store.initialize()

    assert raw(db_path, "PRAGMA user_version") == [(2,)]
    assert documents_snapshot(db_path) == before
    assert store.get_job(job_id) is not None


def test_migration_logs_versions(db_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    build_v1_db(db_path)

    with caplog.at_level("INFO", logger="rag_pipeline.status_db"):
        StatusStore(db_path).initialize()

    assert "migrated from schema version 1 to 2" in caplog.text


def test_newer_version_raises_and_leaves_file_unchanged(db_path: Path) -> None:
    build_v1_db(db_path)
    raw(db_path, "PRAGMA user_version = 3")
    digest = sha256(db_path)

    with pytest.raises(StatusStoreError, match="schema version 3"):
        StatusStore(db_path).initialize()

    assert sha256(db_path) == digest
    names = {row[0] for row in raw(db_path, "SELECT name FROM sqlite_master")}
    assert "jobs" not in names


def test_v1_store_with_processing_document_still_recovers(db_path: Path) -> None:
    build_v1_db(db_path)
    raw(db_path, "UPDATE documents SET status = 'processing' WHERE doc_id = ?", ("e" * 64,))
    store = StatusStore(db_path, clock=FakeClock())
    store.initialize()

    assert store.recover_interrupted() == 1
    assert store.get_document("e" * 64).status is DocumentStatus.FAILED  # type: ignore[union-attr]


# --- create / transitions -------------------------------------------------------------------


def test_create_scan_job_is_queued(store: StatusStore) -> None:
    job, created = store.create_job_or_get_active(JobKind.SCAN)

    assert created is True
    assert job == JobRecord(
        job_id="job-0001",
        kind=JobKind.SCAN,
        target=None,
        status=JobStatus.QUEUED,
        created_at=START,
        started_at=None,
        finished_at=None,
        error_message=None,
        summary=None,
    )
    assert store.get_job("job-0001") == job


def test_create_ingest_job_keeps_target_and_accepts_plain_strings(store: StatusStore) -> None:
    job, created = store.create_job_or_get_active("ingest", "Lecture 3 — notes.pdf")

    assert created is True
    assert (job.kind, job.target) == (JobKind.INGEST, "Lecture 3 — notes.pdf")


def test_default_job_id_is_uuid4_hex(db_path: Path) -> None:
    store = StatusStore(db_path)
    store.initialize()

    job, _ = store.create_job_or_get_active(JobKind.SCAN)

    assert len(job.job_id) == 32
    assert int(job.job_id, 16) >= 0


def test_happy_path_queued_running_succeeded(store: StatusStore) -> None:
    job_id = _queued(store)

    running = store.mark_job_running(job_id)
    done = store.mark_job_succeeded(job_id, {"ingested": 2, "failed": 1})

    assert (running.status, running.started_at, running.finished_at) == (
        JobStatus.RUNNING,
        START + timedelta(seconds=1),
        None,
    )
    assert done.status is JobStatus.SUCCEEDED
    assert done.started_at == running.started_at
    assert done.finished_at == START + timedelta(seconds=2)
    assert done.summary == {"failed": 1, "ingested": 2}
    assert done.error_message is None
    assert store.get_job(job_id) == done


def test_running_to_failed_without_summary(store: StatusStore) -> None:
    job_id = _running(store)

    failed = store.mark_job_failed(job_id, "  StatusStoreError: disk full  ")

    assert failed.status is JobStatus.FAILED
    assert failed.error_message == "StatusStoreError: disk full"
    assert failed.summary is None
    assert failed.finished_at is not None


def test_running_to_failed_with_partial_summary(store: StatusStore) -> None:
    job_id = _running(store)

    failed = store.mark_job_failed(job_id, "crash", {"ingested": 1})

    assert failed.summary == {"ingested": 1}


@pytest.mark.parametrize(
    ("state", "action"),
    INVALID,
    ids=[f"{state.value}->{action}" for state, action in INVALID],
)
def test_invalid_job_transition_raises_and_changes_nothing(
    store: StatusStore, db_path: Path, state: JobStatus, action: str
) -> None:
    job_id = SETUP[state](store)
    before = store.get_job(job_id)
    rows_before = raw(db_path, "SELECT * FROM jobs")

    with pytest.raises(StatusStoreError, match=f"status {state.value}"):
        ACTIONS[action](store, job_id)

    assert store.get_job(job_id) == before
    assert raw(db_path, "SELECT * FROM jobs") == rows_before


@pytest.mark.parametrize("action", sorted(ACTIONS))
def test_transition_of_unknown_job_raises(store: StatusStore, db_path: Path, action: str) -> None:
    with pytest.raises(StatusStoreError, match="unknown job_id"):
        ACTIONS[action](store, "no-such-job")

    assert raw(db_path, "SELECT COUNT(*) FROM jobs") == [(0,)]


@pytest.mark.parametrize(
    ("kind", "target", "message"),
    [
        ("export", None, "Unknown job kind"),
        (JobKind.SCAN, "a.pdf", "scan job takes no target"),
        (JobKind.INGEST, None, "non-empty target"),
        (JobKind.INGEST, "", "non-empty target"),
        (JobKind.INGEST, "   ", "non-empty target"),
    ],
)
def test_create_rejects_bad_kind_or_target(
    store: StatusStore, db_path: Path, kind: Any, target: str | None, message: str
) -> None:
    with pytest.raises(StatusStoreError, match=message):
        store.create_job_or_get_active(kind, target)

    assert raw(db_path, "SELECT COUNT(*) FROM jobs") == [(0,)]


# --- single active job ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ((JobKind.SCAN, None), (JobKind.SCAN, None)),
        ((JobKind.SCAN, None), (JobKind.INGEST, "b.pdf")),
        ((JobKind.INGEST, "a.pdf"), (JobKind.SCAN, None)),
        ((JobKind.INGEST, "a.pdf"), (JobKind.INGEST, "b.pdf")),
    ],
    ids=["scan-after-scan", "ingest-after-scan", "scan-after-ingest", "ingest-after-ingest"],
)
@pytest.mark.parametrize("active_state", [JobStatus.QUEUED, JobStatus.RUNNING])
def test_second_create_returns_the_active_job(
    store: StatusStore,
    db_path: Path,
    first: tuple[JobKind, str | None],
    second: tuple[JobKind, str | None],
    active_state: JobStatus,
) -> None:
    job, created = store.create_job_or_get_active(*first)
    assert created
    if active_state is JobStatus.RUNNING:
        store.mark_job_running(job.job_id)
    expected = store.get_job(job.job_id)

    again, created_again = store.create_job_or_get_active(*second)

    assert created_again is False
    assert again == expected
    assert raw(db_path, "SELECT COUNT(*) FROM jobs") == [(1,)]


@pytest.mark.parametrize("finish", [_succeeded, _failed], ids=["succeeded", "failed"])
def test_new_job_created_after_active_one_finishes(
    store: StatusStore, finish: Callable[[StatusStore], str]
) -> None:
    old_id = finish(store)

    job, created = store.create_job_or_get_active(JobKind.INGEST, "a.pdf")

    assert created is True
    assert job.job_id != old_id
    assert store.get_active_job() == job


def test_get_active_job(store: StatusStore) -> None:
    assert store.get_active_job() is None

    job_id = _running(store)
    active = store.get_active_job()

    assert active is not None and active.job_id == job_id
    store.mark_job_succeeded(job_id, {})
    assert store.get_active_job() is None


def test_get_unknown_job_is_none(store: StatusStore) -> None:
    assert store.get_job("nope") is None


CONCURRENCY_ROUNDS = 50


def test_concurrent_creates_make_exactly_one_job_per_round(db_path: Path) -> None:
    store = StatusStore(db_path)  # real clock and uuid ids: the threads share them
    store.initialize()
    barrier = threading.Barrier(2)
    results: dict[int, list[tuple[str, bool]]] = {0: [], 1: []}
    errors: list[BaseException] = []

    def worker(slot: int, kind: JobKind, target: str | None) -> None:
        try:
            for _ in range(CONCURRENCY_ROUNDS):
                barrier.wait(timeout=30)
                job, created = store.create_job_or_get_active(kind, target)
                results[slot].append((job.job_id, created))
                barrier.wait(timeout=30)  # both done; main-side finish happens below
                if created:
                    store.mark_job_running(job.job_id)
                    store.mark_job_succeeded(job.job_id, {"round": len(results[slot])})
                barrier.wait(timeout=30)  # finished before the next round starts
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            errors.append(exc)
            barrier.abort()

    threads = [
        threading.Thread(target=worker, args=(0, JobKind.SCAN, None)),
        threading.Thread(target=worker, args=(1, JobKind.INGEST, "a.pdf")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert errors == []
    assert len(results[0]) == len(results[1]) == CONCURRENCY_ROUNDS
    for (id0, created0), (id1, created1) in zip(results[0], results[1]):
        assert id0 == id1
        assert created0 != created1  # exactly one caller created the job
    total = raw(db_path, "SELECT COUNT(*), COUNT(DISTINCT job_id) FROM jobs")
    assert total == [(CONCURRENCY_ROUNDS, CONCURRENCY_ROUNDS)]
    assert raw(db_path, "SELECT DISTINCT status FROM jobs") == [("succeeded",)]


# --- recovery ----------------------------------------------------------------------------


@pytest.mark.parametrize("active_state", ACTIVE_STATUSES, ids=lambda s: s.value)
def test_recover_interrupted_jobs(store: StatusStore, active_state: JobStatus) -> None:
    succeeded = _succeeded(store)
    failed = _failed(store)
    active = SETUP[active_state](store)
    others_before = {j: store.get_job(j) for j in (succeeded, failed)}

    changed = store.recover_interrupted_jobs()

    assert changed == 1
    job = store.get_job(active)
    assert job is not None
    assert (job.status, job.error_message) == (JobStatus.FAILED, JOB_INTERRUPTED_MESSAGE)
    assert job.finished_at is not None
    for job_id, before in others_before.items():
        assert store.get_job(job_id) == before
    assert store.get_active_job() is None
    assert store.recover_interrupted_jobs() == 0
    assert store.create_job_or_get_active(JobKind.SCAN)[1] is True  # no stuck job


def test_recover_interrupted_jobs_leaves_documents_alone(store: StatusStore) -> None:
    store.register_document(DOC, "a.pdf")
    store.mark_processing(DOC)
    _running(store)

    store.recover_interrupted_jobs()

    assert store.get_document(DOC).status is DocumentStatus.PROCESSING  # type: ignore[union-attr]


# --- summary & error message ------------------------------------------------------------------


def test_summary_round_trip_with_unicode_and_nesting(store: StatusStore, db_path: Path) -> None:
    summary = {
        "zeta": 1,
        "alpha": [1, 2.5, None, True],
        "files": {"Kya haal hai, dost? \U0001F600.pdf": "done", "مشین.pdf": "failed"},
        "note": "theek hai — sab sahi",
    }
    job_id = _running(store)

    record = store.mark_job_succeeded(job_id, summary)

    assert record.summary == summary
    assert store.get_job(job_id).summary == summary  # type: ignore[union-attr]
    (stored,), = raw(db_path, "SELECT summary_json FROM jobs WHERE job_id = ?", (job_id,))
    assert str(stored).startswith('{"alpha"')  # sorted keys
    assert "\U0001F600" in str(stored)  # ensure_ascii=False


@pytest.mark.parametrize(
    "bad",
    [{"x": object()}, {"x": {1, 2}}, {"x": math.nan}, {"x": math.inf}, ["not", "a", "dict"], None],
    ids=["object", "set", "nan", "inf", "list", "none"],
)
def test_bad_summary_on_success_raises_before_any_write(
    store: StatusStore, db_path: Path, bad: Any
) -> None:
    job_id = _running(store)
    rows_before = raw(db_path, "SELECT * FROM jobs")

    with pytest.raises(StatusStoreError, match="summary"):
        store.mark_job_succeeded(job_id, bad)

    assert raw(db_path, "SELECT * FROM jobs") == rows_before
    assert store.get_job(job_id).status is JobStatus.RUNNING  # type: ignore[union-attr]


def test_bad_summary_on_failure_raises_before_any_write(store: StatusStore) -> None:
    job_id = _running(store)

    with pytest.raises(StatusStoreError, match="not JSON-serialisable"):
        store.mark_job_failed(job_id, "crash", {"x": object()})

    assert store.get_job(job_id).status is JobStatus.RUNNING  # type: ignore[union-attr]


def test_long_job_error_message_is_trimmed(store: StatusStore) -> None:
    job_id = _running(store)

    record = store.mark_job_failed(job_id, "x" * (MAX_ERROR_MESSAGE_CHARS + 500))

    assert record.error_message == "x" * MAX_ERROR_MESSAGE_CHARS


@pytest.mark.parametrize("message", ["", "   \n"])
def test_empty_job_error_message_rejected(store: StatusStore, message: str) -> None:
    job_id = _running(store)

    with pytest.raises(StatusStoreError, match="error_message must not be empty"):
        store.mark_job_failed(job_id, message)

    assert store.get_job(job_id).status is JobStatus.RUNNING  # type: ignore[union-attr]


def test_job_timestamps_stored_as_utc_iso(store: StatusStore, db_path: Path) -> None:
    job_id = _succeeded(store)

    (created, started, finished), = raw(
        db_path, "SELECT created_at, started_at, finished_at FROM jobs WHERE job_id = ?", (job_id,)
    )

    assert created == "2026-11-20T12:00:00.000000+00:00"
    assert started == "2026-11-20T12:00:01.000000+00:00"
    assert finished == "2026-11-20T12:00:02.000000+00:00"


def test_db_file_can_be_deleted_after_job_use(store: StatusStore, db_path: Path) -> None:
    _succeeded(store)
    job_id = _running(store)
    store.get_active_job()
    store.create_job_or_get_active(JobKind.SCAN)
    with pytest.raises(StatusStoreError):
        store.mark_job_running(job_id)  # failing transition must close its connection too
    store.recover_interrupted_jobs()

    db_path.unlink()

    assert not db_path.exists()
