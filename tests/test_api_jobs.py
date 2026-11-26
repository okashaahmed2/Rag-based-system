"""Tests for POST /ingest/scan, GET /status/{job_id}, the job runner and its summary.

tmp_path folders and stores, stub embedding model, mocked Ollama, fake inbox
runner unless a test says otherwise. TestClient runs a background task before
the request call returns, so a job is finished when ``client.post`` returns.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from rag_pipeline.api import app as api_app
from rag_pipeline.api.app import create_app, ingest_scan, job_status
from rag_pipeline.api.jobs import (
    MAX_SUMMARY_ERROR_CHARS,
    MAX_SUMMARY_FILES,
    run_job,
    summary_from_report,
)
from rag_pipeline.config import Settings
from rag_pipeline.inbox_ingestion import FileMove, InboxRunReport, LeftInInbox, MoveFailure
from rag_pipeline.ingestion import IngestContext, IngestOutcome, IngestResult
from rag_pipeline.status_db import (
    JOB_INTERRUPTED_MESSAGE,
    DocumentStatus,
    JobKind,
    JobStatus,
    StatusStore,
    StatusStoreError,
)
from tests.test_api_app import ClientFactory, SpyLoader, ollama_ok, settings_for

HEX32 = re.compile(r"[0-9a-f]{32}")


# --- helpers -------------------------------------------------------------------------


def result(
    filename: str,
    outcome: IngestOutcome,
    *,
    attempts: int | None = 1,
    skip_reason: str | None = None,
    error: str | None = None,
    path: Path | None = None,
) -> IngestResult:
    return IngestResult(
        path=path if path is not None else Path("inbox") / filename,
        doc_id="0" * 64,
        filename=filename,
        outcome=outcome,
        chunk_count=None,
        error_message=error,
        attempts=attempts,
        skip_reason=skip_reason,
        duration_seconds=0.25,
    )


def report_of(
    results: list[IngestResult],
    *,
    moves: int = 0,
    left: int = 0,
    move_failures: int = 0,
    scan_errors: int = 0,
) -> InboxRunReport:
    errors = tuple(LeftInInbox(Path(f"e{i}.pdf"), "scan error: x") for i in range(scan_errors))
    return InboxRunReport(
        ingest_results=tuple(results),
        moves=tuple(FileMove(Path(f"m{i}.pdf"), Path(f"p/m{i}.pdf")) for i in range(moves)),
        left_in_inbox=tuple(LeftInInbox(Path(f"l{i}.pdf"), "pending") for i in range(left))
        + errors,
        move_failures=tuple(
            MoveFailure(Path(f"f{i}.pdf"), Path("p"), "PermissionError: busy")
            for i in range(move_failures)
        ),
        scan_errors=errors,
    )


SAMPLE_REPORT = report_of(
    [
        result("a.pdf", IngestOutcome.DONE),
        result("scan.pdf", IngestOutcome.NEEDS_OCR),
        result("broken.pdf", IngestOutcome.FAILED, error="PdfExtractionError: broken xref"),
    ],
    moves=2,
    left=1,
)


class FakeRunner:
    """inbox_runner stand-in: records its calls, returns a report or raises."""

    def __init__(
        self, report: InboxRunReport = SAMPLE_REPORT, error: Exception | None = None
    ) -> None:
        self.report = report
        self.error = error
        self.calls: list[tuple[Any, ...]] = []

    def __call__(
        self,
        context: IngestContext,
        inbox_dir: Path,
        processed_dir: Path,
        failed_dir: Path,
        needs_ocr_dir: Path,
    ) -> InboxRunReport:
        self.calls.append((context, inbox_dir, processed_dir, failed_dir, needs_ocr_dir))
        if self.error is not None:
            raise self.error
        return self.report


def make_app(tmp_path: Path, runner: Any = None, **overrides: Any) -> tuple[Any, Settings]:
    settings = settings_for(tmp_path, **overrides)
    kwargs: dict[str, Any] = {} if runner is None else {"inbox_runner": runner}
    app = create_app(
        settings,
        model_loader=SpyLoader(),
        http_client_factory=ClientFactory(ollama_ok()),
        configure_logging=False,
        **kwargs,
    )
    return app, settings


def store_of(app: Any) -> StatusStore:
    store: StatusStore = app.state.pipeline.ingest_context.store
    return store


@pytest.fixture(autouse=True)
def clear_chroma_system_cache() -> Iterator[None]:
    yield
    SharedSystemClient.clear_system_cache()


# --- structure -----------------------------------------------------------------------


def test_new_routes_are_plain_functions_with_the_right_methods(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    routes = {r.path: r for r in app.routes if isinstance(r, APIRoute)}

    scan, status = routes["/ingest/scan"], routes["/status/{job_id}"]

    assert scan.endpoint is ingest_scan and status.endpoint is job_status
    assert scan.methods == {"POST"} and status.methods == {"GET"}
    assert scan.status_code == 202
    for endpoint in (ingest_scan, job_status, run_job):
        assert not inspect.iscoroutinefunction(endpoint)


def test_openapi_shows_real_fields(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)

    schemas = app.openapi()["components"]["schemas"]

    assert set(schemas["DocumentCounts"]["properties"]) == {s.value for s in DocumentStatus}
    assert set(schemas["HealthResponse"]["properties"]) >= {"documents", "active_job"}
    assert set(schemas["ActiveJob"]["properties"]) == {"job_id", "kind", "status"}
    assert set(schemas["ScanResponse"]["properties"]) == {
        "job_id", "kind", "status", "already_running"
    }
    assert set(schemas["JobResponse"]["properties"]) == {
        "job_id", "kind", "target", "status", "created_at", "started_at", "finished_at",
        "error_message", "summary",
    }
    status_404 = app.openapi()["paths"]["/status/{job_id}"]["get"]["responses"]["404"]
    assert status_404["content"]["application/json"]["schema"]["$ref"].endswith("ErrorResponse")


# --- startup job recovery ------------------------------------------------------------


@pytest.mark.parametrize("left_state", [JobStatus.QUEUED, JobStatus.RUNNING])
def test_startup_recovers_an_interrupted_job(
    tmp_path: Path, left_state: JobStatus, caplog: pytest.LogCaptureFixture
) -> None:
    settings = settings_for(tmp_path)
    store = StatusStore(settings.status_db_path)
    store.initialize()
    job, _ = store.create_job_or_get_active(JobKind.SCAN)
    if left_state is JobStatus.RUNNING:
        store.mark_job_running(job.job_id)
    app, _ = make_app(tmp_path, FakeRunner())

    with caplog.at_level(logging.INFO, logger="rag_pipeline.api.app"):
        with TestClient(app) as client:
            body = client.get(f"/status/{job.job_id}").json()
            health = client.get("/health").json()

    assert body["status"] == "failed"
    assert body["error_message"] == JOB_INTERRUPTED_MESSAGE
    assert body["finished_at"] is not None
    assert health["active_job"] is None
    assert "Startup recovery: 1 interrupted job(s) marked failed" in caplog.text


def test_job_recovery_runs_exactly_once_before_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = StatusStore.recover_interrupted_jobs

    def counting(self: StatusStore) -> int:
        calls.append(1)
        return real(self)

    monkeypatch.setattr(StatusStore, "recover_interrupted_jobs", counting)
    app, _ = make_app(tmp_path, FakeRunner())

    with TestClient(app) as client:
        assert calls == [1]  # already done when the first request can be sent
        for _ in range(2):
            client.post("/ingest/scan")
            client.get("/health")

    assert calls == [1]


def test_job_recovery_failure_stops_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(self: StatusStore) -> int:
        raise StatusStoreError("disk gone")

    monkeypatch.setattr(StatusStore, "recover_interrupted_jobs", broken)
    app, _ = make_app(tmp_path, FakeRunner())

    with caplog.at_level(logging.ERROR, logger="rag_pipeline.api.app"):
        with pytest.raises(StatusStoreError, match="disk gone"):
            with TestClient(app):
                pass

    assert "job recovery failed" in caplog.text


# --- POST /ingest/scan ------------------------------------------------------------------


def test_scan_creates_a_job_that_runs_and_succeeds(tmp_path: Path) -> None:
    runner = FakeRunner()
    app, settings = make_app(tmp_path, runner)

    with TestClient(app) as client:
        response = client.post("/ingest/scan")
        status = client.get(f"/status/{response.json()['job_id']}")
        context = app.state.pipeline.ingest_context

    assert response.status_code == 202
    body = response.json()
    assert HEX32.fullmatch(body["job_id"])
    assert body == {
        "job_id": body["job_id"], "kind": "scan", "status": "queued", "already_running": False
    }
    assert runner.calls == [
        (
            context,
            settings.inbox_dir,
            settings.processed_dir,
            settings.failed_dir,
            settings.needs_ocr_dir,
        )
    ]
    assert status.status_code == 200
    job = status.json()
    assert (job["kind"], job["target"], job["status"], job["error_message"]) == (
        "scan", None, "succeeded", None
    )
    assert job["created_at"] and job["started_at"] and job["finished_at"]
    expected = summary_from_report(SAMPLE_REPORT, 0.0)
    assert {k: v for k, v in job["summary"].items() if k != "duration_seconds"} == {
        k: v for k, v in expected.items() if k != "duration_seconds"
    }
    assert job["summary"]["duration_seconds"] >= 0


@pytest.mark.parametrize(
    ("kind", "target"), [(JobKind.SCAN, None), (JobKind.INGEST, "a.pdf")], ids=["scan", "ingest"]
)
@pytest.mark.parametrize("state", [JobStatus.QUEUED, JobStatus.RUNNING])
def test_scan_returns_the_active_job_and_starts_nothing(
    tmp_path: Path, kind: JobKind, target: str | None, state: JobStatus
) -> None:
    runner = FakeRunner()
    app, _ = make_app(tmp_path, runner)

    with TestClient(app) as client:
        store = store_of(app)
        active, _ = store.create_job_or_get_active(kind, target)
        if state is JobStatus.RUNNING:
            store.mark_job_running(active.job_id)
        response = client.post("/ingest/scan")
        after = store.get_job(active.job_id)
        health = client.get("/health").json()

    assert response.status_code == 202
    assert response.json() == {
        "job_id": active.job_id,
        "kind": kind.value,
        "status": state.value,
        "already_running": True,
    }
    assert runner.calls == []
    assert after is not None and after.status is state
    assert health["active_job"] == {
        "job_id": active.job_id, "kind": kind.value, "status": state.value
    }


def test_runner_crash_fails_the_job_and_a_new_scan_can_start(tmp_path: Path) -> None:
    runner = FakeRunner(error=RuntimeError("inbox exploded"))
    app, _ = make_app(tmp_path, runner)

    with TestClient(app) as client:
        first = client.post("/ingest/scan").json()
        failed = client.get(f"/status/{first['job_id']}").json()
        second = client.post("/ingest/scan").json()

    assert failed["status"] == "failed"
    assert failed["error_message"] == "RuntimeError: inbox exploded"
    assert failed["summary"] is None
    assert failed["finished_at"] is not None
    assert second["already_running"] is False
    assert second["job_id"] != first["job_id"]
    assert len(runner.calls) == 2


def test_runner_crash_is_logged_with_traceback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    app, _ = make_app(tmp_path, FakeRunner(error=ValueError("bad folder")))

    with caplog.at_level(logging.ERROR, logger="rag_pipeline.api.jobs"):
        with TestClient(app) as client:
            job_id = client.post("/ingest/scan").json()["job_id"]

    crash = [r for r in caplog.records if "crashed" in r.getMessage()]
    assert len(crash) == 1 and job_id in crash[0].getMessage()
    assert crash[0].exc_info is not None


def test_failure_to_record_success_is_critical_and_job_ends_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(self: StatusStore, job_id: str, summary: Any) -> Any:
        raise StatusStoreError("database is locked")

    app, _ = make_app(tmp_path, FakeRunner())

    with caplog.at_level(logging.INFO, logger="rag_pipeline.api.jobs"):
        with TestClient(app) as client:
            monkeypatch.setattr(StatusStore, "mark_job_succeeded", broken)
            response = client.post("/ingest/scan")  # must not raise
            job_id = response.json()["job_id"]
            job = client.get(f"/status/{job_id}").json()
            again = client.post("/ingest/scan").json()

    assert response.status_code == 202
    critical = [
        r for r in caplog.records if r.levelno == logging.CRITICAL and job_id in r.getMessage()
    ]
    assert len(critical) == 1
    assert "database is locked" in critical[0].getMessage()
    assert job["status"] == "failed"
    assert job["error_message"].startswith("could not record success:")
    assert again["already_running"] is False  # not stuck


def test_failure_to_record_anything_is_critical_and_nothing_escapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(self: StatusStore, *args: Any) -> Any:
        raise StatusStoreError("database is locked")

    app, _ = make_app(tmp_path, FakeRunner())

    with caplog.at_level(logging.INFO, logger="rag_pipeline.api.jobs"):
        with TestClient(app) as client:
            monkeypatch.setattr(StatusStore, "mark_job_succeeded", broken)
            monkeypatch.setattr(StatusStore, "mark_job_failed", broken)
            response = client.post("/ingest/scan")
            job = client.get(f"/status/{response.json()['job_id']}").json()

    assert response.status_code == 202
    assert [r.levelno for r in caplog.records].count(logging.CRITICAL) == 2
    assert job["status"] == "running"  # left for the next startup recovery


def test_failure_to_mark_running_is_critical_and_runner_not_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(self: StatusStore, job_id: str) -> Any:
        raise StatusStoreError("database is locked")

    runner = FakeRunner()
    app, _ = make_app(tmp_path, runner)

    with caplog.at_level(logging.INFO, logger="rag_pipeline.api.jobs"):
        with TestClient(app) as client:
            monkeypatch.setattr(StatusStore, "mark_job_running", broken)
            response = client.post("/ingest/scan")

    assert response.status_code == 202
    assert runner.calls == []
    critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(critical) == 1 and response.json()["job_id"] in critical[0].getMessage()


def test_base_exception_from_runner_is_not_caught(tmp_path: Path) -> None:
    class Stop(BaseException):
        pass

    app, _ = make_app(tmp_path, FakeRunner())
    with TestClient(app):
        context = app.state.pipeline.ingest_context
        settings = app.state.pipeline.settings
        job, _ = context.store.create_job_or_get_active(JobKind.SCAN)

        with pytest.raises(Stop):
            run_job(job.job_id, context, settings, FakeRunner(error=Stop()))  # type: ignore[arg-type]

        assert context.store.get_job(job.job_id).status is JobStatus.RUNNING  # type: ignore[union-attr]


# --- GET /status/{job_id} ---------------------------------------------------------------


def test_status_of_unknown_valid_id_is_404(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path, FakeRunner())

    with TestClient(app) as client:
        response = client.get(f"/status/{'0' * 32}")

    assert response.status_code == 404
    assert response.json() == {"detail": "job not found"}


@pytest.mark.parametrize(
    "bad_id",
    ["abc", "A" * 32, "a" * 33, "a" * 31, "g" * 32, "../x", "%2E%2E%2Fx", "a" * 32 + "%0A",
     "a" * 32 + "%20", "a" * 32 + ".json"],
    ids=["short", "uppercase", "33", "31", "non-hex", "dotdot", "encoded-dotdot",
         "trailing-newline", "trailing-space", "suffix"],
)
def test_malformed_ids_are_404_without_touching_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_id: str
) -> None:
    lookups: list[str] = []
    real = StatusStore.get_job

    def spy(self: StatusStore, job_id: str) -> Any:
        lookups.append(job_id)
        return real(self, job_id)

    monkeypatch.setattr(StatusStore, "get_job", spy)
    app, _ = make_app(tmp_path, FakeRunner())

    with TestClient(app) as client:
        response = client.get(f"/status/{bad_id}")

    assert response.status_code == 404
    assert lookups == []


# --- summary_from_report ----------------------------------------------------------------


def test_summary_counts_and_entries() -> None:
    report = report_of(
        [
            result("a.pdf", IngestOutcome.DONE),
            result("b.pdf", IngestOutcome.DONE, attempts=2),
            result("scan.pdf", IngestOutcome.NEEDS_OCR),
            result("bad.pdf", IngestOutcome.FAILED, error="  PdfExtractionError: x  "),
            result("old.pdf", IngestOutcome.SKIPPED, attempts=3, skip_reason="already done"),
            result("gone.pdf", IngestOutcome.SKIPPED, attempts=None, skip_reason="unreadable"),
        ],
        moves=3,
        left=2,
        move_failures=1,
        scan_errors=1,
    )

    summary = summary_from_report(report, 1.23456)

    assert summary == {
        "outcomes": {"done": 2, "needs_ocr": 1, "failed": 1, "skipped": 2},
        "moved": 3,
        "left_in_inbox": 3,  # 2 left + 1 scan error (InboxRunReport includes them)
        "move_failures": 1,
        "scan_errors": 1,
        "duration_seconds": 1.235,
        "files": [
            {"filename": "a.pdf", "outcome": "done", "attempts": 1},
            {"filename": "b.pdf", "outcome": "done", "attempts": 2},
            {"filename": "scan.pdf", "outcome": "needs_ocr", "attempts": 1},
            {"filename": "bad.pdf", "outcome": "failed", "attempts": 1,
             "error": "PdfExtractionError: x"},
            {"filename": "old.pdf", "outcome": "skipped", "attempts": 3,
             "skip_reason": "already done"},
            {"filename": "gone.pdf", "outcome": "skipped", "attempts": None,
             "skip_reason": "unreadable"},
        ],
        "files_truncated": False,
    }
    json.dumps(summary, allow_nan=False)  # storable as the job summary


def test_summary_of_empty_report_has_every_outcome() -> None:
    summary = summary_from_report(report_of([]), 0.0)

    assert summary["outcomes"] == {o.value: 0 for o in IngestOutcome}
    assert summary["files"] == [] and summary["files_truncated"] is False
    assert summary["duration_seconds"] == 0.0


def test_summary_negative_duration_is_clamped() -> None:
    assert summary_from_report(report_of([]), -0.5)["duration_seconds"] == 0.0


def test_summary_error_is_trimmed() -> None:
    long_error = "E" * (MAX_SUMMARY_ERROR_CHARS + 50)

    summary = summary_from_report(
        report_of([result("x.pdf", IngestOutcome.FAILED, error=long_error)]), 0.1
    )

    assert summary["files"][0]["error"] == "E" * MAX_SUMMARY_ERROR_CHARS


def failed_with(path: Path, error: str) -> dict[str, Any]:
    """The summary entry of one failed file at ``path`` with ``error``."""
    report = report_of([result(path.name, IngestOutcome.FAILED, error=error, path=path)])
    entry: dict[str, Any] = summary_from_report(report, 0.1)["files"][0]
    return entry


def test_summary_error_absolute_path_becomes_file_name(tmp_path: Path) -> None:
    path = tmp_path / "inbox" / "broken.pdf"
    error = f"PdfExtractionError: Invalid or empty PDF {path}: Failed to open stream"

    entry = failed_with(path, error)

    assert entry["error"] == (
        "PdfExtractionError: Invalid or empty PDF broken.pdf: Failed to open stream"
    )
    assert str(tmp_path) not in entry["error"]
    assert ":\\" not in entry["error"] and "\\" not in entry["error"] and "/" not in entry["error"]


def test_summary_error_every_occurrence_and_folder_are_replaced(tmp_path: Path) -> None:
    path = tmp_path / "inbox" / "locked.pdf"
    error = f"PDF {path} is password-protected (in {path.parent}; again {path})"

    entry = failed_with(path, error)

    assert entry["error"] == "PDF locked.pdf is password-protected (in locked.pdf; again locked.pdf)"
    assert str(tmp_path) not in entry["error"]


def test_summary_error_resolved_path_is_replaced(tmp_path: Path) -> None:
    resolved = tmp_path.resolve() / "inbox" / "r.pdf"
    unresolved = tmp_path / "inbox" / ".." / "inbox" / "r.pdf"
    assert str(unresolved) != str(resolved)

    entry = failed_with(unresolved, f"PdfExtractionError: cannot open {resolved}")

    assert entry["error"] == "PdfExtractionError: cannot open r.pdf"


def test_summary_error_without_a_path_is_unchanged(tmp_path: Path) -> None:
    error = "PdfExtractionError: broken xref at offset 1024 (see page 2/3)"

    assert failed_with(tmp_path / "inbox" / "x.pdf", error)["error"] == error


def test_summary_error_is_trimmed_after_path_replacement(tmp_path: Path) -> None:
    path = tmp_path / ("d" * 120) / "long.pdf"
    tail = "Z" * (MAX_SUMMARY_ERROR_CHARS - len("E: long.pdf ") - 10)
    error = f"E: {path} {tail}TRAILING_TEXT_BEYOND_LIMIT"
    assert len(error) > MAX_SUMMARY_ERROR_CHARS + 100  # path made it long

    entry = failed_with(path, error)

    expected = f"E: long.pdf {tail}TRAILING_TEXT_BEYOND_LIMIT"[:MAX_SUMMARY_ERROR_CHARS]
    assert entry["error"] == expected
    assert len(entry["error"]) == MAX_SUMMARY_ERROR_CHARS
    assert entry["error"].endswith("TRAILING_T")  # text after the path survived the replacement


@pytest.mark.parametrize(
    ("count", "truncated"),
    [(MAX_SUMMARY_FILES - 1, False), (MAX_SUMMARY_FILES, False), (MAX_SUMMARY_FILES + 50, True)],
)
def test_summary_files_are_capped(count: int, truncated: bool) -> None:
    results = [result(f"f{i:04d}.pdf", IngestOutcome.DONE) for i in range(count)]

    summary = summary_from_report(report_of(results), 0.1)

    assert len(summary["files"]) == min(count, MAX_SUMMARY_FILES)
    assert summary["files"][0]["filename"] == "f0000.pdf"
    assert summary["files_truncated"] is truncated
    assert summary["outcomes"]["done"] == count  # counts are never capped


# --- real ingest_inbox -----------------------------------------------------------------


def write_pdf(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 60), text, fontsize=9)
        pdf.save(path)
    return path


def test_scan_with_the_real_inbox_runner(tmp_path: Path) -> None:
    inbox = tmp_path / "in"
    text = "\n".join(f"Line {i} explains warranty topic {i}." for i in range(30))
    write_pdf(inbox / "good.pdf", text)
    (inbox / "broken.pdf").write_bytes(b"%PDF-1.7 this is not really a pdf")
    app, settings = make_app(
        tmp_path,
        inbox_dir=inbox,
        processed_dir=tmp_path / "out" / "processed",
        failed_dir=tmp_path / "out" / "failed",
        needs_ocr_dir=tmp_path / "out" / "needs_ocr",
    )
    assert app.state is not None

    with TestClient(app) as client:
        job_id = client.post("/ingest/scan").json()["job_id"]
        job = client.get(f"/status/{job_id}").json()
        health = client.get("/health").json()

    assert job["status"] == "succeeded"
    summary = job["summary"]
    assert summary["outcomes"] == {"done": 1, "needs_ocr": 0, "failed": 1, "skipped": 0}
    assert (summary["moved"], summary["left_in_inbox"], summary["move_failures"]) == (1, 1, 0)
    assert summary["scan_errors"] == 0
    by_name = {f["filename"]: f for f in summary["files"]}
    assert by_name["good.pdf"] == {"filename": "good.pdf", "outcome": "done", "attempts": 1}
    assert by_name["broken.pdf"]["outcome"] == "failed"
    assert by_name["broken.pdf"]["error"]
    assert sorted(p.name for p in settings.processed_dir.iterdir()) == ["good.pdf"]
    assert sorted(p.name for p in inbox.iterdir()) == ["broken.pdf"]  # retryable: stays
    assert health["documents"] == {
        "pending": 0, "processing": 0, "done": 1, "failed": 1, "needs_ocr": 0
    }
    assert health["active_job"] is None
    assert health["collection"]["chunk_count"] > 0


def test_default_inbox_runner_is_ingest_inbox() -> None:
    signature = inspect.signature(api_app.create_app)

    assert signature.parameters["inbox_runner"].default is api_app.ingest_inbox
