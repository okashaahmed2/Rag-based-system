"""Tests for POST /ingest (one file already in the inbox).

tmp_path folders and stores, stub embedding model, mocked Ollama. The real
:func:`ingest_one_file` runs unless a test injects a fake runner. TestClient
runs a background task before ``client.post`` returns.
"""

from __future__ import annotations

import inspect
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from rag_pipeline.api import app as api_app
from rag_pipeline.api.app import JOB_CONFLICT, create_app, ingest_file
from rag_pipeline.api.jobs import run_ingest_job
from rag_pipeline.api.validation import FILE_NOT_FOUND
from rag_pipeline.config import Settings
from rag_pipeline.inbox_ingestion import InboxRunReport
from rag_pipeline.ingestion import IngestContext
from rag_pipeline.status_db import JobKind, JobStatus, StatusStore
from tests.test_api_app import ClientFactory, SpyLoader, ollama_ok, settings_for
from tests.test_api_jobs import HEX32, SAMPLE_REPORT, write_pdf

TEXT = "\n".join(f"Line {i} explains warranty topic {i}." for i in range(30))


class FakeSingleFileRunner:
    """single_file_runner stand-in: records its calls, returns a report or raises."""

    def __init__(
        self, report: InboxRunReport = SAMPLE_REPORT, error: Exception | None = None
    ) -> None:
        self.report = report
        self.error = error
        self.calls: list[tuple[Any, ...]] = []

    def __call__(
        self,
        context: IngestContext,
        path: Path,
        processed_dir: Path,
        failed_dir: Path,
        needs_ocr_dir: Path,
        *,
        inbox_dir: Path,
    ) -> InboxRunReport:
        self.calls.append((context, path, processed_dir, failed_dir, needs_ocr_dir, inbox_dir))
        if self.error is not None:
            raise self.error
        return self.report


def make_app(tmp_path: Path, runner: Any = None) -> tuple[Any, Settings]:
    settings = settings_for(
        tmp_path,
        inbox_dir=tmp_path / "in",
        processed_dir=tmp_path / "out" / "processed",
        failed_dir=tmp_path / "out" / "failed",
        needs_ocr_dir=tmp_path / "out" / "needs_ocr",
    )
    settings.inbox_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {} if runner is None else {"single_file_runner": runner}
    app = create_app(
        settings,
        model_loader=SpyLoader(),
        http_client_factory=ClientFactory(ollama_ok()),
        configure_logging=False,
        **kwargs,
    )
    return app, settings


def job_rows(settings: Settings) -> int:
    with sqlite3.connect(settings.status_db_path) as conn:
        count: int = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    return count


def names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def store_of(app: Any) -> StatusStore:
    store: StatusStore = app.state.pipeline.ingest_context.store
    return store


@pytest.fixture(autouse=True)
def clear_chroma_system_cache() -> Iterator[None]:
    yield
    SharedSystemClient.clear_system_cache()


# --- structure -----------------------------------------------------------------------


def test_ingest_route_is_a_plain_post_with_202(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    route = {r.path: r for r in app.routes if isinstance(r, APIRoute)}["/ingest"]

    assert route.endpoint is ingest_file
    assert route.methods == {"POST"} and route.status_code == 202
    assert not inspect.iscoroutinefunction(ingest_file)
    assert not inspect.iscoroutinefunction(run_ingest_job)


def test_openapi_shows_the_request_and_the_409_model(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    spec = app.openapi()
    schemas = spec["components"]["schemas"]
    post = spec["paths"]["/ingest"]["post"]

    assert set(schemas["IngestRequest"]["properties"]) == {"filename"}
    assert schemas["IngestRequest"]["required"] == ["filename"]
    assert post["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "IngestRequest"
    )
    assert set(schemas["IngestResponse"]["properties"]) == {
        "job_id", "kind", "target", "status", "already_running"
    }
    assert set(schemas["IngestConflictResponse"]["properties"]) == {"detail", "active_job"}
    assert set(schemas["ConflictingJob"]["properties"]) == {"job_id", "kind", "status", "target"}
    ref_409 = post["responses"]["409"]["content"]["application/json"]["schema"]["$ref"]
    ref_404 = post["responses"]["404"]["content"]["application/json"]["schema"]["$ref"]
    assert ref_409.endswith("IngestConflictResponse") and ref_404.endswith("ErrorResponse")
    assert "202" in post["responses"]


def test_default_single_file_runner_is_ingest_one_file() -> None:
    signature = inspect.signature(api_app.create_app)

    assert signature.parameters["single_file_runner"].default is api_app.ingest_one_file


# --- 202: real single-file ingestion ---------------------------------------------------


def test_ingest_one_file_succeeds_and_places_it(tmp_path: Path) -> None:
    app, settings = make_app(tmp_path)
    write_pdf(settings.inbox_dir / "good.pdf", TEXT)
    write_pdf(settings.inbox_dir / "other.pdf", TEXT + " other")

    with TestClient(app) as client:
        response = client.post("/ingest", json={"filename": "good.pdf"})
        job = client.get(f"/status/{response.json()['job_id']}").json()
        health = client.get("/health").json()

    assert response.status_code == 202
    body = response.json()
    assert HEX32.fullmatch(body["job_id"])
    assert body == {
        "job_id": body["job_id"],
        "kind": "ingest",
        "target": "good.pdf",
        "status": "queued",
        "already_running": False,
    }
    assert (job["kind"], job["target"], job["status"], job["error_message"]) == (
        "ingest", "good.pdf", "succeeded", None
    )
    summary = job["summary"]
    assert summary["files"] == [{"filename": "good.pdf", "outcome": "done", "attempts": 1}]
    assert summary["outcomes"] == {"done": 1, "needs_ocr": 0, "failed": 0, "skipped": 0}
    assert (summary["moved"], summary["left_in_inbox"], summary["move_failures"]) == (1, 0, 0)
    assert names(settings.processed_dir) == ["good.pdf"]
    assert names(settings.inbox_dir) == ["other.pdf"]  # only the named file was touched
    assert health["documents"]["done"] == 1 and health["collection"]["chunk_count"] > 0
    assert health["active_job"] is None


def test_duplicate_copy_of_a_done_document_is_skipped_and_moved(tmp_path: Path) -> None:
    app, settings = make_app(tmp_path)
    original = write_pdf(settings.inbox_dir / "a.pdf", TEXT)
    shutil.copyfile(original, settings.inbox_dir / "a copy.pdf")

    with TestClient(app) as client:
        client.post("/ingest", json={"filename": "a.pdf"})
        response = client.post("/ingest", json={"filename": "a copy.pdf"})
        job = client.get(f"/status/{response.json()['job_id']}").json()

    assert response.status_code == 202
    assert job["status"] == "succeeded"
    assert job["summary"]["files"] == [
        {"filename": "a copy.pdf", "outcome": "skipped", "attempts": 1,
         "skip_reason": "already ingested"}
    ]
    assert job["summary"]["moved"] == 1
    assert names(settings.processed_dir) == ["a copy.pdf", "a.pdf"]
    assert names(settings.inbox_dir) == []


def test_invalid_pdf_job_succeeds_file_failed_and_stays(tmp_path: Path) -> None:
    app, settings = make_app(tmp_path)
    (settings.inbox_dir / "broken.pdf").write_bytes(b"%PDF-1.7 this is not really a pdf")

    with TestClient(app) as client:
        response = client.post("/ingest", json={"filename": "broken.pdf"})
        job = client.get(f"/status/{response.json()['job_id']}").json()

    assert job["status"] == "succeeded" and job["error_message"] is None
    (entry,) = job["summary"]["files"]
    assert (entry["filename"], entry["outcome"], entry["attempts"]) == ("broken.pdf", "failed", 1)
    assert str(tmp_path) not in entry["error"] and ":\\" not in entry["error"]
    assert job["summary"]["left_in_inbox"] == 1 and job["summary"]["moved"] == 0
    assert names(settings.inbox_dir) == ["broken.pdf"]


def test_runner_gets_the_inbox_path_and_folders(tmp_path: Path) -> None:
    runner = FakeSingleFileRunner()
    app, settings = make_app(tmp_path, runner)
    (settings.inbox_dir / "a.pdf").write_bytes(b"%PDF-1.7")

    with TestClient(app) as client:
        response = client.post("/ingest", json={"filename": "a.pdf"})
        context = app.state.pipeline.ingest_context

    assert response.status_code == 202
    assert runner.calls == [
        (
            context,
            settings.inbox_dir / "a.pdf",
            settings.processed_dir,
            settings.failed_dir,
            settings.needs_ocr_dir,
            settings.inbox_dir,
        )
    ]


def test_runner_crash_fails_the_ingest_job(tmp_path: Path) -> None:
    runner = FakeSingleFileRunner(error=RuntimeError("disk exploded"))
    app, settings = make_app(tmp_path, runner)
    (settings.inbox_dir / "a.pdf").write_bytes(b"%PDF-1.7")

    with TestClient(app) as client:
        first = client.post("/ingest", json={"filename": "a.pdf"}).json()
        job = client.get(f"/status/{first['job_id']}").json()
        again = client.post("/ingest", json={"filename": "a.pdf"})

    assert (job["status"], job["error_message"], job["summary"]) == (
        "failed", "RuntimeError: disk exploded", None
    )
    assert again.status_code == 202 and again.json()["job_id"] != first["job_id"]


# --- 422 / 404: no job is created -----------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["", " a.pdf", "../a.pdf", "..\\a.pdf", "sub/a.pdf", "C:\\Windows\\win.ini", "C:a.pdf",
     "a.pdf:stream", "notes.txt", "CON.pdf", "a\u0000.pdf", "x" * 256],
    ids=["empty", "space", "dotdot", "dotdot-bs", "sub", "absolute", "drive", "stream", "txt",
         "CON", "nul", "long"],
)
def test_bad_names_are_422_and_create_no_job(tmp_path: Path, name: str) -> None:
    runner = FakeSingleFileRunner()
    app, settings = make_app(tmp_path, runner)
    (settings.inbox_dir / "a.pdf").write_bytes(b"%PDF-1.7")

    with TestClient(app) as client:
        response = client.post("/ingest", json={"filename": name})
        rows = job_rows(settings)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, str) and detail.startswith("filename must")
    assert str(tmp_path) not in detail
    assert rows == 0 and runner.calls == []


@pytest.mark.parametrize(
    "payload", [{}, {"filename": 123}, {"filename": None}, {"name": "a.pdf"}],
    ids=["no-field", "int", "null", "wrong-field"],
)
def test_malformed_bodies_are_422_and_create_no_job(tmp_path: Path, payload: Any) -> None:
    runner = FakeSingleFileRunner()
    app, settings = make_app(tmp_path, runner)

    with TestClient(app) as client:
        response = client.post("/ingest", json=payload)
        rows = job_rows(settings)

    assert response.status_code == 422
    assert rows == 0 and runner.calls == []


@pytest.mark.parametrize("setup", ["missing", "directory"])
def test_not_in_inbox_is_404_and_creates_no_job(tmp_path: Path, setup: str) -> None:
    runner = FakeSingleFileRunner()
    app, settings = make_app(tmp_path, runner)
    if setup == "directory":
        (settings.inbox_dir / "d.pdf").mkdir()

    with TestClient(app) as client:
        response = client.post("/ingest", json={"filename": "d.pdf"})
        rows = job_rows(settings)

    assert response.status_code == 404
    assert response.json() == {"detail": FILE_NOT_FOUND}
    assert rows == 0 and runner.calls == []


def test_file_in_processed_is_not_found(tmp_path: Path) -> None:
    app, settings = make_app(tmp_path)
    settings.processed_dir.mkdir(parents=True)
    (settings.processed_dir / "done.pdf").write_bytes(b"%PDF-1.7")

    with TestClient(app) as client:
        response = client.post("/ingest", json={"filename": "done.pdf"})

    assert response.status_code == 404


# --- 409: another job is active --------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "target"), [(JobKind.SCAN, None), (JobKind.INGEST, "other.pdf")],
    ids=["scan", "ingest"],
)
@pytest.mark.parametrize("state", [JobStatus.QUEUED, JobStatus.RUNNING])
def test_active_job_gives_409_with_that_job(
    tmp_path: Path, kind: JobKind, target: str | None, state: JobStatus
) -> None:
    runner = FakeSingleFileRunner()
    app, settings = make_app(tmp_path, runner)
    (settings.inbox_dir / "a.pdf").write_bytes(b"%PDF-1.7")

    with TestClient(app) as client:
        store = store_of(app)
        active, _ = store.create_job_or_get_active(kind, target)
        if state is JobStatus.RUNNING:
            store.mark_job_running(active.job_id)
        response = client.post("/ingest", json={"filename": "a.pdf"})
        rows = job_rows(settings)
        after = store.get_job(active.job_id)

    assert response.status_code == 409
    assert response.json() == {
        "detail": JOB_CONFLICT,
        "active_job": {
            "job_id": active.job_id, "kind": kind.value, "status": state.value, "target": target
        },
    }
    assert rows == 1 and runner.calls == []
    assert after is not None and after.status is state


def test_scan_while_ingest_is_active_returns_the_ingest_job(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path, FakeSingleFileRunner())

    with TestClient(app) as client:
        active, _ = store_of(app).create_job_or_get_active(JobKind.INGEST, "x.pdf")
        response = client.post("/ingest/scan")

    assert response.status_code == 202
    assert response.json() == {
        "job_id": active.job_id, "kind": "ingest", "status": "queued", "already_running": True
    }
