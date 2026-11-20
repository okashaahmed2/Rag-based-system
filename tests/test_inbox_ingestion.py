"""Tests for rag_pipeline.inbox_ingestion (tmp folders, tmp SQLite + Chroma, stub model)."""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from rag_pipeline import inbox as inbox_module
from rag_pipeline import inbox_ingestion
from rag_pipeline.extraction import compute_file_doc_id
from rag_pipeline.inbox_ingestion import (
    DestinationFolders,
    InboxRunReport,
    destination_for,
    ingest_inbox,
    safe_move,
)
from rag_pipeline.ingestion import IngestContext, IngestOutcome
from rag_pipeline.status_db import DocumentRecord, DocumentStatus, StatusStore
from rag_pipeline.vector_store import count_document_chunks, open_collection

DIM = 8
MODEL_NAME = "stub/model"
MAX_ATTEMPTS = 3


# --- stub model and PDFs ------------------------------------------------------------


class StubTokenizer:
    def __call__(
        self, texts: list[str], *, add_special_tokens: bool, truncation: bool
    ) -> dict[str, list[list[int]]]:
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


class StubModel:
    def __init__(self) -> None:
        self.tokenizer = StubTokenizer()
        self.max_seq_length = 256

    def get_embedding_dimension(self) -> int:
        return DIM

    def encode(self, sentences: str | list[str], **kwargs: Any) -> np.ndarray:
        texts = [sentences] if isinstance(sentences, str) else sentences
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.frombuffer(digest[:DIM], dtype=np.uint8).astype(np.float64) + 1
            rows.append(vector / np.linalg.norm(vector))
        out = np.stack(rows)
        return out[0] if isinstance(sentences, str) else out


def write_pdf(path: Path, page_texts: list[str | None], **save_kwargs: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as pdf:
        for text in page_texts:
            page = pdf.new_page()
            if text is not None:
                page.insert_text((72, 60), text, fontsize=9)
        pdf.save(path, **save_kwargs)
    return path


def body(page: int, lines: int = 10) -> str:
    return "\n".join(f"Page {page} line {i} covers topic {page * 100 + i}." for i in range(lines))


# --- fixtures ------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_chroma_system_cache() -> Iterator[None]:
    yield
    SharedSystemClient.clear_system_cache()


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    result = {name: tmp_path / name for name in ("inbox", "processed", "failed", "needs_ocr")}
    result["inbox"].mkdir()
    return result


@pytest.fixture
def context(tmp_path: Path) -> IngestContext:
    store = StatusStore(tmp_path / "state" / "status.db")
    store.initialize()
    return IngestContext(
        store=store,
        collection=open_collection(tmp_path / "chroma", "chunks", MODEL_NAME, DIM),
        model=StubModel(),
        embedding_model_name=MODEL_NAME,
        embedding_batch_size=4,
        chunk_size=200,
        chunk_overlap=40,
        max_ingest_attempts=MAX_ATTEMPTS,
    )


def run(context: IngestContext, dirs: dict[str, Path]) -> InboxRunReport:
    return ingest_inbox(
        context, dirs["inbox"], dirs["processed"], dirs["failed"], dirs["needs_ocr"]
    )


def names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def content_multiset(dirs: dict[str, Path]) -> Counter[str]:
    """SHA-256 of every file in the four folders (recursive), counted."""
    found: Counter[str] = Counter()
    for folder in dirs.values():
        if folder.exists():
            for path in folder.rglob("*"):
                if path.is_file():
                    found[hashlib.sha256(path.read_bytes()).hexdigest()] += 1
    return found


def make_mixed_inbox(inbox: Path) -> dict[str, Path]:
    invalid = inbox / "invalid.pdf"
    invalid.write_bytes(b"not a pdf at all")
    files = {
        "text": write_pdf(inbox / "lecture.pdf", [body(1), body(2), body(3)]),
        "scanned": write_pdf(inbox / "scanned.pdf", [None, None]),
        "invalid": invalid,
        "password": write_pdf(
            inbox / "locked.pdf",
            ["Secret"],
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="user",
        ),
    }
    files["copy"] = inbox / "lecture_copy_renamed.pdf"
    shutil.copyfile(files["text"], files["copy"])
    return files


class IngestSpy:
    def __init__(self, target: Any) -> None:
        self.target = target
        self.paths: list[str] = []

    def __call__(self, path: Path, context: IngestContext) -> Any:
        self.paths.append(path.name)
        return self.target(path, context)


@pytest.fixture
def ingest_spy(monkeypatch: pytest.MonkeyPatch) -> IngestSpy:
    spy = IngestSpy(inbox_ingestion.ingest_document)
    monkeypatch.setattr(inbox_ingestion, "ingest_document", spy)
    return spy


# --- destination_for -------------------------------------------------------------------


def _record(status: DocumentStatus, attempts: int) -> DocumentRecord:
    now = datetime(2026, 11, 20, tzinfo=timezone.utc)
    return DocumentRecord("d" * 64, "f.pdf", status, None, attempts, None, now, now + timedelta(1))


FOLDERS = DestinationFolders(Path("P"), Path("F"), Path("O"))


@pytest.mark.parametrize(
    ("status", "attempts", "expected"),
    [
        (DocumentStatus.PENDING, 0, None),
        (DocumentStatus.PROCESSING, 1, None),
        (DocumentStatus.PROCESSING, 5, None),
        (DocumentStatus.DONE, 1, Path("P")),
        (DocumentStatus.DONE, 3, Path("P")),
        (DocumentStatus.NEEDS_OCR, 1, Path("O")),
        (DocumentStatus.FAILED, 1, None),
        (DocumentStatus.FAILED, 2, None),
        (DocumentStatus.FAILED, 3, Path("F")),
        (DocumentStatus.FAILED, 4, Path("F")),
    ],
)
def test_destination_for_matrix(
    status: DocumentStatus, attempts: int, expected: Path | None
) -> None:
    assert destination_for(_record(status, attempts), MAX_ATTEMPTS, FOLDERS) == expected


def test_destination_for_unregistered_is_none() -> None:
    assert destination_for(None, MAX_ATTEMPTS, FOLDERS) is None


def test_destination_for_respects_max_attempts() -> None:
    record = _record(DocumentStatus.FAILED, 1)

    assert destination_for(record, 1, FOLDERS) == Path("F")
    assert destination_for(record, 2, FOLDERS) is None


# --- safe_move ---------------------------------------------------------------------------


def test_safe_move_free_name(tmp_path: Path) -> None:
    source = tmp_path / "a.pdf"
    source.write_bytes(b"new")

    moved = safe_move(source, tmp_path / "dest" / "nested")

    assert moved == tmp_path / "dest" / "nested" / "a.pdf"
    assert moved.read_bytes() == b"new" and not source.exists()


def test_safe_move_collisions_get_suffixes_and_keep_existing(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "a.pdf").write_bytes(b"old0")
    first = tmp_path / "a.pdf"
    first.write_bytes(b"new1")

    assert safe_move(first, dest) == dest / "a (1).pdf"

    second = tmp_path / "a.pdf"
    second.write_bytes(b"new2")
    assert safe_move(second, dest) == dest / "a (2).pdf"
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == {
        "a.pdf": b"old0",
        "a (1).pdf": b"new1",
        "a (2).pdf": b"new2",
    }


def test_safe_move_file_exists_race_tries_next_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    source = tmp_path / "a.pdf"
    source.write_bytes(b"mine")
    original = inbox_ingestion._rename_no_overwrite
    calls: list[str] = []

    def racing(src: Path, target: Path) -> None:
        calls.append(target.name)
        if len(calls) == 1:  # another process grabs the name after the exists() check
            target.write_bytes(b"theirs")
            raise FileExistsError(target)
        original(src, target)

    monkeypatch.setattr(inbox_ingestion, "_rename_no_overwrite", racing)

    moved = safe_move(source, dest)

    assert calls == ["a.pdf", "a (1).pdf"]
    assert moved == dest / "a (1).pdf"
    assert (dest / "a.pdf").read_bytes() == b"theirs"
    assert moved.read_bytes() == b"mine"


def test_safe_move_real_rename_refuses_to_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "a.pdf"
    target = tmp_path / "b.pdf"
    source.write_bytes(b"src")
    target.write_bytes(b"dst")

    with pytest.raises(FileExistsError):
        inbox_ingestion._rename_no_overwrite(source, target)

    assert (source.read_bytes(), target.read_bytes()) == (b"src", b"dst")


def test_safe_move_other_error_propagates_and_keeps_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "a.pdf"
    source.write_bytes(b"x")

    def locked(src: Path, target: Path) -> None:
        raise PermissionError("file is open in a viewer")

    monkeypatch.setattr(inbox_ingestion, "_rename_no_overwrite", locked)

    with pytest.raises(PermissionError):
        safe_move(source, tmp_path / "dest")

    assert source.read_bytes() == b"x"


# --- ingest_inbox ------------------------------------------------------------------------


def test_mixed_inbox_run_one(context: IngestContext, dirs: dict[str, Path]) -> None:
    files = make_mixed_inbox(dirs["inbox"])
    text_id = compute_file_doc_id(files["text"])
    before = content_multiset(dirs)

    report = run(context, dirs)

    assert names(dirs["processed"]) == ["lecture.pdf", "lecture_copy_renamed.pdf"]
    assert names(dirs["needs_ocr"]) == ["scanned.pdf"]
    assert names(dirs["inbox"]) == ["invalid.pdf", "locked.pdf"]
    assert names(dirs["failed"]) == []
    outcomes = {r.filename: r.outcome for r in report.ingest_results}
    assert outcomes == {
        "invalid.pdf": IngestOutcome.FAILED,
        "lecture.pdf": IngestOutcome.DONE,
        "locked.pdf": IngestOutcome.FAILED,
        "scanned.pdf": IngestOutcome.NEEDS_OCR,
    }
    assert {left.path.name: left.reason for left in report.left_in_inbox} == {
        "invalid.pdf": "failed, attempts 1 of 3",
        "locked.pdf": "failed, attempts 1 of 3",
    }
    record = context.store.get_document(text_id)
    assert record is not None and record.chunk_count is not None
    assert count_document_chunks(context.collection, text_id) == record.chunk_count
    assert context.collection.count() == record.chunk_count
    assert content_multiset(dirs) == before


def test_failures_move_to_failed_after_last_attempt(
    context: IngestContext, dirs: dict[str, Path], ingest_spy: IngestSpy
) -> None:
    make_mixed_inbox(dirs["inbox"])
    before = content_multiset(dirs)

    for expected_attempts in (1, 2):
        run(context, dirs)
        assert names(dirs["inbox"]) == ["invalid.pdf", "locked.pdf"]
        for doc in context.store.list_documents(DocumentStatus.FAILED):
            assert doc.attempts == expected_attempts
        assert content_multiset(dirs) == before

    run(context, dirs)  # third attempt
    assert names(dirs["inbox"]) == []
    assert names(dirs["failed"]) == ["invalid.pdf", "locked.pdf"]
    assert content_multiset(dirs) == before
    calls_after_three = list(ingest_spy.paths)

    report = run(context, dirs)  # fourth run: nothing to do

    assert ingest_spy.paths == calls_after_three
    assert report.ingest_results == () and report.moves == ()
    assert names(dirs["inbox"]) == []
    assert calls_after_three.count("invalid.pdf") == 3
    assert calls_after_three.count("lecture.pdf") == 1
    assert "lecture_copy_renamed.pdf" not in calls_after_three
    assert content_multiset(dirs) == before


def test_duplicate_in_same_scan_is_ingested_once(
    context: IngestContext, dirs: dict[str, Path], ingest_spy: IngestSpy
) -> None:
    original = write_pdf(dirs["inbox"] / "a.pdf", [body(1)])
    shutil.copyfile(original, dirs["inbox"] / "b_copy.pdf")

    run(context, dirs)

    assert ingest_spy.paths == ["a.pdf"]
    assert names(dirs["processed"]) == ["a.pdf", "b_copy.pdf"]
    assert len(context.store.list_documents()) == 1


def test_later_copy_of_done_doc_moves_without_ingestion(
    context: IngestContext, dirs: dict[str, Path], ingest_spy: IngestSpy, tmp_path: Path
) -> None:
    source = write_pdf(tmp_path / "src" / "a.pdf", [body(1)])
    shutil.copyfile(source, dirs["inbox"] / "a.pdf")
    run(context, dirs)
    chunks = context.collection.count()
    shutil.copyfile(source, dirs["inbox"] / "a_again.pdf")

    report = run(context, dirs)

    assert ingest_spy.paths == ["a.pdf"]
    assert report.ingest_results == ()
    assert [m.destination.name for m in report.moves] == ["a_again.pdf"]
    assert names(dirs["processed"]) == ["a.pdf", "a_again.pdf"]
    assert context.collection.count() == chunks


def test_move_failure_stays_and_heals_next_run(
    context: IngestContext,
    dirs: dict[str, Path],
    ingest_spy: IngestSpy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_pdf(dirs["inbox"] / "a.pdf", [body(1)])
    doc_id = compute_file_doc_id(path)
    original = inbox_ingestion._rename_no_overwrite
    calls = {"n": 0}

    def locked_once(src: Path, target: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("The process cannot access the file (viewer open)")
        original(src, target)

    monkeypatch.setattr(inbox_ingestion, "_rename_no_overwrite", locked_once)

    first = run(context, dirs)

    assert names(dirs["inbox"]) == ["a.pdf"]
    assert [f.path.name for f in first.move_failures] == ["a.pdf"]
    assert "PermissionError" in first.move_failures[0].error
    record = context.store.get_document(doc_id)
    assert record is not None and record.status is DocumentStatus.DONE

    second = run(context, dirs)

    assert names(dirs["inbox"]) == [] and names(dirs["processed"]) == ["a.pdf"]
    assert second.ingest_results == () and second.move_failures == ()
    assert ingest_spy.paths == ["a.pdf"]


def test_processing_record_file_stays(context: IngestContext, dirs: dict[str, Path]) -> None:
    path = write_pdf(dirs["inbox"] / "a.pdf", [body(1)])
    doc_id = compute_file_doc_id(path)
    context.store.register_document(doc_id, "a.pdf")
    context.store.mark_processing(doc_id)

    report = run(context, dirs)

    assert report.ingest_results == ()
    assert [(left.path.name, left.reason) for left in report.left_in_inbox] == [
        ("a.pdf", "processing")
    ]
    assert names(dirs["inbox"]) == ["a.pdf"]


def test_scan_error_file_stays_and_is_reported(
    context: IngestContext, dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    good = write_pdf(dirs["inbox"] / "good.pdf", [body(1)])
    bad = write_pdf(dirs["inbox"] / "unreadable.pdf", [body(2)])
    real_hash = inbox_module.compute_file_doc_id

    def flaky_hash(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.name == "unreadable.pdf":
            raise PermissionError("locked by another program")
        return real_hash(path, *args, **kwargs)

    monkeypatch.setattr(inbox_module, "compute_file_doc_id", flaky_hash)

    report = run(context, dirs)

    assert [e.path.name for e in report.scan_errors] == ["unreadable.pdf"]
    assert "locked by another program" in report.scan_errors[0].reason
    assert names(dirs["inbox"]) == ["unreadable.pdf"]
    assert names(dirs["processed"]) == ["good.pdf"]
    assert bad.exists() and not good.exists()


def test_name_collision_in_processed_keeps_old_file(
    context: IngestContext, dirs: dict[str, Path]
) -> None:
    dirs["processed"].mkdir()
    (dirs["processed"] / "a.pdf").write_bytes(b"an older, different file")
    write_pdf(dirs["inbox"] / "a.pdf", [body(1)])
    before = content_multiset(dirs)

    report = run(context, dirs)

    assert [m.destination.name for m in report.moves] == ["a (1).pdf"]
    assert (dirs["processed"] / "a.pdf").read_bytes() == b"an older, different file"
    assert content_multiset(dirs) == before


def test_empty_inbox_gives_empty_report(context: IngestContext, dirs: dict[str, Path]) -> None:
    report = run(context, dirs)

    assert report == InboxRunReport((), (), (), (), ())


def test_moved_files_are_not_rescanned(
    context: IngestContext, dirs: dict[str, Path], ingest_spy: IngestSpy
) -> None:
    dirs["processed"] = dirs["inbox"] / "processed"  # destination inside the inbox
    write_pdf(dirs["inbox"] / "a.pdf", [body(1)])
    run(context, dirs)
    assert names(dirs["processed"]) == ["a.pdf"]

    report = run(context, dirs)

    assert report.ingest_results == () and report.moves == ()
    assert ingest_spy.paths == ["a.pdf"]
    assert names(dirs["processed"]) == ["a.pdf"]


def test_destination_equal_to_inbox_is_rejected(
    context: IngestContext, dirs: dict[str, Path]
) -> None:
    with pytest.raises(ValueError, match="must not be the inbox"):
        ingest_inbox(context, dirs["inbox"], dirs["inbox"], dirs["failed"], dirs["needs_ocr"])
