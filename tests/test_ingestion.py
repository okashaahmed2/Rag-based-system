"""Tests for rag_pipeline.ingestion (tmp_path SQLite + Chroma, stub model, real PDFs)."""

from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Iterator
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pytest
from chromadb.api.models.Collection import Collection
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.errors import InternalError

from rag_pipeline import ingestion
from rag_pipeline.config import Settings
from rag_pipeline.embeddings import EmbeddingError, embed_chunks
from rag_pipeline.extraction import compute_file_doc_id
from rag_pipeline.ingestion import (
    SKIP_ALREADY_INGESTED,
    SKIP_ATTEMPTS_EXHAUSTED,
    SKIP_IN_PROGRESS,
    IngestContext,
    IngestionSetupError,
    IngestOutcome,
    build_ingest_context,
    ingest_document,
)
from rag_pipeline.preparation import DocumentChunk, prepare_document
from rag_pipeline.status_db import DocumentStatus, StatusStore, StatusStoreError
from rag_pipeline.vector_store import (
    count_document_chunks,
    open_collection,
    replace_document_chunks,
)

DIM = 8
MODEL_NAME = "stub/model"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40


# --- stub embedding model ------------------------------------------------------


class StubTokenizer:
    def __call__(
        self, texts: list[str], *, add_special_tokens: bool, truncation: bool
    ) -> dict[str, list[list[int]]]:
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


class StubModel:
    """Deterministic stand-in for SentenceTransformer (normalized float vectors)."""

    def __init__(self, dimension: int = DIM, fail: bool = False) -> None:
        self.tokenizer = StubTokenizer()
        self.max_seq_length = 256
        self._dimension = dimension
        self.fail = fail
        self.encode_calls = 0

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, sentences: str | list[str], **kwargs: Any) -> np.ndarray:
        self.encode_calls += 1
        if self.fail:
            raise RuntimeError("stub encoder exploded")
        texts = [sentences] if isinstance(sentences, str) else sentences
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.frombuffer(digest[: self._dimension], dtype=np.uint8).astype(np.float64) + 1
            rows.append(vector / np.linalg.norm(vector))
        out = np.stack(rows)
        return out[0] if isinstance(sentences, str) else out


# --- PDFs ------------------------------------------------------------------------


def write_pdf(path: Path, page_texts: list[str | None], **save_kwargs: Any) -> Path:
    """One page per entry; ``None`` makes a blank page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as pdf:
        for text in page_texts:
            page = pdf.new_page()
            if text is not None:
                page.insert_text((72, 60), text, fontsize=9)
        pdf.save(path, **save_kwargs)
    return path


def body(page: int, lines: int = 12) -> str:
    return "\n".join(
        f"Page {page} line {i} describes topic {page * 100 + i} in plain words." for i in range(lines)
    )


# --- fixtures ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_chroma_system_cache() -> Iterator[None]:
    yield
    SharedSystemClient.clear_system_cache()


@pytest.fixture
def model() -> StubModel:
    return StubModel()


@pytest.fixture
def store(tmp_path: Path) -> StatusStore:
    status_store = StatusStore(tmp_path / "state" / "status.db")
    status_store.initialize()
    return status_store


@pytest.fixture
def collection(tmp_path: Path) -> Collection:
    return open_collection(tmp_path / "chroma", "chunks", MODEL_NAME, DIM)


@pytest.fixture
def context(store: StatusStore, collection: Collection, model: StubModel) -> IngestContext:
    return IngestContext(
        store=store,
        collection=collection,
        model=model,
        embedding_model_name=MODEL_NAME,
        embedding_batch_size=4,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        max_ingest_attempts=3,
    )


@pytest.fixture
def pdfs(tmp_path: Path) -> dict[str, Path]:
    folder = tmp_path / "pdfs"
    folder.mkdir()
    invalid = folder / "invalid.pdf"
    invalid.write_bytes(b"this is not a pdf")
    return {
        "text": write_pdf(folder / "lecture.pdf", [body(1), body(2), body(3)]),
        "other": write_pdf(folder / "other.pdf", [body(7), body(8)]),
        "blank": write_pdf(folder / "scanned.pdf", [None, None]),
        "invalid": invalid,
        "password": write_pdf(
            folder / "locked.pdf",
            ["Secret"],
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="user",
        ),
        "only_footers": write_pdf(
            folder / "footers.pdf", ["Page 1 of 3", "Page 2 of 3", "Page 3 of 3"]
        ),
    }


def expected_chunks(path: Path) -> int:
    settings = Settings.model_construct(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return len(prepare_document(path, settings).chunks)


def stored_ids(collection: Collection, doc_id: str) -> list[str]:
    return sorted(collection.get(where={"doc_id": doc_id}, include=[])["ids"])


class Spy:
    """Wraps a module function and counts calls."""

    def __init__(self, target: Any) -> None:
        self.target = target
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.target(*args, **kwargs)


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> dict[str, Spy]:
    result = {}
    for name in ("prepare_document", "embed_chunks", "replace_document_chunks"):
        spy = Spy(getattr(ingestion, name))
        monkeypatch.setattr(ingestion, name, spy)
        result[name] = spy
    return result


def seed_stale_chunks(collection: Collection, doc_id: str, n: int) -> None:
    """Store ``n`` stale chunks for ``doc_id`` directly in Chroma."""
    chunks = [
        DocumentChunk(
            chunk_id=f"{doc_id}:{i}",
            doc_id=doc_id,
            filename="stale.pdf",
            page_number=1,
            chunk_index=i,
            text=f"stale chunk {i}",
            char_start=0,
            char_end=5,
        )
        for i in range(n)
    ]
    embeddings = embed_chunks(chunks, StubModel(), 8, model_name=MODEL_NAME)
    replace_document_chunks(collection, doc_id, chunks, embeddings)


# --- happy paths ------------------------------------------------------------------


def test_happy_path_done(context: IngestContext, pdfs: dict[str, Path]) -> None:
    path = pdfs["text"]
    doc_id = compute_file_doc_id(path)

    result = ingest_document(path, context)

    expected = expected_chunks(path)
    assert expected > 3
    assert (result.outcome, result.doc_id, result.chunk_count, result.attempts) == (
        IngestOutcome.DONE,
        doc_id,
        expected,
        1,
    )
    assert (result.error_message, result.skip_reason) == (None, None)
    assert result.filename == "lecture.pdf" and result.path == path
    assert result.duration_seconds >= 0
    record = context.store.get_document(doc_id)
    assert record is not None
    assert (record.status, record.chunk_count, record.attempts) == (DocumentStatus.DONE, expected, 1)
    assert count_document_chunks(context.collection, doc_id) == expected
    assert stored_ids(context.collection, doc_id) == sorted(f"{doc_id}:{i}" for i in range(expected))


def test_needs_ocr_clears_old_chunks(context: IngestContext, pdfs: dict[str, Path]) -> None:
    path = pdfs["blank"]
    doc_id = compute_file_doc_id(path)
    seed_stale_chunks(context.collection, doc_id, 3)

    result = ingest_document(path, context)

    assert (result.outcome, result.chunk_count, result.attempts) == (IngestOutcome.NEEDS_OCR, 0, 1)
    record = context.store.get_document(doc_id)
    assert record is not None and (record.status, record.chunk_count) == (DocumentStatus.NEEDS_OCR, 0)
    assert count_document_chunks(context.collection, doc_id) == 0


@pytest.mark.parametrize("kind", ["invalid", "password"])
def test_unreadable_pdf_fails(context: IngestContext, pdfs: dict[str, Path], kind: str) -> None:
    path = pdfs[kind]
    doc_id = compute_file_doc_id(path)

    result = ingest_document(path, context)

    assert result.outcome is IngestOutcome.FAILED
    assert result.error_message is not None and "PdfExtractionError" in result.error_message
    assert (result.attempts, result.chunk_count, result.doc_id) == (1, None, doc_id)
    record = context.store.get_document(doc_id)
    assert record is not None and record.status is DocumentStatus.FAILED
    assert record.error_message == result.error_message
    assert count_document_chunks(context.collection, doc_id) == 0


def test_text_without_chunks_fails(context: IngestContext, pdfs: dict[str, Path]) -> None:
    result = ingest_document(pdfs["only_footers"], context)

    assert result.outcome is IngestOutcome.FAILED
    assert result.error_message is not None
    assert result.error_message.startswith("IngestionError: no chunks produced")


# --- skips ------------------------------------------------------------------------


def test_second_call_on_done_doc_is_skipped_without_work(
    context: IngestContext, pdfs: dict[str, Path], spies: dict[str, Spy]
) -> None:
    first = ingest_document(pdfs["text"], context)
    calls_before = {name: spy.calls for name, spy in spies.items()}

    second = ingest_document(pdfs["text"], context)

    assert (second.outcome, second.skip_reason) == (IngestOutcome.SKIPPED, SKIP_ALREADY_INGESTED)
    assert (second.chunk_count, second.attempts) == (first.chunk_count, 1)
    assert {name: spy.calls for name, spy in spies.items()} == calls_before


def test_skipped_needs_ocr_doc(context: IngestContext, pdfs: dict[str, Path]) -> None:
    ingest_document(pdfs["blank"], context)

    again = ingest_document(pdfs["blank"], context)

    assert (again.outcome, again.skip_reason) == (IngestOutcome.SKIPPED, SKIP_ALREADY_INGESTED)


def test_renamed_copy_of_done_doc_is_skipped(
    context: IngestContext, pdfs: dict[str, Path], tmp_path: Path
) -> None:
    first = ingest_document(pdfs["text"], context)
    copy = tmp_path / "copy_renamed.pdf"
    shutil.copyfile(pdfs["text"], copy)
    total = context.collection.count()

    result = ingest_document(copy, context)

    assert (result.outcome, result.skip_reason, result.doc_id) == (
        IngestOutcome.SKIPPED,
        SKIP_ALREADY_INGESTED,
        first.doc_id,
    )
    assert result.filename == "copy_renamed.pdf"
    assert context.collection.count() == total
    assert len(context.store.list_documents()) == 1


def test_processing_doc_is_skipped(context: IngestContext, pdfs: dict[str, Path]) -> None:
    doc_id = compute_file_doc_id(pdfs["text"])
    context.store.register_document(doc_id, "lecture.pdf")
    context.store.mark_processing(doc_id)

    result = ingest_document(pdfs["text"], context)

    assert (result.outcome, result.skip_reason, result.attempts) == (
        IngestOutcome.SKIPPED,
        SKIP_IN_PROGRESS,
        1,
    )


# --- retries ------------------------------------------------------------------------


def test_retry_after_failure_succeeds(
    context: IngestContext, pdfs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ingestion.prepare_document
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient read problem")
        return original(*args, **kwargs)

    monkeypatch.setattr(ingestion, "prepare_document", flaky)

    first = ingest_document(pdfs["text"], context)
    second = ingest_document(pdfs["text"], context)

    assert first.outcome is IngestOutcome.FAILED
    assert first.error_message == "RuntimeError: transient read problem"
    assert (second.outcome, second.attempts) == (IngestOutcome.DONE, 2)
    record = context.store.get_document(second.doc_id or "")
    assert record is not None
    assert (record.status, record.attempts, record.error_message) == (DocumentStatus.DONE, 2, None)


def test_attempts_exhausted_is_skipped_without_work(
    context: IngestContext, pdfs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def always_fail(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        raise RuntimeError(f"broken {calls['n']}")

    monkeypatch.setattr(ingestion, "prepare_document", always_fail)
    results = [ingest_document(pdfs["text"], context) for _ in range(context.max_ingest_attempts)]

    final = ingest_document(pdfs["text"], context)

    assert [r.outcome for r in results] == [IngestOutcome.FAILED] * 3
    assert [r.attempts for r in results] == [1, 2, 3]
    assert (final.outcome, final.skip_reason, final.attempts) == (
        IngestOutcome.SKIPPED,
        SKIP_ATTEMPTS_EXHAUSTED,
        3,
    )
    assert final.error_message == "RuntimeError: broken 3"
    assert calls["n"] == 3


# --- fault injection --------------------------------------------------------------


def test_upsert_failure_after_delete_then_retry_leaves_no_orphans(
    context: IngestContext, pdfs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    done_a = ingest_document(pdfs["text"], context)
    a_ids = stored_ids(context.collection, done_a.doc_id or "")
    path_b = pdfs["other"]
    doc_b = compute_file_doc_id(path_b)
    expected_b = expected_chunks(path_b)
    seed_stale_chunks(context.collection, doc_b, expected_b + 4)
    original_upsert = Collection.upsert

    def failing_upsert(self: Collection, *args: Any, **kwargs: Any) -> Any:
        raise InternalError("disk full (injected)")

    monkeypatch.setattr(Collection, "upsert", failing_upsert)
    failed = ingest_document(path_b, context)

    assert failed.outcome is IngestOutcome.FAILED
    assert failed.error_message is not None
    assert failed.error_message.startswith("VectorStoreError:")
    assert "disk full (injected)" in failed.error_message
    assert count_document_chunks(context.collection, doc_b) == 0  # delete ran, upsert failed
    record = context.store.get_document(doc_b)
    assert record is not None and record.status is DocumentStatus.FAILED

    monkeypatch.setattr(Collection, "upsert", original_upsert)
    retried = ingest_document(path_b, context)

    assert (retried.outcome, retried.chunk_count, retried.attempts) == (
        IngestOutcome.DONE,
        expected_b,
        2,
    )
    assert stored_ids(context.collection, doc_b) == sorted(f"{doc_b}:{i}" for i in range(expected_b))
    assert stored_ids(context.collection, done_a.doc_id or "") == a_ids


def test_keyboard_interrupt_propagates_and_is_recovered(
    context: IngestContext, pdfs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = compute_file_doc_id(pdfs["text"])
    original = ingestion.replace_document_chunks

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(ingestion, "replace_document_chunks", interrupted)
    with pytest.raises(KeyboardInterrupt):
        ingest_document(pdfs["text"], context)

    record = context.store.get_document(doc_id)
    assert record is not None and (record.status, record.attempts) == (DocumentStatus.PROCESSING, 1)
    assert context.store.recover_interrupted() == 1
    record = context.store.get_document(doc_id)
    assert record is not None and record.status is DocumentStatus.FAILED

    monkeypatch.setattr(ingestion, "replace_document_chunks", original)
    result = ingest_document(pdfs["text"], context)

    assert (result.outcome, result.attempts, result.chunk_count) == (
        IngestOutcome.DONE,
        2,
        expected_chunks(pdfs["text"]),
    )


def test_embedding_failure_keeps_existing_chunks(
    context: IngestContext, pdfs: dict[str, Path], model: StubModel
) -> None:
    path = pdfs["text"]
    doc_id = compute_file_doc_id(path)
    seed_stale_chunks(context.collection, doc_id, 3)
    before = context.collection.get(where={"doc_id": doc_id}, include=["documents"])
    model.fail = True

    result = ingest_document(path, context)

    assert result.outcome is IngestOutcome.FAILED
    assert result.error_message == "RuntimeError: stub encoder exploded"
    after = context.collection.get(where={"doc_id": doc_id}, include=["documents"])
    assert sorted(after["ids"]) == sorted(before["ids"])
    assert sorted(after["documents"]) == sorted(before["documents"])


def test_doc_id_mismatch_fails(
    context: IngestContext, pdfs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ingestion.prepare_document

    def changed(*args: Any, **kwargs: Any) -> Any:
        return dc_replace(original(*args, **kwargs), doc_id="f" * 64)

    monkeypatch.setattr(ingestion, "prepare_document", changed)

    result = ingest_document(pdfs["text"], context)

    assert result.outcome is IngestOutcome.FAILED
    assert result.error_message is not None
    assert result.error_message.startswith("IngestionError: file changed during processing")
    assert count_document_chunks(context.collection, result.doc_id or "") == 0
    assert count_document_chunks(context.collection, "f" * 64) == 0


def test_hashing_failure_registers_nothing(context: IngestContext, tmp_path: Path) -> None:
    result = ingest_document(tmp_path / "missing.pdf", context)

    assert (result.outcome, result.doc_id, result.attempts) == (IngestOutcome.FAILED, None, None)
    assert result.error_message is not None
    assert result.error_message.startswith("FileNotFoundError:")
    assert context.store.list_documents() == []


@pytest.mark.parametrize("method", ["register_document", "mark_processing", "mark_done", "mark_failed"])
def test_status_store_error_is_reraised(
    context: IngestContext,
    pdfs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    method: str,
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise StatusStoreError("database disk image is malformed (injected)")

    monkeypatch.setattr(context.store, method, broken)
    path = pdfs["invalid"] if method == "mark_failed" else pdfs["text"]

    with caplog.at_level(logging.CRITICAL, logger="rag_pipeline.ingestion"):
        with pytest.raises(StatusStoreError, match="injected"):
            ingest_document(path, context)

    assert any(r.levelno == logging.CRITICAL for r in caplog.records)


def test_logs_summary_without_chunk_text(
    context: IngestContext, pdfs: dict[str, Path], caplog: pytest.LogCaptureFixture
) -> None:
    doc_id = compute_file_doc_id(pdfs["text"])

    with caplog.at_level(logging.INFO, logger="rag_pipeline"):
        ingest_document(pdfs["text"], context)

    summary = [r.getMessage() for r in caplog.records if r.name == "rag_pipeline.ingestion"]
    assert any(m.startswith(f"Ingest {doc_id[:12]} (lecture.pdf): done, chunks=") for m in summary)
    assert "describes topic" not in caplog.text


def _failure_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == "rag_pipeline.ingestion" and r.levelno == logging.ERROR
        and "failed on attempt" in r.getMessage()
    ]


@pytest.mark.parametrize("kind", ["invalid", "password", "only_footers"])
def test_expected_failure_logs_one_line_without_traceback(
    context: IngestContext,
    pdfs: dict[str, Path],
    caplog: pytest.LogCaptureFixture,
    kind: str,
) -> None:
    with caplog.at_level(logging.INFO, logger="rag_pipeline"):
        result = ingest_document(pdfs[kind], context)

    records = _failure_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert not record.exc_info  # exc_info=False is stored as False
    assert "Traceback" not in caplog.text
    message = record.getMessage()
    assert (result.doc_id or "")[:12] in message and pdfs[kind].name in message
    assert result.error_message is not None and message.endswith(result.error_message)


def test_unexpected_failure_logs_traceback(
    context: IngestContext,
    pdfs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr(ingestion, "prepare_document", broken)

    with caplog.at_level(logging.INFO, logger="rag_pipeline"):
        result = ingest_document(pdfs["text"], context)

    records = _failure_records(caplog)
    assert len(records) == 1
    exc_info = records[0].exc_info
    assert exc_info is not None and exc_info[0] is RuntimeError
    assert "Traceback" in caplog.text
    assert result.error_message == "RuntimeError: unexpected bug"


def test_no_file_is_moved_or_changed(context: IngestContext, pdfs: dict[str, Path]) -> None:
    folder = pdfs["text"].parent
    before = {p.name: p.read_bytes() for p in folder.iterdir()}

    for path in pdfs.values():
        ingest_document(path, context)

    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


# --- IngestContext / build_ingest_context -----------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_model_name": " "},
        {"embedding_batch_size": 0},
        {"chunk_size": 0},
        {"chunk_overlap": CHUNK_SIZE},
        {"max_ingest_attempts": 0},
    ],
)
def test_context_rejects_bad_values(context: IngestContext, overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        dc_replace(context, **overrides)


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "chroma_path": tmp_path / "ctx_chroma",
        "status_db_path": tmp_path / "ctx_state" / "status.db",
        "embedding_model_name": MODEL_NAME,
        "embedding_local_files_only": True,
        "max_ingest_attempts": 2,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_build_ingest_context_wires_everything_and_recovers(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    pre = StatusStore(settings.status_db_path)
    pre.initialize()
    pre.register_document("x" * 64, "stuck.pdf")
    pre.mark_processing("x" * 64)
    loader_calls: list[tuple[str, bool]] = []

    def loader(name: str, local_files_only: bool) -> StubModel:
        loader_calls.append((name, local_files_only))
        return StubModel(dimension=12)

    context = build_ingest_context(settings, model_loader=loader)

    assert loader_calls == [(MODEL_NAME, True)]
    assert context.collection.metadata == {"embedding_model": MODEL_NAME, "embedding_dim": 12}
    assert (context.chunk_size, context.chunk_overlap) == (settings.chunk_size, settings.chunk_overlap)
    assert (context.max_ingest_attempts, context.embedding_model_name) == (2, MODEL_NAME)
    record = context.store.get_document("x" * 64)
    assert record is not None and record.status is DocumentStatus.FAILED


def test_build_ingest_context_rejects_mismatched_saved_model(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    open_collection(settings.chroma_path, settings.chroma_collection_name, "older/model", DIM)
    SharedSystemClient.clear_system_cache()

    with pytest.raises(IngestionSetupError, match="Vector store setup failed.*older/model"):
        build_ingest_context(settings, model_loader=lambda name, local: StubModel())


def test_build_ingest_context_reports_model_failure(tmp_path: Path) -> None:
    def loader(name: str, local_files_only: bool) -> StubModel:
        raise EmbeddingError("model not in cache")

    with pytest.raises(IngestionSetupError, match="Embedding model setup failed: model not in cache"):
        build_ingest_context(_settings(tmp_path), model_loader=loader)


def test_build_ingest_context_reports_status_store_failure(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.status_db_path.parent.mkdir(parents=True)
    settings.status_db_path.mkdir()  # a folder where the db file should be

    with pytest.raises(IngestionSetupError, match="Status store setup failed"):
        build_ingest_context(settings, model_loader=lambda name, local: StubModel())
