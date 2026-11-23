"""Tests for rag_pipeline.retrieval (real Chroma + SQLite in tmp_path, stub model)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pytest
from chromadb.api.models.Collection import Collection
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.config import Settings as ChromaSettings

from rag_pipeline import retrieval
from rag_pipeline.config import Settings
from rag_pipeline.embeddings import ChunkEmbeddings, EmbeddingError
from rag_pipeline.preparation import DocumentChunk
from rag_pipeline.retrieval import (
    RetrievalContext,
    RetrievalError,
    build_retrieval_context,
    done_doc_ids,
    retrieve,
)
from rag_pipeline.status_db import DocumentStatus, StatusStore
from rag_pipeline.vector_store import (
    RetrievedChunk,
    open_collection,
    replace_document_chunks,
)

DIM = 4
MODEL_NAME = "stub/model"
QUERY = "what is a decision tree"
QUERY_VECTOR = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

DOC_DONE = "d" * 64
DOC_DONE_2 = "e" * 64
DOC_FAILED = "f" * 64
DOC_OTHER = "0" * 64


class StubQueryModel:
    """Returns QUERY_VECTOR for every query and counts encode calls."""

    def __init__(self, dimension: int = DIM) -> None:
        self._dimension = dimension
        self.encode_calls = 0
        self.max_seq_length = 256
        self.tokenizer = None

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, sentences: str | list[str], **kwargs: Any) -> np.ndarray:
        self.encode_calls += 1
        return QUERY_VECTOR[: self._dimension].copy()


@pytest.fixture(autouse=True)
def clear_chroma_system_cache() -> Iterator[None]:
    yield
    SharedSystemClient.clear_system_cache()


@pytest.fixture
def model() -> StubQueryModel:
    return StubQueryModel()


@pytest.fixture
def store(tmp_path: Path) -> StatusStore:
    status_store = StatusStore(tmp_path / "status.db")
    status_store.initialize()
    return status_store


@pytest.fixture
def collection(tmp_path: Path) -> Collection:
    return open_collection(tmp_path / "chroma", "chunks", MODEL_NAME, DIM)


@pytest.fixture
def context(store: StatusStore, collection: Collection, model: StubQueryModel) -> RetrievalContext:
    return RetrievalContext(store=store, collection=collection, model=model)


def vector_at(similarity: float) -> np.ndarray:
    """Unit vector whose cosine similarity to QUERY_VECTOR is ``similarity``."""
    return np.array([similarity, np.sqrt(1.0 - similarity**2), 0.0, 0.0], dtype=np.float32)


def store_chunks(
    collection: Collection, doc_id: str, similarities: list[float], filename: str = "doc.pdf"
) -> list[DocumentChunk]:
    """Store one chunk per similarity (page = position + 1)."""
    chunks = [
        DocumentChunk(
            chunk_id=f"{doc_id}:{i}",
            doc_id=doc_id,
            filename=filename,
            page_number=i + 1,
            chunk_index=i,
            text=f"SECRET chunk text {doc_id[:4]} number {i}",
            char_start=0,
            char_end=10,
        )
        for i in range(len(similarities))
    ]
    embeddings = ChunkEmbeddings(
        chunk_ids=tuple(c.chunk_id for c in chunks),
        vectors=np.stack([vector_at(s) for s in similarities]),
        model_name=MODEL_NAME,
        dimension=DIM,
        max_seq_length=256,
        token_counts=tuple(5 for _ in chunks),
        over_limit_chunk_ids=(),
    )
    replace_document_chunks(collection, doc_id, chunks, embeddings)
    return chunks


def set_status(store: StatusStore, doc_id: str, status: DocumentStatus, chunks: int = 1) -> None:
    """Register ``doc_id`` and move it to ``status`` through valid transitions."""
    store.register_document(doc_id, f"{doc_id[:4]}.pdf")
    if status is DocumentStatus.PENDING:
        return
    store.mark_processing(doc_id)
    if status is DocumentStatus.DONE:
        store.mark_done(doc_id, chunks)
    elif status is DocumentStatus.FAILED:
        store.mark_failed(doc_id, "boom")
    elif status is DocumentStatus.NEEDS_OCR:
        store.mark_needs_ocr(doc_id)


def settings_for(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "project_root": tmp_path,
        "status_db_path": tmp_path / "status.db",
        "chroma_path": tmp_path / "chroma",
        "embedding_model_name": MODEL_NAME,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


# --- retrieve: done filter -------------------------------------------------------------


def test_done_document_is_returned(
    context: RetrievalContext, store: StatusStore, collection: Collection
) -> None:
    chunks = store_chunks(collection, DOC_DONE, [0.9, 0.5], filename="Lecture07.pdf")
    set_status(store, DOC_DONE, DocumentStatus.DONE, 2)

    results = retrieve(QUERY, context, top_k=5)

    assert [r.chunk_id for r in results] == [chunks[0].chunk_id, chunks[1].chunk_id]
    top = results[0]
    assert isinstance(top, RetrievedChunk)
    assert (top.doc_id, top.filename, top.page_number, top.chunk_index) == (
        DOC_DONE,
        "Lecture07.pdf",
        1,
        0,
    )
    assert top.similarity == pytest.approx(0.9, abs=1e-5)


def test_failed_document_with_leftover_chunks_is_never_returned(
    context: RetrievalContext, store: StatusStore, collection: Collection
) -> None:
    # The failed document's chunk is identical to the query (the nearest one).
    store_chunks(collection, DOC_FAILED, [1.0, 0.99, 0.98])
    set_status(store, DOC_FAILED, DocumentStatus.FAILED)
    store_chunks(collection, DOC_DONE, [0.2])
    set_status(store, DOC_DONE, DocumentStatus.DONE)

    for top_k in (1, 3, 50):
        results = retrieve(QUERY, context, top_k=top_k)
        assert [r.doc_id for r in results] == [DOC_DONE]


@pytest.mark.parametrize(
    "status",
    [
        DocumentStatus.PENDING,
        DocumentStatus.PROCESSING,
        DocumentStatus.FAILED,
        DocumentStatus.NEEDS_OCR,
    ],
)
def test_non_done_statuses_are_excluded(
    context: RetrievalContext,
    store: StatusStore,
    collection: Collection,
    status: DocumentStatus,
) -> None:
    store_chunks(collection, DOC_OTHER, [1.0, 0.95])
    set_status(store, DOC_OTHER, status)
    store_chunks(collection, DOC_DONE, [0.1, 0.3])
    set_status(store, DOC_DONE, DocumentStatus.DONE, 2)

    results = retrieve(QUERY, context, top_k=10)

    assert {r.doc_id for r in results} == {DOC_DONE}
    assert len(results) == 2


def test_chunks_of_unregistered_document_are_excluded(
    context: RetrievalContext, store: StatusStore, collection: Collection
) -> None:
    store_chunks(collection, DOC_OTHER, [1.0])
    store_chunks(collection, DOC_DONE, [0.4])
    set_status(store, DOC_DONE, DocumentStatus.DONE)

    assert [r.doc_id for r in retrieve(QUERY, context, top_k=5)] == [DOC_DONE]


def test_no_done_documents_returns_empty_without_querying(
    context: RetrievalContext,
    store: StatusStore,
    collection: Collection,
    model: StubQueryModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_chunks(collection, DOC_FAILED, [1.0])
    set_status(store, DOC_FAILED, DocumentStatus.FAILED)
    calls: list[Any] = []
    monkeypatch.setattr(retrieval, "query_by_embedding", lambda *a, **k: calls.append(a))

    assert retrieve(QUERY, context, top_k=5) == []
    assert calls == []
    assert model.encode_calls == 0


def test_empty_store_returns_empty(context: RetrievalContext) -> None:
    assert retrieve(QUERY, context, top_k=3) == []


def test_top_k_larger_than_matches_returns_all_matches(
    context: RetrievalContext, store: StatusStore, collection: Collection
) -> None:
    store_chunks(collection, DOC_DONE, [0.1, 0.2, 0.3])
    set_status(store, DOC_DONE, DocumentStatus.DONE, 3)
    store_chunks(collection, DOC_FAILED, [0.5, 0.6, 0.7, 0.8, 0.9])
    set_status(store, DOC_FAILED, DocumentStatus.FAILED)

    results = retrieve(QUERY, context, top_k=50)

    assert len(results) == 3
    assert {r.doc_id for r in results} == {DOC_DONE}


def test_results_are_ordered_by_distance_across_documents(
    context: RetrievalContext, store: StatusStore, collection: Collection
) -> None:
    store_chunks(collection, DOC_DONE, [0.3, 0.9, 0.1])
    set_status(store, DOC_DONE, DocumentStatus.DONE, 3)
    store_chunks(collection, DOC_DONE_2, [0.7, 0.5])
    set_status(store, DOC_DONE_2, DocumentStatus.DONE, 2)

    results = retrieve(QUERY, context, top_k=4)

    assert [round(r.similarity, 4) for r in results] == [0.9, 0.7, 0.5, 0.3]
    distances = [r.distance for r in results]
    assert distances == sorted(distances)
    assert [(r.doc_id, r.page_number) for r in results] == [
        (DOC_DONE, 2),
        (DOC_DONE_2, 1),
        (DOC_DONE_2, 2),
        (DOC_DONE, 1),
    ]


# --- retrieve: validation and errors ------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\n\t", None, 5])
def test_invalid_query_raises(context: RetrievalContext, query: Any) -> None:
    with pytest.raises(RetrievalError, match="query must be a non-empty string"):
        retrieve(query, context, top_k=3)


@pytest.mark.parametrize("top_k", [0, -1, True, 2.0, "3", None])
def test_invalid_top_k_raises(context: RetrievalContext, top_k: Any) -> None:
    with pytest.raises(RetrievalError, match="top_k must be an int >= 1"):
        retrieve(QUERY, context, top_k=top_k)


def test_status_store_failure_raises_retrieval_error(
    context: RetrievalContext, store: StatusStore
) -> None:
    store.path.unlink()

    with pytest.raises(RetrievalError, match="Could not read done documents"):
        retrieve(QUERY, context, top_k=3)
    assert not store.path.exists()


def test_wrong_query_dimension_raises_retrieval_error(
    store: StatusStore, collection: Collection
) -> None:
    store_chunks(collection, DOC_DONE, [0.5])
    set_status(store, DOC_DONE, DocumentStatus.DONE)
    context = RetrievalContext(store=store, collection=collection, model=StubQueryModel(3))

    with pytest.raises(RetrievalError, match="Retrieval failed"):
        retrieve(QUERY, context, top_k=3)


def test_leaked_non_done_chunk_raises(
    context: RetrievalContext, store: StatusStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_status(store, DOC_DONE, DocumentStatus.DONE)
    leaked = RetrievedChunk(f"{DOC_FAILED}:0", DOC_FAILED, "x.pdf", 1, 0, "t", 0.1, 0.9)
    monkeypatch.setattr(retrieval, "query_by_embedding", lambda *a, **k: [leaked])

    with pytest.raises(RetrievalError, match="non-done documents"):
        retrieve(QUERY, context, top_k=3)


def test_filter_passed_to_vector_store_is_exactly_the_done_ids(
    context: RetrievalContext, store: StatusStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_status(store, DOC_DONE, DocumentStatus.DONE)
    set_status(store, DOC_DONE_2, DocumentStatus.DONE)
    set_status(store, DOC_FAILED, DocumentStatus.FAILED)
    seen: dict[str, Any] = {}

    def spy(collection: Any, vector: Any, top_k: int, *, doc_ids: Any) -> list[Any]:
        seen.update(top_k=top_k, doc_ids=set(doc_ids))
        return []

    monkeypatch.setattr(retrieval, "query_by_embedding", spy)

    assert retrieve(QUERY, context, top_k=7) == []
    assert seen == {"top_k": 7, "doc_ids": {DOC_DONE, DOC_DONE_2}}


def test_done_doc_ids_reads_only_done(store: StatusStore) -> None:
    set_status(store, DOC_DONE, DocumentStatus.DONE)
    set_status(store, DOC_FAILED, DocumentStatus.FAILED)
    set_status(store, DOC_OTHER, DocumentStatus.NEEDS_OCR)

    assert done_doc_ids(store) == {DOC_DONE}


def test_logs_contain_no_chunk_or_query_text(
    context: RetrievalContext,
    store: StatusStore,
    collection: Collection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store_chunks(collection, DOC_DONE, [0.9, 0.8])
    set_status(store, DOC_DONE, DocumentStatus.DONE, 2)
    caplog.set_level(logging.DEBUG)

    results = retrieve(QUERY, context, top_k=2)
    with pytest.raises(RetrievalError):
        retrieve(QUERY, context, top_k=0)

    assert results
    assert "Retrieved 2 chunk(s)" in caplog.text
    assert "SECRET" not in caplog.text
    assert QUERY not in caplog.text


# --- build_retrieval_context ---------------------------------------------------------------


def _prepare_stores(tmp_path: Path) -> None:
    StatusStore(tmp_path / "status.db").initialize()
    open_collection(tmp_path / "chroma", "chunks", MODEL_NAME, DIM)
    SharedSystemClient.clear_system_cache()


def test_build_context_never_recovers_interrupted_documents(
    tmp_path: Path, model: StubQueryModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_stores(tmp_path)
    StatusStore(tmp_path / "status.db").register_document(DOC_OTHER, "x.pdf")
    StatusStore(tmp_path / "status.db").mark_processing(DOC_OTHER)
    calls: list[Any] = []
    monkeypatch.setattr(StatusStore, "recover_interrupted", lambda self: calls.append(self) or 0)

    context = build_retrieval_context(settings_for(tmp_path), model_loader=lambda name, local: model)
    retrieve(QUERY, context, top_k=3)

    assert calls == []
    record = context.store.get_document(DOC_OTHER)
    assert record is not None and record.status is DocumentStatus.PROCESSING


def test_build_context_retrieves_from_existing_stores(
    tmp_path: Path, model: StubQueryModel
) -> None:
    _prepare_stores(tmp_path)
    collection = open_collection(tmp_path / "chroma", "chunks", MODEL_NAME, DIM)
    store_chunks(collection, DOC_DONE, [0.6])
    set_status(StatusStore(tmp_path / "status.db"), DOC_DONE, DocumentStatus.DONE)
    loaded: list[tuple[str, bool]] = []

    def loader(name: str, local_only: bool) -> StubQueryModel:
        loaded.append((name, local_only))
        return model

    context = build_retrieval_context(
        settings_for(tmp_path, embedding_local_files_only=True), model_loader=loader
    )

    assert loaded == [(MODEL_NAME, True)]
    assert [r.doc_id for r in retrieve(QUERY, context, top_k=3)] == [DOC_DONE]


def test_build_context_missing_status_db_raises_and_creates_nothing(
    tmp_path: Path, model: StubQueryModel
) -> None:
    open_collection(tmp_path / "chroma", "chunks", MODEL_NAME, DIM)

    with pytest.raises(RetrievalError, match="Status database not found"):
        build_retrieval_context(settings_for(tmp_path), model_loader=lambda name, local: model)
    assert not (tmp_path / "status.db").exists()


def test_build_context_missing_vector_store_raises_and_creates_nothing(
    tmp_path: Path, model: StubQueryModel
) -> None:
    StatusStore(tmp_path / "status.db").initialize()

    with pytest.raises(RetrievalError, match="Vector store not found"):
        build_retrieval_context(settings_for(tmp_path), model_loader=lambda name, local: model)
    assert not (tmp_path / "chroma").exists()


def _collection_names(path: Path) -> list[str]:
    """Collection names seen by a fresh client (system cache cleared first)."""
    SharedSystemClient.clear_system_cache()
    client = chromadb.PersistentClient(
        path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
    )
    return sorted(c.name for c in client.list_collections())


def test_build_context_missing_collection_raises_and_creates_nothing(
    tmp_path: Path, model: StubQueryModel
) -> None:
    StatusStore(tmp_path / "status.db").initialize()
    open_collection(tmp_path / "chroma", "other", MODEL_NAME, DIM)
    SharedSystemClient.clear_system_cache()

    with pytest.raises(
        RetrievalError, match="Collection 'chunks' not found .*; run ingestion first"
    ):
        build_retrieval_context(settings_for(tmp_path), model_loader=lambda name, local: model)

    assert _collection_names(tmp_path / "chroma") == ["other"]


def test_build_context_empty_chroma_folder_raises_and_creates_no_collection(
    tmp_path: Path, model: StubQueryModel
) -> None:
    StatusStore(tmp_path / "status.db").initialize()
    (tmp_path / "chroma").mkdir()

    with pytest.raises(RetrievalError, match="Collection 'chunks' not found"):
        build_retrieval_context(settings_for(tmp_path), model_loader=lambda name, local: model)

    assert _collection_names(tmp_path / "chroma") == []


def test_build_context_model_mismatch_raises(tmp_path: Path) -> None:
    _prepare_stores(tmp_path)

    with pytest.raises(RetrievalError, match="Vector store setup failed"):
        build_retrieval_context(
            settings_for(tmp_path), model_loader=lambda name, local: StubQueryModel(dimension=8)
        )


def test_build_context_model_load_failure_raises(tmp_path: Path) -> None:
    _prepare_stores(tmp_path)

    def failing_loader(name: str, local_only: bool) -> StubQueryModel:
        raise EmbeddingError("no such model")

    with pytest.raises(RetrievalError, match="Embedding model setup failed"):
        build_retrieval_context(settings_for(tmp_path), model_loader=failing_loader)
