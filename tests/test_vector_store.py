"""Tests for rag_pipeline.vector_store (real ChromaDB in tmp_path, fake vectors)."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pytest
from chromadb.api.models.Collection import Collection
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.config import Settings as ChromaSettings

from rag_pipeline.embeddings import ChunkEmbeddings
from rag_pipeline.preparation import DocumentChunk
from rag_pipeline.vector_store import (
    DISTANCE_SPACE,
    RetrievedChunk,
    VectorStoreError,
    count_document_chunks,
    open_collection,
    query_by_embedding,
    replace_document_chunks,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
NAME = "chunks"
DOC_A = "a" * 64
DOC_B = "b" * 64


@pytest.fixture(autouse=True)
def clear_chroma_system_cache() -> Iterator[None]:
    """Drop Chroma's per-path in-process system cache after every test."""
    yield
    SharedSystemClient.clear_system_cache()


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "store"


@pytest.fixture
def collection(store_path: Path) -> Collection:
    return open_collection(store_path, NAME, MODEL, DIM)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


def unit_vectors(rng: np.random.Generator, n: int, dim: int = DIM) -> np.ndarray:
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def make_chunks(
    doc_id: str, n: int, filename: str = "doc.pdf", label: str = "v1"
) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"{doc_id}:{i}",
            doc_id=doc_id,
            filename=filename,
            page_number=i // 2 + 1,
            chunk_index=i,
            text=f"{label} text of chunk {i} in {filename}",
            char_start=i * 10,
            char_end=i * 10 + 9,
        )
        for i in range(n)
    ]


def make_embeddings(
    chunks: list[DocumentChunk],
    rng: np.random.Generator,
    *,
    model: str = MODEL,
    dim: int = DIM,
    vectors: np.ndarray | None = None,
    chunk_ids: tuple[str, ...] | None = None,
) -> ChunkEmbeddings:
    vecs = vectors if vectors is not None else unit_vectors(rng, len(chunks), dim)
    return ChunkEmbeddings(
        chunk_ids=chunk_ids if chunk_ids is not None else tuple(c.chunk_id for c in chunks),
        vectors=vecs,
        model_name=model,
        dimension=dim,
        max_seq_length=256,
        token_counts=tuple(10 for _ in chunks),
        over_limit_chunk_ids=(),
    )


def store_doc(
    collection: Collection,
    rng: np.random.Generator,
    doc_id: str,
    n: int,
    **chunk_kwargs: Any,
) -> tuple[list[DocumentChunk], ChunkEmbeddings]:
    chunks = make_chunks(doc_id, n, **chunk_kwargs)
    embeddings = make_embeddings(chunks, rng)
    assert replace_document_chunks(collection, doc_id, chunks, embeddings) == n
    return chunks, embeddings


def snapshot(collection: Collection, doc_id: str) -> dict[str, Any]:
    """Everything stored for one document, sorted by id."""
    got = collection.get(
        where={"doc_id": doc_id}, include=["documents", "metadatas", "embeddings"]
    )
    order = np.argsort(got["ids"])
    return {
        "ids": [got["ids"][i] for i in order],
        "documents": [got["documents"][i] for i in order],
        "metadatas": [got["metadatas"][i] for i in order],
        "embeddings": [np.asarray(got["embeddings"][i]).tolist() for i in order],
    }


def raw_client(path: Path) -> Any:
    return chromadb.PersistentClient(
        path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
    )


# --- open_collection -----------------------------------------------------------


def test_new_collection_saves_metadata_and_cosine(store_path: Path) -> None:
    collection = open_collection(store_path, NAME, MODEL, DIM)

    assert store_path.is_dir()
    assert collection.name == NAME
    assert collection.metadata == {"embedding_model": MODEL, "embedding_dim": DIM}
    with pytest.warns(DeprecationWarning, match="legacy embedding function config"):
        configuration = collection.configuration
    assert configuration["hnsw"]["space"] == DISTANCE_SPACE == "cosine"
    assert collection.count() == 0


def test_collection_has_no_embedding_function(collection: Collection) -> None:
    # Chroma's own query_texts path reads the configuration (legacy warning).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="legacy embedding function config", category=DeprecationWarning
        )
        with pytest.raises(ValueError, match="embedding function"):
            collection.query(query_texts=["probe"], n_results=1)


def test_open_creates_missing_nested_directory(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "store"

    open_collection(path, NAME, MODEL, DIM)

    assert path.is_dir()


def test_reopen_with_same_settings_keeps_data(
    store_path: Path, rng: np.random.Generator
) -> None:
    first = open_collection(store_path, NAME, MODEL, DIM)
    store_doc(first, rng, DOC_A, 3)
    SharedSystemClient.clear_system_cache()

    reopened = open_collection(store_path, NAME, MODEL, DIM)

    assert count_document_chunks(reopened, DOC_A) == 3
    assert reopened.metadata == {"embedding_model": MODEL, "embedding_dim": DIM}


def test_persistence_in_fresh_process(store_path: Path, rng: np.random.Generator) -> None:
    collection = open_collection(store_path, NAME, MODEL, DIM)
    store_doc(collection, rng, DOC_A, 4)
    probe = textwrap.dedent(
        """
        import json, sys
        from pathlib import Path
        from rag_pipeline.vector_store import count_document_chunks, open_collection
        col = open_collection(Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]))
        print(json.dumps({"count": count_document_chunks(col, sys.argv[5]),
                          "metadata": col.metadata}))
        """
    )

    proc = subprocess.run(
        [sys.executable, "-c", probe, str(store_path), NAME, MODEL, str(DIM), DOC_A],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {
        "count": 4,
        "metadata": {"embedding_model": MODEL, "embedding_dim": DIM},
    }


@pytest.mark.parametrize(
    ("model", "dim", "message"),
    [
        ("other/model", DIM, "embedding_model: saved 'sentence-transformers/all-MiniLM-L6-v2', expected 'other/model'"),
        (MODEL, 768, "embedding_dim: saved 384, expected 768"),
    ],
)
def test_reopen_with_different_settings_raises_and_keeps_metadata(
    store_path: Path, rng: np.random.Generator, model: str, dim: int, message: str
) -> None:
    collection = open_collection(store_path, NAME, MODEL, DIM)
    store_doc(collection, rng, DOC_A, 2)
    SharedSystemClient.clear_system_cache()

    with pytest.raises(VectorStoreError, match=message):
        open_collection(store_path, NAME, model, dim)

    SharedSystemClient.clear_system_cache()
    reopened = open_collection(store_path, NAME, MODEL, DIM)
    assert reopened.metadata == {"embedding_model": MODEL, "embedding_dim": DIM}
    assert count_document_chunks(reopened, DOC_A) == 2


def test_existing_collection_without_metadata_raises(store_path: Path) -> None:
    raw_client(store_path).create_collection(
        NAME, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
    )

    with pytest.raises(VectorStoreError, match="embedding_model: saved None"):
        open_collection(store_path, NAME, MODEL, DIM)


def test_existing_collection_with_other_space_raises(store_path: Path) -> None:
    raw_client(store_path).create_collection(
        NAME,
        configuration={"hnsw": {"space": "l2"}},
        metadata={"embedding_model": MODEL, "embedding_dim": DIM},
        embedding_function=None,
    )

    with pytest.raises(VectorStoreError, match="space: saved 'l2', expected 'cosine'"):
        open_collection(store_path, NAME, MODEL, DIM)


@pytest.mark.parametrize(("model", "dim"), [("", DIM), ("  ", DIM), (MODEL, 0)])
def test_open_rejects_invalid_arguments(store_path: Path, model: str, dim: int) -> None:
    with pytest.raises(VectorStoreError):
        open_collection(store_path, NAME, model, dim)


def test_open_wraps_invalid_collection_name(store_path: Path) -> None:
    with pytest.raises(VectorStoreError, match="Could not open collection"):
        open_collection(store_path, "x", MODEL, DIM)


def test_open_does_not_leak_the_legacy_warning(
    store_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    open_collection(store_path, NAME, MODEL, DIM)
    SharedSystemClient.clear_system_cache()
    open_collection(store_path, NAME, MODEL, DIM)

    assert not [w for w in recwarn if "legacy embedding function" in str(w.message)]


# --- replace_document_chunks ---------------------------------------------------


def test_replace_five_then_three_leaves_exactly_three(
    collection: Collection, rng: np.random.Generator
) -> None:
    store_doc(collection, rng, DOC_A, 5, label="old")
    store_doc(collection, rng, DOC_B, 2, filename="other.pdf")
    b_before = snapshot(collection, DOC_B)

    new_chunks, _ = store_doc(collection, rng, DOC_A, 3, label="new")

    stored = snapshot(collection, DOC_A)
    assert stored["ids"] == [f"{DOC_A}:{i}" for i in range(3)]
    assert stored["documents"] == [c.text for c in new_chunks]
    assert all(doc.startswith("new ") for doc in stored["documents"])
    assert snapshot(collection, DOC_B) == b_before
    assert collection.count() == 5


def test_replace_with_zero_chunks_deletes_document(
    collection: Collection, rng: np.random.Generator
) -> None:
    store_doc(collection, rng, DOC_A, 4)
    store_doc(collection, rng, DOC_B, 2)

    assert replace_document_chunks(collection, DOC_A, [], None) == 0

    assert count_document_chunks(collection, DOC_A) == 0
    assert count_document_chunks(collection, DOC_B) == 2


def test_replace_with_zero_chunks_and_empty_embeddings(
    collection: Collection, rng: np.random.Generator
) -> None:
    store_doc(collection, rng, DOC_A, 2)
    empty = make_embeddings([], rng, vectors=np.empty((0, DIM), dtype=np.float32))

    assert replace_document_chunks(collection, DOC_A, [], empty) == 0
    assert count_document_chunks(collection, DOC_A) == 0


def test_replace_is_idempotent(collection: Collection, rng: np.random.Generator) -> None:
    chunks = make_chunks(DOC_A, 4)
    embeddings = make_embeddings(chunks, rng)

    replace_document_chunks(collection, DOC_A, chunks, embeddings)
    first = snapshot(collection, DOC_A)
    replace_document_chunks(collection, DOC_A, chunks, embeddings)

    assert snapshot(collection, DOC_A) == first
    assert collection.count() == 4


def test_metadata_round_trip_equals_to_metadata(
    collection: Collection, rng: np.random.Generator
) -> None:
    chunks, embeddings = store_doc(collection, rng, DOC_A, 3, filename="paper.pdf")

    stored = snapshot(collection, DOC_A)

    assert stored["metadatas"] == [c.to_metadata() for c in chunks]
    assert stored["documents"] == [c.text for c in chunks]
    np.testing.assert_allclose(stored["embeddings"], embeddings.vectors, atol=1e-7)


def _bad_inputs(rng: np.random.Generator) -> dict[str, tuple[list[DocumentChunk], Any, dict[str, Any]]]:
    good = make_chunks(DOC_A, 3, label="bad")
    wrong_doc = good[:2] + [make_chunks(DOC_B, 3)[2]]
    nan_vectors = unit_vectors(rng, 3)
    nan_vectors[1, 5] = np.nan
    duplicate = good[:2] + [good[1]]
    return {
        "chunk of another doc": (wrong_doc, make_embeddings(wrong_doc, rng), {}),
        "fewer vectors than chunks": (
            good,
            make_embeddings(good, rng, vectors=unit_vectors(rng, 2)),
            {},
        ),
        "dimension mismatch": (
            good,
            make_embeddings(good, rng, dim=383, vectors=unit_vectors(rng, 3, 383)),
            {},
        ),
        "model mismatch": (good, make_embeddings(good, rng, model="other/model"), {}),
        "chunk id order mismatch": (
            good,
            make_embeddings(good, rng, chunk_ids=tuple(c.chunk_id for c in reversed(good))),
            {},
        ),
        "missing embeddings": (good, None, {}),
        "duplicate chunk ids": (duplicate, make_embeddings(duplicate, rng), {}),
        "NaN in vectors": (good, make_embeddings(good, rng, vectors=nan_vectors), {}),
        "max_batch_size zero": (good, make_embeddings(good, rng), {"max_batch_size": 0}),
    }


@pytest.mark.parametrize(
    "case",
    [
        "chunk of another doc",
        "fewer vectors than chunks",
        "dimension mismatch",
        "model mismatch",
        "chunk id order mismatch",
        "missing embeddings",
        "duplicate chunk ids",
        "NaN in vectors",
        "max_batch_size zero",
    ],
)
def test_invalid_replace_raises_before_delete_and_keeps_store(
    case: str,
    collection: Collection,
    rng: np.random.Generator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_doc(collection, rng, DOC_A, 5, label="kept")
    store_doc(collection, rng, DOC_B, 2)
    before_a, before_b = snapshot(collection, DOC_A), snapshot(collection, DOC_B)
    chunks, embeddings, kwargs = _bad_inputs(rng)[case]
    calls: list[str] = []
    original_delete, original_upsert = Collection.delete, Collection.upsert

    def spy_delete(self: Collection, *args: Any, **kw: Any) -> Any:
        calls.append("delete")
        return original_delete(self, *args, **kw)

    def spy_upsert(self: Collection, *args: Any, **kw: Any) -> Any:
        calls.append("upsert")
        return original_upsert(self, *args, **kw)

    monkeypatch.setattr(Collection, "delete", spy_delete)
    monkeypatch.setattr(Collection, "upsert", spy_upsert)

    with pytest.raises(VectorStoreError):
        replace_document_chunks(collection, DOC_A, chunks, embeddings, **kwargs)

    assert calls == []
    assert snapshot(collection, DOC_A) == before_a
    assert snapshot(collection, DOC_B) == before_b


def test_batching_uses_several_upserts(
    collection: Collection, rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_sizes: list[int] = []
    original_upsert = Collection.upsert

    def spy_upsert(self: Collection, *args: Any, **kw: Any) -> Any:
        batch_sizes.append(len(kw["ids"]))
        return original_upsert(self, *args, **kw)

    monkeypatch.setattr(Collection, "upsert", spy_upsert)
    chunks = make_chunks(DOC_A, 5)

    stored = replace_document_chunks(
        collection, DOC_A, chunks, make_embeddings(chunks, rng), max_batch_size=2
    )

    assert stored == 5
    assert batch_sizes == [2, 2, 1]
    assert snapshot(collection, DOC_A)["ids"] == sorted(c.chunk_id for c in chunks)


def test_batch_size_is_capped_by_client_limit(
    collection: Collection, rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_sizes: list[int] = []
    original_upsert = Collection.upsert

    def spy_upsert(self: Collection, *args: Any, **kw: Any) -> Any:
        batch_sizes.append(len(kw["ids"]))
        return original_upsert(self, *args, **kw)

    monkeypatch.setattr(Collection, "upsert", spy_upsert)
    monkeypatch.setattr(type(collection._client), "get_max_batch_size", lambda self: 3)
    chunks = make_chunks(DOC_A, 5)

    replace_document_chunks(
        collection, DOC_A, chunks, make_embeddings(chunks, rng), max_batch_size=100
    )

    assert batch_sizes == [3, 2]


def test_replace_logs_counts_but_no_chunk_text(
    collection: Collection, rng: np.random.Generator, caplog: pytest.LogCaptureFixture
) -> None:
    chunks = make_chunks(DOC_A, 2, filename="secret.pdf", label="CONFIDENTIAL")

    with caplog.at_level(logging.INFO, logger="rag_pipeline.vector_store"):
        replace_document_chunks(collection, DOC_A, chunks, make_embeddings(chunks, rng))

    assert "Stored 2 chunk(s)" in caplog.text
    assert DOC_A[:12] in caplog.text
    assert "secret.pdf" in caplog.text
    assert "CONFIDENTIAL" not in caplog.text


def test_count_document_chunks_unknown_doc_is_zero(collection: Collection) -> None:
    assert count_document_chunks(collection, "f" * 64) == 0


# --- query_by_embedding --------------------------------------------------------


def test_query_orders_by_distance_and_similarity_is_dot(
    collection: Collection, rng: np.random.Generator
) -> None:
    _, emb_a = store_doc(collection, rng, DOC_A, 5)
    _, emb_b = store_doc(collection, rng, DOC_B, 2)
    vectors = {
        cid: vec
        for e in (emb_a, emb_b)
        for cid, vec in zip(e.chunk_ids, e.vectors)
    }
    query = unit_vectors(rng, 1)[0]

    results = query_by_embedding(collection, query, top_k=4)

    assert len(results) == 4
    assert all(isinstance(r, RetrievedChunk) for r in results)
    distances = [r.distance for r in results]
    assert distances == sorted(distances)
    for r in results:
        assert r.similarity == pytest.approx(1.0 - r.distance, abs=1e-12)
        assert r.similarity == pytest.approx(float(np.dot(query, vectors[r.chunk_id])), abs=1e-5)
        assert type(r.page_number) is int and type(r.chunk_index) is int
    expected_top = sorted(vectors, key=lambda cid: -float(np.dot(query, vectors[cid])))[:4]
    assert [r.chunk_id for r in results] == expected_top


def test_query_returns_stored_fields(collection: Collection, rng: np.random.Generator) -> None:
    chunks, embeddings = store_doc(collection, rng, DOC_A, 3, filename="paper.pdf")

    top = query_by_embedding(collection, embeddings.vectors[1], top_k=1)[0]

    assert top.chunk_id == chunks[1].chunk_id
    assert (top.doc_id, top.filename, top.page_number, top.chunk_index, top.text) == (
        DOC_A,
        "paper.pdf",
        chunks[1].page_number,
        1,
        chunks[1].text,
    )
    assert top.similarity == pytest.approx(1.0, abs=1e-5)


def test_query_top_k_larger_than_collection_returns_all(
    collection: Collection, rng: np.random.Generator
) -> None:
    store_doc(collection, rng, DOC_A, 3)

    results = query_by_embedding(collection, unit_vectors(rng, 1)[0], top_k=50)

    assert len(results) == 3


def test_query_empty_collection_returns_empty_list(
    collection: Collection, rng: np.random.Generator
) -> None:
    assert query_by_embedding(collection, unit_vectors(rng, 1)[0], top_k=5) == []


@pytest.mark.parametrize("shape", [(383,), (385,), (1, DIM), ()])
def test_query_wrong_vector_shape_raises(
    collection: Collection, rng: np.random.Generator, shape: tuple[int, ...]
) -> None:
    store_doc(collection, rng, DOC_A, 2)

    with pytest.raises(VectorStoreError, match="does not match collection dimension"):
        query_by_embedding(collection, np.zeros(shape, dtype=np.float32), top_k=1)


@pytest.mark.parametrize("top_k", [0, -1, True, 2.0])
def test_query_invalid_top_k_raises(
    collection: Collection, rng: np.random.Generator, top_k: Any
) -> None:
    with pytest.raises(VectorStoreError, match="top_k"):
        query_by_embedding(collection, unit_vectors(rng, 1)[0], top_k=top_k)


def test_query_non_finite_vector_raises(collection: Collection) -> None:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[0] = np.inf

    with pytest.raises(VectorStoreError, match="NaN or inf"):
        query_by_embedding(collection, vector, top_k=1)
