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
    CollectionNotFoundError,
    PageChunk,
    RetrievedChunk,
    VectorStoreError,
    count_document_chunks,
    get_page_chunks,
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


def test_open_without_create_missing_folder_raises_and_creates_nothing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "absent" / "store"

    with pytest.raises(CollectionNotFoundError, match="folder .* not found"):
        open_collection(path, NAME, MODEL, DIM, create=False)
    assert not (tmp_path / "absent").exists()


def test_open_without_create_missing_collection_raises_and_creates_nothing(
    store_path: Path,
) -> None:
    raw_client(store_path).create_collection("other", embedding_function=None)
    SharedSystemClient.clear_system_cache()

    with pytest.raises(CollectionNotFoundError, match=f"Collection '{NAME}' not found"):
        open_collection(store_path, NAME, MODEL, DIM, create=False)

    SharedSystemClient.clear_system_cache()
    assert [c.name for c in raw_client(store_path).list_collections()] == ["other"]


def test_open_without_create_opens_existing_collection(
    store_path: Path, rng: np.random.Generator
) -> None:
    store_doc(open_collection(store_path, NAME, MODEL, DIM), rng, DOC_A, 2)
    SharedSystemClient.clear_system_cache()

    reopened = open_collection(store_path, NAME, MODEL, DIM, create=False)

    assert count_document_chunks(reopened, DOC_A) == 2


@pytest.mark.parametrize("model, dim", [("other/model", DIM), (MODEL, DIM + 1)])
def test_open_without_create_still_checks_saved_settings(
    store_path: Path, model: str, dim: int
) -> None:
    open_collection(store_path, NAME, MODEL, DIM)
    SharedSystemClient.clear_system_cache()

    with pytest.raises(VectorStoreError, match="does not match the current settings") as info:
        open_collection(store_path, NAME, model, dim, create=False)
    assert not isinstance(info.value, CollectionNotFoundError)


def test_open_without_create_still_checks_space(store_path: Path) -> None:
    raw_client(store_path).create_collection(
        NAME,
        configuration={"hnsw": {"space": "l2"}},
        metadata={"embedding_model": MODEL, "embedding_dim": DIM},
        embedding_function=None,
    )
    SharedSystemClient.clear_system_cache()

    with pytest.raises(VectorStoreError, match="space"):
        open_collection(store_path, NAME, MODEL, DIM, create=False)


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


def test_query_doc_ids_filter_keeps_only_listed_documents(
    collection: Collection, rng: np.random.Generator
) -> None:
    chunks_a, emb_a = store_doc(collection, rng, DOC_A, 4)
    store_doc(collection, rng, DOC_B, 4)
    query = emb_a.vectors[0]

    only_b = query_by_embedding(collection, query, top_k=2, doc_ids={DOC_B})
    both = query_by_embedding(collection, query, top_k=8, doc_ids={DOC_A, DOC_B})
    unfiltered = query_by_embedding(collection, query, top_k=8)

    assert len(only_b) == 2 and {r.doc_id for r in only_b} == {DOC_B}
    assert [r.chunk_id for r in both] == [r.chunk_id for r in unfiltered]
    assert both[0].chunk_id == chunks_a[0].chunk_id


def test_query_doc_ids_top_k_larger_than_matches_returns_all_matches(
    collection: Collection, rng: np.random.Generator
) -> None:
    store_doc(collection, rng, DOC_A, 3)
    store_doc(collection, rng, DOC_B, 5)

    results = query_by_embedding(collection, unit_vectors(rng, 1)[0], top_k=50, doc_ids={DOC_A})

    assert len(results) == 3 and {r.doc_id for r in results} == {DOC_A}


def test_query_doc_ids_without_matches_returns_empty(
    collection: Collection, rng: np.random.Generator
) -> None:
    store_doc(collection, rng, DOC_A, 2)

    assert query_by_embedding(collection, unit_vectors(rng, 1)[0], top_k=5, doc_ids={"c" * 64}) == []


@pytest.mark.parametrize("doc_ids", [set(), frozenset(), {""}, {DOC_A, 5}])
def test_query_invalid_doc_ids_raise(
    collection: Collection, rng: np.random.Generator, doc_ids: Any
) -> None:
    with pytest.raises(VectorStoreError, match="doc_ids must"):
        query_by_embedding(collection, unit_vectors(rng, 1)[0], top_k=1, doc_ids=doc_ids)


def test_query_non_finite_vector_raises(collection: Collection) -> None:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[0] = np.inf

    with pytest.raises(VectorStoreError, match="NaN or inf"):
        query_by_embedding(collection, vector, top_k=1)


# --------------------------------------------------------------------------- get_page_chunks


#: (chunk_index, page_number) per document: several chunks per page, pages 1..3.
PAGE_LAYOUT = [(0, 1), (1, 1), (2, 2), (3, 2), (4, 2), (5, 3)]


def page_chunks(doc_id: str, filename: str, order: list[int]) -> list[DocumentChunk]:
    """Chunks of PAGE_LAYOUT, in the given (possibly shuffled) chunk_index order."""
    pages = dict(PAGE_LAYOUT)
    return [
        DocumentChunk(
            chunk_id=f"{doc_id}:{i}",
            doc_id=doc_id,
            filename=filename,
            page_number=pages[i],
            chunk_index=i,
            text=f"text {i} of {filename} page {pages[i]}",
            char_start=0,
            char_end=10,
        )
        for i in order
    ]


@pytest.fixture
def paged_collection(collection: Collection, rng: np.random.Generator) -> Collection:
    """DOC_A and DOC_B with PAGE_LAYOUT, upserted out of chunk_index order."""
    for doc_id, filename, order in (
        (DOC_A, "a.pdf", [4, 1, 5, 2, 0, 3]),
        (DOC_B, "b.pdf", [3, 2, 1, 0, 5, 4]),
    ):
        chunks = page_chunks(doc_id, filename, order)
        replace_document_chunks(collection, doc_id, chunks, make_embeddings(chunks, rng))
    return collection


def test_get_page_chunks_returns_only_the_requested_doc_and_page(
    paged_collection: Collection,
) -> None:
    result = get_page_chunks(paged_collection, DOC_A, 2)

    assert [c.chunk_id for c in result] == [f"{DOC_A}:2", f"{DOC_A}:3", f"{DOC_A}:4"]
    assert {(c.doc_id, c.page_number, c.filename) for c in result} == {(DOC_A, 2, "a.pdf")}
    other = get_page_chunks(paged_collection, DOC_B, 2)
    assert {c.doc_id for c in other} == {DOC_B} and len(other) == 3


def test_get_page_chunks_sorts_out_of_order_inserts_by_chunk_index(
    paged_collection: Collection,
) -> None:
    raw = paged_collection.get(
        where={"$and": [{"doc_id": {"$eq": DOC_A}}, {"page_number": {"$eq": 2}}]},
        include=["metadatas"],
    )
    raw_order = [m["chunk_index"] for m in raw["metadatas"] or []]

    result = get_page_chunks(paged_collection, DOC_A, 2)

    # Chroma 1.5.9 returned these rows out of chunk_index order when this test was
    # written (Phase 10); not asserted, because get() order is not part of its API.
    assert sorted(raw_order) == [2, 3, 4]
    assert [c.chunk_index for c in result] == [2, 3, 4]


def test_get_page_chunks_full_record_and_int_metadata_round_trip(
    paged_collection: Collection,
) -> None:
    (chunk,) = get_page_chunks(paged_collection, DOC_B, 3)

    assert chunk == PageChunk(
        chunk_id=f"{DOC_B}:5",
        doc_id=DOC_B,
        filename="b.pdf",
        page_number=3,
        chunk_index=5,
        text="text 5 of b.pdf page 3",
    )
    assert type(chunk.page_number) is int and type(chunk.chunk_index) is int
    assert not hasattr(chunk, "similarity")


@pytest.mark.parametrize(("doc_id", "page"), [(DOC_A, 4), (DOC_A, 99), ("c" * 64, 1)])
def test_get_page_chunks_missing_page_or_unknown_doc_is_empty(
    paged_collection: Collection, doc_id: str, page: int
) -> None:
    assert get_page_chunks(paged_collection, doc_id, page) == []


def test_get_page_chunks_on_empty_collection_is_empty(collection: Collection) -> None:
    assert get_page_chunks(collection, DOC_A, 1) == []


class ExplodingCollection:
    """Stands in for a collection; any Chroma call fails the test."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"Chroma was called: {name}")


@pytest.mark.parametrize(
    ("doc_id", "page", "message"),
    [
        ("", 1, "doc_id must be"),
        (None, 1, "doc_id must be"),
        (5, 1, "doc_id must be"),
        (DOC_A, 0, "page_number must be"),
        (DOC_A, -1, "page_number must be"),
        (DOC_A, True, "page_number must be"),
        (DOC_A, "2", "page_number must be"),
        (DOC_A, 2.0, "page_number must be"),
        (DOC_A, None, "page_number must be"),
    ],
)
def test_get_page_chunks_invalid_input_raises_before_chroma(
    doc_id: Any, page: Any, message: str
) -> None:
    with pytest.raises(VectorStoreError, match=message):
        get_page_chunks(ExplodingCollection(), doc_id, page)  # type: ignore[arg-type]


def test_get_page_chunks_chroma_error_becomes_vector_store_error() -> None:
    from chromadb.errors import InternalError

    class FailingCollection:
        def get(self, **kwargs: Any) -> Any:
            raise InternalError("simulated")

    with pytest.raises(VectorStoreError, match="Could not fetch chunks"):
        get_page_chunks(FailingCollection(), DOC_A, 1)  # type: ignore[arg-type]


def test_get_page_chunks_rejects_rows_from_another_page() -> None:
    class WrongPageCollection:
        def get(self, **kwargs: Any) -> Any:
            meta = {"doc_id": DOC_A, "filename": "a.pdf", "page_number": 9, "chunk_index": 0}
            return {"ids": [f"{DOC_A}:0"], "documents": ["t"], "metadatas": [meta]}

    with pytest.raises(VectorStoreError, match="outside doc_id"):
        get_page_chunks(WrongPageCollection(), DOC_A, 1)  # type: ignore[arg-type]
