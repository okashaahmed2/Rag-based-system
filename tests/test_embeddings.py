"""Tests for rag_pipeline.embeddings (stub model only, no download)."""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
import types
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from rag_pipeline.config import Settings
from rag_pipeline.embeddings import (
    EMBEDDING_DEVICE,
    ChunkEmbeddings,
    EmbeddingError,
    count_tokens,
    embed_chunks,
    embed_query,
    load_embedding_model,
)
from rag_pipeline.preparation import DocumentChunk

DIM = 8
MAX_SEQ = 6
MODEL_NAME = "stub/model"


class StubTokenizer:
    """One token per whitespace-separated word, plus [CLS] and [SEP]."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, texts: list[str], *, add_special_tokens: bool, truncation: bool
    ) -> dict[str, list[list[int]]]:
        self.calls.append(
            {"texts": texts, "add_special_tokens": add_special_tokens, "truncation": truncation}
        )
        special = 2 if add_special_tokens else 0
        return {"input_ids": [[0] * (len(text.split()) + special) for text in texts]}


def _vector(text: str, normalize: bool) -> np.ndarray:
    """Deterministic float64 vector derived from the text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector = np.frombuffer(digest[:DIM], dtype=np.uint8).astype(np.float64) + 1.0
    return vector / np.linalg.norm(vector) if normalize else vector


class StubModel:
    """Minimal stand-in for SentenceTransformer that records encode calls."""

    def __init__(self, dimension: int | None = DIM, max_seq_length: int | None = MAX_SEQ) -> None:
        self.tokenizer = StubTokenizer()
        self.max_seq_length = max_seq_length
        self._dimension = dimension
        self.encode_calls: list[dict[str, Any]] = []

    def get_embedding_dimension(self) -> int | None:
        return self._dimension

    def encode(self, sentences: str | list[str], **kwargs: Any) -> np.ndarray:
        self.encode_calls.append({"sentences": sentences, **kwargs})
        normalize = kwargs["normalize_embeddings"]
        if isinstance(sentences, str):
            return _vector(sentences, normalize)
        return np.stack([_vector(text, normalize) for text in sentences])


def _chunk(index: int, text: str, doc_id: str = "d" * 64) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc_id}:{index}",
        doc_id=doc_id,
        filename="doc.pdf",
        page_number=1,
        chunk_index=index,
        text=text,
        char_start=0,
        char_end=len(text),
    )


@pytest.fixture
def model() -> StubModel:
    return StubModel()


# --- embed_chunks -----------------------------------------------------------


def test_embed_chunks_preserves_order_and_length(model: StubModel) -> None:
    chunks = [_chunk(i, f"chunk number {i}") for i in (2, 0, 1)]

    result = embed_chunks(chunks, model, 32, model_name=MODEL_NAME)

    assert result.chunk_ids == tuple(c.chunk_id for c in chunks)
    assert len(result.token_counts) == 3
    for row, chunk in zip(result.vectors, chunks):
        np.testing.assert_allclose(row, _vector(chunk.text, True), rtol=1e-6)


def test_embed_chunks_shape_dtype_and_metadata(model: StubModel) -> None:
    result = embed_chunks([_chunk(0, "a b"), _chunk(1, "c d e")], model, 4, model_name=MODEL_NAME)

    assert isinstance(result, ChunkEmbeddings)
    assert result.vectors.shape == (2, DIM)
    assert result.vectors.dtype == np.float32
    assert result.model_name == MODEL_NAME
    assert result.dimension == DIM
    assert result.max_seq_length == MAX_SEQ
    assert result.token_counts == (4, 5)
    assert result.over_limit_chunk_ids == ()


def test_embed_chunks_normalizes_and_passes_batch_size(model: StubModel) -> None:
    result = embed_chunks([_chunk(0, "x"), _chunk(1, "y")], model, 7, model_name=MODEL_NAME)

    assert len(model.encode_calls) == 1
    call = model.encode_calls[0]
    assert call["batch_size"] == 7
    assert call["normalize_embeddings"] is True
    assert call["sentences"] == ["x", "y"]
    np.testing.assert_allclose(np.linalg.norm(result.vectors, axis=1), 1.0, rtol=1e-6)


def test_embed_chunks_empty_input_does_not_call_encode(model: StubModel) -> None:
    result = embed_chunks([], model, 32, model_name=MODEL_NAME)

    assert model.encode_calls == []
    assert model.tokenizer.calls == []
    assert result.chunk_ids == ()
    assert result.vectors.shape == (0, DIM)
    assert result.vectors.dtype == np.float32
    assert result.token_counts == ()
    assert result.over_limit_chunk_ids == ()


def test_over_limit_chunk_is_warned_listed_and_still_embedded(
    model: StubModel, caplog: pytest.LogCaptureFixture
) -> None:
    short = _chunk(0, "one two")  # 4 tokens
    exact = _chunk(1, "one two three four")  # 6 tokens == limit, not over
    long = _chunk(2, "one two three four five six seven")  # 9 tokens

    with caplog.at_level(logging.WARNING, logger="rag_pipeline.embeddings"):
        result = embed_chunks([short, exact, long], model, 32, model_name=MODEL_NAME)

    assert result.token_counts == (4, 6, 9)
    assert result.over_limit_chunk_ids == (long.chunk_id,)
    assert result.vectors.shape == (3, DIM)
    np.testing.assert_allclose(result.vectors[2], _vector(long.text, True), rtol=1e-6)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert long.chunk_id in warnings[0].getMessage()
    assert "9 tokens" in warnings[0].getMessage()


@pytest.mark.parametrize("bad_size", [0, -1])
def test_embed_chunks_rejects_bad_batch_size(model: StubModel, bad_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        embed_chunks([_chunk(0, "x")], model, bad_size, model_name=MODEL_NAME)
    assert model.encode_calls == []


def test_embed_chunks_rejects_wrong_output_shape() -> None:
    class WrongShape(StubModel):
        def encode(self, sentences: str | list[str], **kwargs: Any) -> np.ndarray:
            return np.zeros((1, DIM))

    with pytest.raises(EmbeddingError, match="shape"):
        embed_chunks([_chunk(0, "a"), _chunk(1, "b")], WrongShape(), 2, model_name=MODEL_NAME)


@pytest.mark.parametrize(("dimension", "max_seq"), [(None, MAX_SEQ), (DIM, None), (0, MAX_SEQ)])
def test_embed_chunks_rejects_unusable_model_metadata(
    dimension: int | None, max_seq: int | None
) -> None:
    with pytest.raises(EmbeddingError, match="invalid"):
        embed_chunks([], StubModel(dimension, max_seq), 1, model_name=MODEL_NAME)


# --- count_tokens -----------------------------------------------------------


def test_count_tokens_includes_special_tokens_without_truncation(model: StubModel) -> None:
    assert count_tokens(["a b c", ""], model) == [5, 2]
    assert model.tokenizer.calls[0]["add_special_tokens"] is True
    assert model.tokenizer.calls[0]["truncation"] is False


def test_count_tokens_empty_input_skips_tokenizer(model: StubModel) -> None:
    assert count_tokens([], model) == []
    assert model.tokenizer.calls == []


# --- embed_query ------------------------------------------------------------


def test_embed_query_returns_normalized_1d_float32(model: StubModel) -> None:
    vector = embed_query("What is a decision tree?", model)

    assert vector.shape == (DIM,)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0, rel=1e-6)
    assert model.encode_calls[0]["normalize_embeddings"] is True


def test_embed_query_matches_chunk_embedding_of_same_text(model: StubModel) -> None:
    text = "same text"
    chunk_vector = embed_chunks([_chunk(0, text)], model, 1, model_name=MODEL_NAME).vectors[0]

    np.testing.assert_array_equal(embed_query(text, model), chunk_vector)


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_embed_query_rejects_blank_text(model: StubModel, blank: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        embed_query(blank, model)
    assert model.encode_calls == []


# --- load_embedding_model (fake sentence_transformers module) ----------------


@pytest.fixture
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    """Install a fake ``sentence_transformers`` module that records constructor calls."""
    calls: list[dict[str, Any]] = []

    class FakeSentenceTransformer(StubModel):
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            calls.append({"model_name": model_name, **kwargs})
            if model_name == "missing/model":
                raise OSError("not in local cache")
            super().__init__()

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    load_embedding_model.cache_clear()
    yield calls
    load_embedding_model.cache_clear()


def test_load_embedding_model_uses_cpu_and_local_files_flag(
    fake_sentence_transformers: list[dict[str, Any]],
) -> None:
    load_embedding_model("some/model", True)

    assert fake_sentence_transformers == [
        {"model_name": "some/model", "device": EMBEDDING_DEVICE, "local_files_only": True}
    ]
    assert EMBEDDING_DEVICE == "cpu"


def test_load_embedding_model_is_cached_per_name_and_flag(
    fake_sentence_transformers: list[dict[str, Any]],
) -> None:
    first = load_embedding_model("some/model", False)

    assert load_embedding_model("some/model", False) is first
    assert load_embedding_model("some/model", True) is not first
    assert load_embedding_model("other/model", False) is not first
    assert len(fake_sentence_transformers) == 3


def test_load_embedding_model_logs_facts(
    fake_sentence_transformers: list[dict[str, Any]], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="rag_pipeline.embeddings"):
        load_embedding_model("some/model", False)

    message = caplog.records[-1].getMessage()
    assert "some/model" in message
    assert f"dimension={DIM}" in message
    assert f"max_seq_length={MAX_SEQ}" in message


def test_load_embedding_model_wraps_os_error(
    fake_sentence_transformers: list[dict[str, Any]], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger="rag_pipeline.embeddings"):
        with pytest.raises(EmbeddingError, match="missing/model") as info:
            load_embedding_model("missing/model", True)

    assert isinstance(info.value.__cause__, OSError)
    assert any("missing/model" in r.getMessage() for r in caplog.records)


def test_importing_module_does_not_import_sentence_transformers() -> None:
    code = (
        "import sys, rag_pipeline.embeddings; "
        "print('sentence_transformers' in sys.modules, 'torch' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )

    assert completed.stdout.strip() == "False False"


# --- Settings ---------------------------------------------------------------


def test_embedding_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_batch_size == 32
    assert settings.embedding_local_files_only is False


def test_embedding_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_MODEL_NAME", "other/model")
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "1")
    monkeypatch.setenv("RAG_EMBEDDING_LOCAL_FILES_ONLY", "true")

    settings = Settings(_env_file=None)

    assert settings.embedding_model_name == "other/model"
    assert settings.embedding_batch_size == 1
    assert settings.embedding_local_files_only is True


@pytest.mark.parametrize("bad_value", ["0", "-5"])
def test_embedding_batch_size_below_one_is_rejected(
    bad_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", bad_value)

    with pytest.raises(ValidationError, match="RAG_EMBEDDING_BATCH_SIZE must be >= 1"):
        Settings(_env_file=None)
