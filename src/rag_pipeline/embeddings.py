"""Embed document chunks and queries with a Sentence Transformers model.

``sentence_transformers`` is imported lazily inside
:func:`load_embedding_model` (the import alone takes seconds), so importing
this module stays cheap and unit tests can use a stub model.

Every function except :func:`load_embedding_model` takes the model as an
argument typed with the small :class:`EmbeddingModel` protocol, which
describes only the attributes used here.

The model silently truncates input longer than ``max_seq_length`` word-piece
tokens (special tokens included). :func:`embed_chunks` therefore counts the
real tokens of every chunk and logs a WARNING for each chunk over the limit;
such chunks are still embedded (from their first ``max_seq_length`` tokens).

No vectors are stored here; storage happens in a later phase.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from rag_pipeline.preparation import DocumentChunk

logger = logging.getLogger(__name__)

#: Device the model runs on (the target machine has no usable GPU).
EMBEDDING_DEVICE = "cpu"

#: Disables huggingface_hub telemetry, including its agent-harness registry
#: request (``GET {ENDPOINT}/api/agent-harnesses``), made even for local loads.
HF_TELEMETRY_ENV = "HF_HUB_DISABLE_TELEMETRY"
#: Forbids every Hugging Face Hub request (set only for local-only loading).
HF_OFFLINE_ENV = "HF_HUB_OFFLINE"

FloatArray = npt.NDArray[np.float32]


class EmbeddingError(Exception):
    """Raised when a model cannot be loaded or returns unusable output."""


class EmbeddingModel(Protocol):
    """The parts of a ``SentenceTransformer`` this module uses."""

    @property
    def tokenizer(self) -> Any:
        """Hugging Face tokenizer: ``tokenizer(texts, ...)["input_ids"]``."""
        ...

    @property
    def max_seq_length(self) -> int | None:
        """Maximum input length in tokens, special tokens included."""
        ...

    def get_embedding_dimension(self) -> int | None:
        """Length of one embedding vector."""
        ...

    def encode(
        self,
        sentences: str | list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Any:
        """Embed one text (1-D result) or a list of texts (2-D result)."""
        ...


@dataclass(frozen=True)
class ChunkEmbeddings:
    """Embeddings of a sequence of chunks, row ``i`` belonging to ``chunk_ids[i]``.

    Attributes:
        chunk_ids: Chunk ids in input order.
        vectors: L2-normalised ``float32`` array of shape ``(n, dimension)``.
        model_name: Name of the model that produced the vectors.
        dimension: Length of one vector.
        max_seq_length: Model input limit in tokens, special tokens included.
        token_counts: Real token count of each chunk (special tokens
            included, no truncation), in input order.
        over_limit_chunk_ids: Ids of chunks with more than ``max_seq_length``
            tokens (embedded from their truncated start), in input order.
    """

    chunk_ids: tuple[str, ...]
    vectors: FloatArray
    model_name: str
    dimension: int
    max_seq_length: int
    token_counts: tuple[int, ...]
    over_limit_chunk_ids: tuple[str, ...]


def _require_positive(name: str, value: int | None) -> int:
    """Return ``value`` if it is a positive int, else raise :class:`EmbeddingError`."""
    if value is None or value < 1:
        raise EmbeddingError(f"Embedding model reports an invalid {name}: {value!r}")
    return value


def model_dimension(model: EmbeddingModel) -> int:
    """Embedding dimension of ``model`` (raises :class:`EmbeddingError` if unknown)."""
    return _require_positive("embedding dimension", model.get_embedding_dimension())


def model_max_seq_length(model: EmbeddingModel) -> int:
    """Input limit of ``model`` in tokens (raises :class:`EmbeddingError` if unknown)."""
    return _require_positive("max_seq_length", model.max_seq_length)


def apply_hf_runtime_env(local_files_only: bool) -> None:
    """Set the Hugging Face environment variables that keep model loading local.

    ``HF_HUB_DISABLE_TELEMETRY=1`` is always set; with ``local_files_only``
    ``HF_HUB_OFFLINE=1`` is set too. A value already present (set by the user)
    is never overwritten. huggingface_hub reads these variables once, when it
    is imported, so this must run before that import: if huggingface_hub is
    already loaded and its constants do not reflect the variables, a WARNING
    says the change came too late (the constants are not patched).

    The variables must live in the process environment: values in ``.env``
    are read by pydantic-settings only and never reach huggingface_hub.
    """
    wanted = [HF_TELEMETRY_ENV] + ([HF_OFFLINE_ENV] if local_files_only else [])
    set_by_us = [name for name in wanted if name not in os.environ]
    for name in wanted:
        os.environ.setdefault(name, "1")
    logger.debug(
        "Hugging Face env: set %s, already present %s",
        set_by_us or "none",
        [name for name in wanted if name not in set_by_us] or "none",
    )

    hub = sys.modules.get("huggingface_hub")
    if hub is None:
        return
    constants = sys.modules.get("huggingface_hub.constants") or getattr(hub, "constants", None)
    if constants is None:
        return
    not_applied = [name for name in wanted if getattr(constants, name, None) is not True]
    if not_applied:
        logger.warning(
            "huggingface_hub was imported before the Hugging Face environment was set; "
            "%s not in effect for this process (set them in the real environment to "
            "stop Hub requests)",
            ", ".join(not_applied),
        )


@lru_cache(maxsize=4)
def load_embedding_model(model_name: str, local_files_only: bool = False) -> EmbeddingModel:
    """Load a Sentence Transformers model on CPU, once per ``(name, flag)``.

    With ``local_files_only=True`` the model is read from the local Hugging
    Face cache only and the Hub is never contacted; loading then fails if the
    model was not downloaded before. The result is cached; call
    ``load_embedding_model.cache_clear()`` to force a reload.

    :func:`apply_hf_runtime_env` runs first, before ``sentence_transformers``
    (and with it huggingface_hub) is imported.

    Raises:
        EmbeddingError: the model cannot be found, downloaded or read, or
            reports no usable dimension / max_seq_length.
    """
    apply_hf_runtime_env(local_files_only)
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    try:
        model = SentenceTransformer(
            model_name, device=EMBEDDING_DEVICE, local_files_only=local_files_only
        )
    except OSError as exc:
        logger.error(
            "Could not load embedding model %s (local_files_only=%s): %s",
            model_name,
            local_files_only,
            exc,
        )
        raise EmbeddingError(
            f"Could not load embedding model {model_name!r} "
            f"(local_files_only={local_files_only}): {exc}"
        ) from exc
    elapsed = time.perf_counter() - started
    logger.info(
        "Loaded embedding model %s: dimension=%d, max_seq_length=%d, "
        "local_files_only=%s, %.2f s",
        model_name,
        model_dimension(model),
        model_max_seq_length(model),
        local_files_only,
        elapsed,
    )
    return model


def count_tokens(texts: Sequence[str], model: EmbeddingModel) -> list[int]:
    """Token count of each text with the model's tokenizer.

    Special tokens are included and nothing is truncated, so a count above
    the model's ``max_seq_length`` means the model would ignore the excess.
    """
    if not texts:
        return []
    encoded = model.tokenizer(list(texts), add_special_tokens=True, truncation=False)
    return [len(ids) for ids in encoded["input_ids"]]


def _to_float32_matrix(raw: Any, rows: int, dimension: int) -> FloatArray:
    """Convert encoder output to ``float32`` and check its shape."""
    vectors = np.asarray(raw, dtype=np.float32)
    if vectors.shape != (rows, dimension):
        raise EmbeddingError(
            f"Embedding model returned shape {vectors.shape}, "
            f"expected {(rows, dimension)}"
        )
    return vectors


def embed_chunks(
    chunks: Sequence[DocumentChunk],
    model: EmbeddingModel,
    batch_size: int,
    *,
    model_name: str,
) -> ChunkEmbeddings:
    """Embed chunk texts with L2 normalisation, keeping the input order.

    ``model_name`` is recorded in the result (a loaded model does not carry
    its name). Each chunk over ``max_seq_length`` tokens is logged as a
    WARNING with its ``chunk_id`` and token count and still embedded. Empty
    input returns an empty ``(0, dimension)`` result without calling
    ``model.encode``.

    Raises:
        ValueError: ``batch_size`` is smaller than 1.
        EmbeddingError: the model reports no usable dimension or limit, or
            returns vectors of the wrong shape.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    dimension = model_dimension(model)
    limit = model_max_seq_length(model)
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    if not chunks:
        return ChunkEmbeddings(
            chunk_ids=(),
            vectors=np.empty((0, dimension), dtype=np.float32),
            model_name=model_name,
            dimension=dimension,
            max_seq_length=limit,
            token_counts=(),
            over_limit_chunk_ids=(),
        )

    texts = [chunk.text for chunk in chunks]
    token_counts = tuple(count_tokens(texts, model))
    over_limit: list[str] = []
    for chunk_id, tokens in zip(chunk_ids, token_counts):
        if tokens > limit:
            over_limit.append(chunk_id)
            logger.warning(
                "Chunk %s has %d tokens, over max_seq_length %d; "
                "the model ignores the excess",
                chunk_id,
                tokens,
                limit,
            )

    raw = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = _to_float32_matrix(raw, len(texts), dimension)
    logger.info(
        "Embedded %d chunk(s) with %s (dimension=%d, %d over the token limit)",
        len(texts),
        model_name,
        dimension,
        len(over_limit),
    )
    return ChunkEmbeddings(
        chunk_ids=chunk_ids,
        vectors=vectors,
        model_name=model_name,
        dimension=dimension,
        max_seq_length=limit,
        token_counts=token_counts,
        over_limit_chunk_ids=tuple(over_limit),
    )


def embed_query(text: str, model: EmbeddingModel) -> FloatArray:
    """Embed one query as a 1-D L2-normalised ``float32`` vector.

    Uses the same model and normalisation as :func:`embed_chunks`.

    Raises:
        ValueError: ``text`` is empty or whitespace only.
        EmbeddingError: the model returns a vector of the wrong shape.
    """
    if not text.strip():
        raise ValueError("query text must not be empty")
    dimension = model_dimension(model)
    raw = model.encode(
        text,
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vector = np.asarray(raw, dtype=np.float32)
    if vector.shape != (dimension,):
        raise EmbeddingError(
            f"Embedding model returned shape {vector.shape} for a query, "
            f"expected {(dimension,)}"
        )
    return vector
