"""Ingest one PDF end to end with status tracking.

:func:`ingest_document` hashes the file, registers it in the status store,
skips it if there is nothing to do, and otherwise runs
prepare -> embed -> replace chunks in Chroma -> mark the final status. Every
ordinary failure ends as ``failed`` in the status store (and is returned, not
raised) so a caller can continue with the next document. A ``BaseException``
(e.g. ``KeyboardInterrupt``) is not caught: the document stays
``processing`` and is turned into ``failed`` by
:meth:`StatusStore.recover_interrupted` at the next startup. A failing status
store itself (:class:`StatusStoreError`) is logged as CRITICAL and re-raised:
continuing without a working register is unsafe.

The Chroma replace always happens before ``mark_done``, so a document is never
``done`` without its chunks. No file is moved, renamed or deleted here.

:func:`build_ingest_context` is the only place that wires the real
dependencies (status store, embedding model, collection) from settings.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from chromadb.api.models.Collection import Collection

from rag_pipeline.config import Settings
from rag_pipeline.embeddings import (
    EmbeddingError,
    EmbeddingModel,
    embed_chunks,
    load_embedding_model,
    model_dimension,
)
from rag_pipeline.extraction import PdfExtractionError, compute_file_doc_id
from rag_pipeline.preparation import LOG_DOC_ID_CHARS, prepare_document
from rag_pipeline.status_db import (
    DocumentRecord,
    DocumentStatus,
    StatusStore,
    StatusStoreError,
)
from rag_pipeline.vector_store import (
    VectorStoreError,
    open_collection,
    replace_document_chunks,
)

logger = logging.getLogger(__name__)

SKIP_ALREADY_INGESTED = "already ingested"
SKIP_IN_PROGRESS = "in progress"
SKIP_ATTEMPTS_EXHAUSTED = "attempts exhausted"


class IngestionError(Exception):
    """A document cannot be ingested for a reason found by this module."""


class IngestionSetupError(Exception):
    """The ingestion dependencies could not be built from the settings."""


#: Failures expected for bad input files: logged as one line, no traceback.
_EXPECTED_FAILURES: tuple[type[Exception], ...] = (PdfExtractionError, IngestionError)


class IngestOutcome(str, Enum):
    """Result of one :func:`ingest_document` call."""

    DONE = "done"
    NEEDS_OCR = "needs_ocr"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class IngestContext:
    """Everything :func:`ingest_document` needs, built once per process.

    Raises:
        ValueError: a numeric setting is out of range or the model name is empty.
    """

    store: StatusStore
    collection: Collection
    model: EmbeddingModel
    embedding_model_name: str
    embedding_batch_size: int
    chunk_size: int
    chunk_overlap: int
    max_ingest_attempts: int

    def __post_init__(self) -> None:
        if not self.embedding_model_name.strip():
            raise ValueError("embedding_model_name must not be empty")
        if self.embedding_batch_size < 1:
            raise ValueError(f"embedding_batch_size must be >= 1, got {self.embedding_batch_size}")
        if self.chunk_size < 1 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                f"need chunk_size >= 1 and 0 <= chunk_overlap < chunk_size, got "
                f"{self.chunk_size}/{self.chunk_overlap}"
            )
        if self.max_ingest_attempts < 1:
            raise ValueError(f"max_ingest_attempts must be >= 1, got {self.max_ingest_attempts}")


@dataclass(frozen=True)
class IngestResult:
    """What happened to one PDF.

    Attributes:
        path: The file that was ingested.
        doc_id: SHA-256 of the file, or ``None`` if it could not be hashed.
        filename: Base name of ``path`` (may differ from the registered name
            for a renamed copy).
        outcome: done / needs_ocr / failed / skipped.
        chunk_count: Chunks stored (done, needs_ocr; for a skipped document the
            stored count), else ``None``.
        error_message: Failure reason (failed; for a document skipped after
            exhausting its attempts, its last error), else ``None``.
        attempts: Processing attempts recorded for the document, or ``None``
            if it was never registered.
        skip_reason: Why the document was skipped, else ``None``.
        duration_seconds: Wall time of the call.
    """

    path: Path
    doc_id: str | None
    filename: str
    outcome: IngestOutcome
    chunk_count: int | None
    error_message: str | None
    attempts: int | None
    skip_reason: str | None
    duration_seconds: float


ModelLoader = Callable[[str, bool], EmbeddingModel]


def build_ingest_context(
    settings: Settings, *, model_loader: ModelLoader = load_embedding_model
) -> IngestContext:
    """Wire the real dependencies from ``settings`` (the only place this happens).

    Initializes the status store, recovers documents interrupted by an earlier
    process, loads the embedding model and opens the collection with the
    model's dimension and ``settings.embedding_model_name``.

    Raises:
        IngestionSetupError: any part fails; the message names the part.
    """
    store = StatusStore(settings.status_db_path)
    try:
        store.initialize()
        recovered = store.recover_interrupted()
    except StatusStoreError as exc:
        logger.error("Status store setup failed: %s", exc)
        raise IngestionSetupError(f"Status store setup failed: {exc}") from exc
    logger.info("Startup recovery: %d interrupted document(s) marked failed", recovered)

    try:
        model = model_loader(settings.embedding_model_name, settings.embedding_local_files_only)
        dimension = model_dimension(model)
    except EmbeddingError as exc:
        logger.error("Embedding model setup failed: %s", exc)
        raise IngestionSetupError(f"Embedding model setup failed: {exc}") from exc

    try:
        collection = open_collection(
            settings.chroma_path,
            settings.chroma_collection_name,
            settings.embedding_model_name,
            dimension,
        )
    except VectorStoreError as exc:
        logger.error("Vector store setup failed: %s", exc)
        raise IngestionSetupError(f"Vector store setup failed: {exc}") from exc

    return IngestContext(
        store=store,
        collection=collection,
        model=model,
        embedding_model_name=settings.embedding_model_name,
        embedding_batch_size=settings.embedding_batch_size,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_ingest_attempts=settings.max_ingest_attempts,
    )


def _skip_reason(record: DocumentRecord, max_attempts: int) -> str | None:
    """Why ``record`` must not be processed now, or ``None`` if it may be."""
    if record.status in (DocumentStatus.DONE, DocumentStatus.NEEDS_OCR):
        return SKIP_ALREADY_INGESTED
    if record.status is DocumentStatus.PROCESSING:
        return SKIP_IN_PROGRESS
    if record.status is DocumentStatus.FAILED and record.attempts >= max_attempts:
        return SKIP_ATTEMPTS_EXHAUSTED
    return None


def _preparation_settings(context: IngestContext) -> Settings:
    """A Settings object carrying only the chunk values ``prepare_document`` reads.

    ``model_construct`` skips environment/.env loading and validation; the
    values were validated by :class:`IngestContext` (and by Settings when the
    context was built from settings).
    """
    return Settings.model_construct(
        chunk_size=context.chunk_size, chunk_overlap=context.chunk_overlap
    )


def _process(path: Path, doc_id: str, context: IngestContext) -> tuple[IngestOutcome, int]:
    """prepare -> (needs_ocr | embed -> replace) -> final status; return outcome."""
    prepared = prepare_document(path, _preparation_settings(context))
    if prepared.doc_id != doc_id:
        raise IngestionError(
            f"file changed during processing (doc_id {doc_id[:LOG_DOC_ID_CHARS]} at "
            f"registration, {prepared.doc_id[:LOG_DOC_ID_CHARS]} at extraction)"
        )
    if prepared.needs_ocr:
        replace_document_chunks(context.collection, doc_id, [], None)
        context.store.mark_needs_ocr(doc_id)
        return IngestOutcome.NEEDS_OCR, 0
    if not prepared.chunks:
        raise IngestionError(
            "no chunks produced (the PDF has text, but cleaning and header/footer "
            "removal left none)"
        )
    embeddings = embed_chunks(
        prepared.chunks,
        context.model,
        context.embedding_batch_size,
        model_name=context.embedding_model_name,
    )
    stored = replace_document_chunks(context.collection, doc_id, prepared.chunks, embeddings)
    context.store.mark_done(doc_id, stored)
    return IngestOutcome.DONE, stored


def ingest_document(path: Path, context: IngestContext) -> IngestResult:
    """Ingest one PDF and record its status; see the module docstring.

    Returns:
        The outcome; ordinary failures are returned as ``failed``.

    Raises:
        StatusStoreError: the status store failed (logged as CRITICAL).
        BaseException: non-``Exception`` interrupts propagate unchanged; the
            document is then left ``processing`` for startup recovery.
    """
    started = time.perf_counter()

    def result(
        outcome: IngestOutcome,
        doc_id: str | None,
        *,
        chunk_count: int | None = None,
        error_message: str | None = None,
        attempts: int | None = None,
        skip_reason: str | None = None,
    ) -> IngestResult:
        ingest_result = IngestResult(
            path=path,
            doc_id=doc_id,
            filename=path.name,
            outcome=outcome,
            chunk_count=chunk_count,
            error_message=error_message,
            attempts=attempts,
            skip_reason=skip_reason,
            duration_seconds=time.perf_counter() - started,
        )
        logger.info(
            "Ingest %s (%s): %s, chunks=%s, attempts=%s%s, %.2f s",
            (doc_id or "-")[:LOG_DOC_ID_CHARS],
            path.name,
            outcome.value,
            chunk_count,
            attempts,
            f", skipped: {skip_reason}" if skip_reason else "",
            ingest_result.duration_seconds,
        )
        return ingest_result

    try:
        doc_id = compute_file_doc_id(path)
    except OSError as exc:
        message = f"{type(exc).__name__}: {exc}"
        logger.error("Cannot hash %s, nothing registered: %s", path, message)
        return result(IngestOutcome.FAILED, None, error_message=message)

    try:
        record = context.store.register_document(doc_id, path.name)
        reason = _skip_reason(record, context.max_ingest_attempts)
        if reason is not None:
            return result(
                IngestOutcome.SKIPPED,
                doc_id,
                chunk_count=record.chunk_count,
                error_message=record.error_message,
                attempts=record.attempts,
                skip_reason=reason,
            )
        attempts = context.store.mark_processing(doc_id).attempts

        try:
            outcome, count = _process(path, doc_id, context)
        except StatusStoreError:
            raise
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            # Expected failures: one line; anything else: full traceback.
            logger.error(
                "Ingest of %s (%s) failed on attempt %d: %s",
                doc_id[:LOG_DOC_ID_CHARS],
                path.name,
                attempts,
                message,
                exc_info=not isinstance(exc, _EXPECTED_FAILURES),
            )
            failed = context.store.mark_failed(doc_id, message)
            return result(
                IngestOutcome.FAILED,
                doc_id,
                error_message=failed.error_message,
                attempts=failed.attempts,
            )
        return result(outcome, doc_id, chunk_count=count, attempts=attempts)
    except StatusStoreError as exc:
        logger.critical(
            "Status store failed while ingesting %s (%s); stopping: %s",
            doc_id[:LOG_DOC_ID_CHARS],
            path.name,
            exc,
        )
        raise
