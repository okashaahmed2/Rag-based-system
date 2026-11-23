"""Retrieve the chunks nearest to a question, from ``done`` documents only.

This is the read path. :func:`retrieve` embeds the question with the same
model used for ingestion and asks Chroma for the ``top_k`` nearest chunks,
restricted inside Chroma to documents whose SQLite status is ``done``. A
``failed`` document may still hold partial chunks in Chroma, and ``pending``,
``processing`` and ``needs_ocr`` documents are not finished, so none of them
may ever be returned. With no ``done`` document at all, Chroma is not
queried (Chroma 1.5.9 rejects an empty ``$in`` list).

:func:`build_retrieval_context` wires the status store, model and collection
from settings. It never runs :meth:`StatusStore.recover_interrupted` (that is
ingestion startup work, and a reader must not change document status) and it
never creates a missing status database, vector store folder or collection:
all must have been created by ingestion first.

No chunk or question text is logged (only lengths and counts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chromadb.api.models.Collection import Collection

from rag_pipeline.config import Settings
from rag_pipeline.embeddings import (
    EmbeddingError,
    EmbeddingModel,
    embed_query,
    load_embedding_model,
    model_dimension,
)
from rag_pipeline.ingestion import ModelLoader
from rag_pipeline.status_db import DocumentStatus, StatusStore, StatusStoreError
from rag_pipeline.vector_store import (
    CollectionNotFoundError,
    RetrievedChunk,
    VectorStoreError,
    open_collection,
    query_by_embedding,
)

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when retrieval cannot be set up or a query cannot be answered."""


@dataclass(frozen=True)
class RetrievalContext:
    """Everything :func:`retrieve` needs, built once per process.

    The model must be the one the collection was built with; only
    :func:`build_retrieval_context` guarantees that (it opens the collection
    with ``settings.embedding_model_name``, which must match the name saved
    in the collection).
    """

    store: StatusStore
    collection: Collection
    model: EmbeddingModel


def build_retrieval_context(
    settings: Settings, *, model_loader: ModelLoader = load_embedding_model
) -> RetrievalContext:
    """Wire the status store, embedding model and collection from ``settings``.

    The status database file, the Chroma folder and the collection must
    already exist (made by ingestion); a missing one raises instead of being
    created (the collection is opened with ``create=False``), so a wrong path
    or name never yields a silent, empty store. The collection's saved model,
    dimension and distance space are still checked. Interrupted documents are
    not recovered here.

    Raises:
        RetrievalError: a store or the collection is missing, the model
            cannot be loaded, or
            the collection cannot be opened or does not match the model.
    """
    if not settings.status_db_path.is_file():
        message = (
            f"Status database not found at {settings.status_db_path}; "
            "run ingestion first"
        )
        logger.error(message)
        raise RetrievalError(message)
    if not settings.chroma_path.is_dir():
        message = f"Vector store not found at {settings.chroma_path}; run ingestion first"
        logger.error(message)
        raise RetrievalError(message)

    try:
        model = model_loader(settings.embedding_model_name, settings.embedding_local_files_only)
        dimension = model_dimension(model)
    except EmbeddingError as exc:
        logger.error("Embedding model setup failed: %s", exc)
        raise RetrievalError(f"Embedding model setup failed: {exc}") from exc

    try:
        collection = open_collection(
            settings.chroma_path,
            settings.chroma_collection_name,
            settings.embedding_model_name,
            dimension,
            create=False,
        )
    except CollectionNotFoundError as exc:
        message = f"{exc}; run ingestion first"
        logger.error(message)
        raise RetrievalError(message) from exc
    except VectorStoreError as exc:
        logger.error("Vector store setup failed: %s", exc)
        raise RetrievalError(f"Vector store setup failed: {exc}") from exc

    return RetrievalContext(
        store=StatusStore(settings.status_db_path), collection=collection, model=model
    )


def done_doc_ids(store: StatusStore) -> set[str]:
    """``doc_id`` of every document whose status is ``done``.

    Raises:
        RetrievalError: the status store cannot be read.
    """
    try:
        records = store.list_documents(DocumentStatus.DONE)
    except StatusStoreError as exc:
        logger.error("Could not read done documents: %s", exc)
        raise RetrievalError(f"Could not read done documents: {exc}") from exc
    return {record.doc_id for record in records}


def _validate(query: object, top_k: object) -> str:
    """Return the stripped query; raise :class:`RetrievalError` on bad input."""
    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise RetrievalError(f"top_k must be an int >= 1, got {top_k!r}")
    return query.strip()


def retrieve(query: str, context: RetrievalContext, top_k: int) -> list[RetrievedChunk]:
    """Return up to ``top_k`` chunks of ``done`` documents nearest to ``query``.

    Results are nearest first (ascending cosine distance). Fewer than
    ``top_k`` matching chunks returns all of them; no ``done`` document
    returns ``[]`` without embedding the query or querying Chroma.

    A document whose status changes after the ``done`` list was read may
    still be included in that one call (no lock across SQLite and Chroma).

    Raises:
        RetrievalError: empty query, ``top_k < 1``, or the status store,
            model or vector store fails.
    """
    text = _validate(query, top_k)
    allowed = done_doc_ids(context.store)
    if not allowed:
        logger.info("Retrieval skipped: no done documents")
        return []

    try:
        vector = embed_query(text, context.model)
        results = query_by_embedding(context.collection, vector, top_k, doc_ids=allowed)
    except (EmbeddingError, VectorStoreError, ValueError) as exc:
        logger.error("Retrieval failed (query length %d): %s", len(text), exc)
        raise RetrievalError(f"Retrieval failed: {exc}") from exc

    leaked = [r.chunk_id for r in results if r.doc_id not in allowed]
    if leaked:
        # Guard against a filter that silently stops working (e.g. a Chroma upgrade).
        message = f"Vector store returned {len(leaked)} chunk(s) of non-done documents"
        logger.error(message)
        raise RetrievalError(message)

    logger.info(
        "Retrieved %d chunk(s) (top_k=%d, %d done document(s), query length %d)",
        len(results),
        top_k,
        len(allowed),
        len(text),
    )
    return results
