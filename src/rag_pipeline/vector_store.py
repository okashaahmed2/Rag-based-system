"""Store, replace and query document chunks in a persistent ChromaDB collection.

ChromaDB never computes embeddings here: every collection is created and
opened with ``embedding_function=None`` (otherwise Chroma attaches its own
default model, which downloads on first use), and queries always pass
``query_embeddings``. Vectors come from :mod:`rag_pipeline.embeddings`.

The collection uses cosine distance, so for L2-normalised vectors Chroma
returns ``distance = 1 - cosine similarity``. The embedding model name and
dimension are saved in the collection metadata when it is created and checked
on every open, so vectors from different models are never mixed.

Re-ingestion is idempotent: :func:`replace_document_chunks` validates all
input first, then deletes every chunk of the document and upserts the new
ones. Delete and upsert are separate calls (Chroma has no transactions); a
failure between them leaves the document with no or only some chunks and is
reported as :class:`VectorStoreError`.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import numpy.typing as npt
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import ChromaError, NotFoundError, UniqueConstraintError

from rag_pipeline.embeddings import ChunkEmbeddings
from rag_pipeline.preparation import LOG_DOC_ID_CHARS, DocumentChunk

logger = logging.getLogger(__name__)

#: Distance space of every collection created here.
DISTANCE_SPACE = "cosine"
#: Collection metadata keys holding the embedding model name and dimension.
META_EMBEDDING_MODEL = "embedding_model"
META_EMBEDDING_DIM = "embedding_dim"
#: Warning Chroma 1.5 emits when ``collection.configuration`` is read on a
#: collection without an embedding function (harmless here, by design).
_LEGACY_EF_WARNING = "legacy embedding function config"
#: Chunk metadata keys that :class:`RetrievedChunk` needs.
_REQUIRED_CHUNK_KEYS = ("doc_id", "filename", "page_number", "chunk_index")


class VectorStoreError(Exception):
    """Raised when the vector store cannot be opened, written or queried."""


class CollectionNotFoundError(VectorStoreError):
    """Raised by ``open_collection(..., create=False)`` when the store folder or
    the collection does not exist."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One query result.

    Attributes:
        chunk_id: ``f"{doc_id}:{chunk_index}"``.
        doc_id: SHA-256 hex digest of the source PDF.
        filename: Base name of the source PDF.
        page_number: 1-based page the chunk comes from.
        chunk_index: Index of the chunk within its document.
        text: Chunk text.
        distance: Cosine distance to the query (``1 - similarity``).
        similarity: Cosine similarity to the query (``1 - distance``).
    """

    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    distance: float
    similarity: float


@dataclass(frozen=True)
class PageChunk:
    """One stored chunk fetched by document and page (no query, so no similarity).

    Attributes:
        chunk_id: ``f"{doc_id}:{chunk_index}"``.
        doc_id: SHA-256 hex digest of the source PDF.
        filename: Base name of the source PDF.
        page_number: 1-based page the chunk comes from.
        chunk_index: Index of the chunk within its document.
        text: Chunk text.
    """

    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str


def _read_space(collection: Collection) -> Any:
    """Distance space from the collection configuration (HNSW or SPANN)."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=_LEGACY_EF_WARNING, category=DeprecationWarning
        )
        configuration = collection.configuration
    for index in ("hnsw", "spann"):
        index_config = configuration.get(index)
        if index_config:
            return index_config.get("space")
    return None


def _check_saved_settings(
    collection: Collection, embedding_model: str, embedding_dim: int
) -> None:
    """Compare saved model, dimension and space with the expected values."""
    saved = collection.metadata or {}
    expected = {
        META_EMBEDDING_MODEL: embedding_model,
        META_EMBEDDING_DIM: embedding_dim,
        "space": DISTANCE_SPACE,
    }
    actual = {
        META_EMBEDDING_MODEL: saved.get(META_EMBEDDING_MODEL),
        META_EMBEDDING_DIM: saved.get(META_EMBEDDING_DIM),
        "space": _read_space(collection),
    }
    mismatches = [
        f"{key}: saved {actual[key]!r}, expected {value!r}"
        for key, value in expected.items()
        if actual[key] != value
    ]
    if mismatches:
        message = (
            f"Collection {collection.name!r} does not match the current settings: "
            + "; ".join(mismatches)
        )
        logger.error(message)
        raise VectorStoreError(message)


def open_collection(
    path: Path,
    name: str,
    embedding_model: str,
    embedding_dim: int,
    *,
    create: bool = True,
) -> Collection:
    """Open the chunk collection at ``path``, creating it if it does not exist.

    An existing collection is opened as is and its saved ``embedding_model``,
    ``embedding_dim`` and distance space must equal the expected values; its
    metadata is never changed here. A new collection is created with cosine
    distance and that metadata. The directory is created if missing.
    Telemetry is off and no embedding function is attached.

    With ``create=False`` (read path) neither the directory nor the
    collection is created: a missing one raises
    :class:`CollectionNotFoundError`. The saved settings are checked the same
    way.

    Raises:
        CollectionNotFoundError: ``create=False`` and the directory or the
            collection does not exist.
        VectorStoreError: invalid arguments, the directory or store cannot be
            opened, or the saved settings differ (or are missing).
    """
    if not embedding_model.strip():
        raise VectorStoreError("embedding_model must not be empty")
    if embedding_dim < 1:
        raise VectorStoreError(f"embedding_dim must be >= 1, got {embedding_dim}")
    if not create and not path.is_dir():
        raise CollectionNotFoundError(f"Vector store folder {path} not found")
    try:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
    except (OSError, ChromaError) as exc:
        logger.error("Could not open vector store at %s: %s", path, exc)
        raise VectorStoreError(f"Could not open vector store at {path}: {exc}") from exc

    try:
        try:
            collection = client.get_collection(name, embedding_function=None)
            created = False
        except NotFoundError as exc:
            if not create:
                raise CollectionNotFoundError(
                    f"Collection {name!r} not found at {path}"
                ) from exc
            try:
                collection = client.create_collection(
                    name,
                    configuration={"hnsw": {"space": DISTANCE_SPACE}},
                    metadata={
                        META_EMBEDDING_MODEL: embedding_model,
                        META_EMBEDDING_DIM: embedding_dim,
                    },
                    embedding_function=None,
                )
                created = True
            except UniqueConstraintError:
                # Created by someone else between get and create: open theirs.
                collection = client.get_collection(name, embedding_function=None)
                created = False
    except ChromaError as exc:
        logger.error("Could not open collection %r at %s: %s", name, path, exc)
        raise VectorStoreError(
            f"Could not open collection {name!r} at {path}: {exc}"
        ) from exc

    _check_saved_settings(collection, embedding_model, embedding_dim)
    logger.info(
        "%s collection %r at %s (model=%s, dim=%d, space=%s, %d chunk(s))",
        "Created" if created else "Opened",
        name,
        path,
        embedding_model,
        embedding_dim,
        DISTANCE_SPACE,
        collection.count(),
    )
    return collection


def _collection_model_and_dim(collection: Collection) -> tuple[str, int]:
    """Embedding model and dimension saved in the collection metadata."""
    saved = collection.metadata or {}
    model = saved.get(META_EMBEDDING_MODEL)
    dim = saved.get(META_EMBEDDING_DIM)
    if not isinstance(model, str) or not isinstance(dim, int) or isinstance(dim, bool):
        raise VectorStoreError(
            f"Collection {collection.name!r} has no valid {META_EMBEDDING_MODEL}/"
            f"{META_EMBEDDING_DIM} metadata: {saved!r}"
        )
    return model, dim


def _client_batch_limit(collection: Collection) -> int:
    """Largest batch the collection's client accepts in one call."""
    try:
        return int(collection._client.get_max_batch_size())
    except (AttributeError, ChromaError) as exc:
        raise VectorStoreError(f"Could not read the client's max batch size: {exc}") from exc


def _validate_replace(
    collection: Collection,
    doc_id: str,
    chunks: Sequence[DocumentChunk],
    embeddings: ChunkEmbeddings | None,
    max_batch_size: int | None,
) -> int:
    """Check all input of a replace; return the batch size to use."""
    if not doc_id:
        raise VectorStoreError("doc_id must not be empty")
    model, dim = _collection_model_and_dim(collection)

    wrong_doc = [c.chunk_id for c in chunks if c.doc_id != doc_id]
    if wrong_doc:
        raise VectorStoreError(
            f"{len(wrong_doc)} chunk(s) do not belong to doc_id {doc_id}: "
            f"first {wrong_doc[0]}"
        )
    chunk_ids = [c.chunk_id for c in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise VectorStoreError(f"Duplicate chunk ids for doc_id {doc_id}")

    if embeddings is None:
        if chunks:
            raise VectorStoreError(
                f"{len(chunks)} chunk(s) given for doc_id {doc_id} but no embeddings"
            )
    else:
        if embeddings.model_name != model:
            raise VectorStoreError(
                f"Embeddings come from model {embeddings.model_name!r}, "
                f"collection uses {model!r}"
            )
        vectors = embeddings.vectors
        if embeddings.dimension != dim or vectors.ndim != 2 or vectors.shape[1] != dim:
            raise VectorStoreError(
                f"Embedding dimension {embeddings.dimension} (vectors shape "
                f"{vectors.shape}) does not match collection dimension {dim}"
            )
        if vectors.shape[0] != len(chunks):
            raise VectorStoreError(
                f"{len(chunks)} chunk(s) but {vectors.shape[0]} vector(s) "
                f"for doc_id {doc_id}"
            )
        if list(embeddings.chunk_ids) != chunk_ids:
            raise VectorStoreError(
                f"Embedding chunk ids do not match the chunks (ids or order) "
                f"for doc_id {doc_id}"
            )
        if not np.all(np.isfinite(vectors)):
            raise VectorStoreError(f"Embeddings for doc_id {doc_id} contain NaN or inf")

    if max_batch_size is not None and max_batch_size < 1:
        raise VectorStoreError(f"max_batch_size must be >= 1, got {max_batch_size}")
    limit = _client_batch_limit(collection)
    return limit if max_batch_size is None else min(max_batch_size, limit)


def replace_document_chunks(
    collection: Collection,
    doc_id: str,
    chunks: Sequence[DocumentChunk],
    embeddings: ChunkEmbeddings | None,
    *,
    max_batch_size: int | None = None,
) -> int:
    """Replace all stored chunks of ``doc_id`` with ``chunks``; return the count.

    All input is validated before anything is deleted: every chunk belongs to
    ``doc_id``, chunk ids are unique, the embeddings' model, dimension, row
    count and chunk ids (including order) match the chunks and the collection,
    and all values are finite. ``embeddings`` may be ``None`` only when there
    are no chunks (e.g. a ``needs_ocr`` document), which deletes the document's
    chunks and stores nothing.

    Then every chunk with this ``doc_id`` is deleted and the new chunks are
    upserted in batches of ``min(max_batch_size, client limit)``.

    Raises:
        VectorStoreError: invalid input (the collection is unchanged), or a
            Chroma failure during delete/upsert (the document may then have no
            or only some chunks; re-run the replace).
    """
    batch_size = _validate_replace(collection, doc_id, chunks, embeddings, max_batch_size)
    short_id = doc_id[:LOG_DOC_ID_CHARS]
    filename = chunks[0].filename if chunks else "-"

    try:
        collection.delete(where={"doc_id": doc_id})
        if embeddings is not None:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                collection.upsert(
                    ids=[c.chunk_id for c in batch],
                    embeddings=embeddings.vectors[start : start + batch_size],
                    documents=[c.text for c in batch],
                    metadatas=[c.to_metadata() for c in batch],
                )
    except ChromaError as exc:
        logger.error(
            "Replacing chunks of %s (%s) failed after validation: %s",
            short_id,
            filename,
            exc,
        )
        raise VectorStoreError(
            f"Replacing chunks of doc_id {doc_id} failed: {exc}"
        ) from exc

    logger.info(
        "Stored %d chunk(s) for %s (%s), batch size %d",
        len(chunks),
        short_id,
        filename,
        batch_size,
    )
    return len(chunks)


def count_document_chunks(collection: Collection, doc_id: str) -> int:
    """Number of chunks stored for ``doc_id``.

    Raises:
        VectorStoreError: Chroma fails to read the collection.
    """
    try:
        result = collection.get(where={"doc_id": doc_id}, include=[])
    except ChromaError as exc:
        raise VectorStoreError(f"Could not count chunks of doc_id {doc_id}: {exc}") from exc
    return len(result["ids"])


def _to_page_chunk(chunk_id: str, text: str | None, metadata: Any) -> PageChunk:
    """Build a :class:`PageChunk` from one row of a ``get`` result."""
    meta = dict(metadata) if metadata else {}
    missing = [key for key in _REQUIRED_CHUNK_KEYS if key not in meta]
    if missing or text is None:
        raise VectorStoreError(f"Stored chunk {chunk_id} lacks {missing or ['text']}")
    return PageChunk(
        chunk_id=chunk_id,
        doc_id=str(meta["doc_id"]),
        filename=str(meta["filename"]),
        page_number=int(meta["page_number"]),
        chunk_index=int(meta["chunk_index"]),
        text=text,
    )


def get_page_chunks(collection: Collection, doc_id: str, page_number: int) -> list[PageChunk]:
    """Every stored chunk of one document page, in ``chunk_index`` order.

    Uses ``collection.get`` with
    ``where={"$and": [{"doc_id": {"$eq": doc_id}}, {"page_number": {"$eq": page_number}}]}``
    (no embedding, no similarity). Chroma's ``get`` has no guaranteed order,
    so the chunks are sorted by ``chunk_index`` here. An unknown document or
    a page without chunks gives ``[]``. ``page_number`` must be an int:
    the metadata stores it as an int, so ``"3"`` would match nothing.

    Raises:
        VectorStoreError: ``doc_id`` is not a non-empty string or
            ``page_number`` is not an int >= 1 (checked before any Chroma
            call); Chroma fails; a stored row lacks metadata or text, or does
            not belong to the requested document page.
    """
    if not isinstance(doc_id, str) or not doc_id:
        raise VectorStoreError(f"doc_id must be a non-empty string, got {doc_id!r}")
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise VectorStoreError(f"page_number must be an int >= 1, got {page_number!r}")
    where: dict[str, Any] = {
        "$and": [{"doc_id": {"$eq": doc_id}}, {"page_number": {"$eq": page_number}}]
    }
    try:
        result = collection.get(where=where, include=["documents", "metadatas"])
    except ChromaError as exc:
        logger.error(
            "Fetching chunks of %s page %d failed: %s",
            doc_id[:LOG_DOC_ID_CHARS],
            page_number,
            exc,
        )
        raise VectorStoreError(
            f"Could not fetch chunks of doc_id {doc_id} page {page_number}: {exc}"
        ) from exc

    ids = result["ids"] or []
    documents = result["documents"] or []
    metadatas = result["metadatas"] or []
    if not len(ids) == len(documents) == len(metadatas):
        raise VectorStoreError("Chroma returned get lists of different lengths")
    chunks = [
        _to_page_chunk(chunk_id, text, meta)
        for chunk_id, text, meta in zip(ids, documents, metadatas)
    ]
    foreign = [c.chunk_id for c in chunks if (c.doc_id, c.page_number) != (doc_id, page_number)]
    if foreign:
        raise VectorStoreError(
            f"Chroma returned {len(foreign)} chunk(s) outside doc_id {doc_id} page "
            f"{page_number}: first {foreign[0]}"
        )
    chunks.sort(key=lambda c: c.chunk_index)
    return chunks


def _to_retrieved(
    chunk_id: str, text: str | None, metadata: dict[str, Any] | None, distance: float
) -> RetrievedChunk:
    """Build a :class:`RetrievedChunk` from one row of a query result."""
    meta = metadata or {}
    missing = [key for key in _REQUIRED_CHUNK_KEYS if key not in meta]
    if missing or text is None:
        raise VectorStoreError(
            f"Stored chunk {chunk_id} lacks {missing or ['text']}"
        )
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=str(meta["doc_id"]),
        filename=str(meta["filename"]),
        page_number=int(meta["page_number"]),
        chunk_index=int(meta["chunk_index"]),
        text=text,
        distance=float(distance),
        similarity=1.0 - float(distance),
    )


def query_by_embedding(
    collection: Collection,
    query_vector: npt.ArrayLike,
    top_k: int,
    *,
    doc_ids: AbstractSet[str] | None = None,
) -> list[RetrievedChunk]:
    """Return up to ``top_k`` chunks nearest to ``query_vector``, nearest first.

    Uses ``query_embeddings`` only (no text embedding in Chroma). A ``top_k``
    larger than the collection (or than the matching chunks) returns every
    match; an empty collection returns ``[]``.

    ``doc_ids`` (optional) restricts the search to chunks of these documents
    inside Chroma (``where={"doc_id": {"$in": [...]}}``), so the ``top_k``
    nearest chunks are taken among the allowed documents only. It must not be
    empty: Chroma 1.5.9 rejects an empty ``$in`` list with a ``ValueError``,
    so a caller with no allowed documents must not query at all.

    Raises:
        VectorStoreError: ``top_k < 1``, ``doc_ids`` is empty or holds a
            non-string or empty value, the vector is not 1-D with the
            collection's dimension or is not finite, or the query fails.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise VectorStoreError(f"top_k must be an int >= 1, got {top_k!r}")
    where: dict[str, Any] | None = None
    if doc_ids is not None:
        if not doc_ids:
            raise VectorStoreError("doc_ids must not be empty (query nothing instead)")
        if not all(isinstance(d, str) and d for d in doc_ids):
            raise VectorStoreError("doc_ids must be non-empty strings")
        where = {"doc_id": {"$in": sorted(doc_ids)}}
    _, dim = _collection_model_and_dim(collection)
    vector = np.asarray(query_vector, dtype=np.float32)
    if vector.shape != (dim,):
        raise VectorStoreError(
            f"Query vector shape {vector.shape} does not match collection dimension {dim}"
        )
    if not np.all(np.isfinite(vector)):
        raise VectorStoreError("Query vector contains NaN or inf")

    try:
        result = collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except ChromaError as exc:
        logger.error("Vector query failed: %s", exc)
        raise VectorStoreError(f"Vector query failed: {exc}") from exc

    ids = result["ids"][0] if result["ids"] else []
    if not ids:
        return []
    documents = (result["documents"] or [[]])[0]
    metadatas = (result["metadatas"] or [[]])[0]
    distances = (result["distances"] or [[]])[0]
    if not len(ids) == len(documents) == len(metadatas) == len(distances):
        raise VectorStoreError("Chroma returned query lists of different lengths")
    retrieved = [
        _to_retrieved(chunk_id, text, dict(meta) if meta else None, dist)
        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances)
    ]
    if any(math.isnan(r.distance) for r in retrieved):
        raise VectorStoreError("Chroma returned a NaN distance")
    retrieved.sort(key=lambda r: r.distance)
    return retrieved
