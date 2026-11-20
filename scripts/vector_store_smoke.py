"""One-off smoke check: real PDFs -> real embeddings -> ChromaDB -> query.

Prepares every PDF in the configured inbox, embeds the chunks with the real
model (settings from ``.env``, including ``local_files_only``), replaces each
document in a collection in a NEW TEMPORARY directory (never ``data/chroma``),
then runs a sanity query and re-runs one replace to check idempotency.

The temporary store is not deleted here: Windows keeps it locked until the
process exits (known Chroma limitation); its path is logged for removal.

Usage::

    .venv\\Scripts\\python.exe scripts\\vector_store_smoke.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path

from rag_pipeline.config import get_settings
from rag_pipeline.embeddings import (
    embed_chunks,
    embed_query,
    load_embedding_model,
    model_dimension,
)
from rag_pipeline.extraction import PdfExtractionError
from rag_pipeline.preparation import PreparedDocument, prepare_document
from rag_pipeline.vector_store import (
    count_document_chunks,
    open_collection,
    query_by_embedding,
    replace_document_chunks,
)

logger = logging.getLogger("vector_store_smoke")

QUERY = "What is a decision tree?"
TOP_K = 5
EXPECTED_TOP_FILE_PART = "Lecture07"
EXPECTED_TOP_PAGE = 4
EXPECTED_TOP_SIMILARITY = 0.838
SIMILARITY_TOLERANCE = 1e-3
#: Chunks prepared from the inbox at 600/120 (Phase 4), counting the renamed copy.
EXPECTED_TOTAL_CHUNKS = 129


def main() -> int:
    """Run the smoke check; return 0 if every expectation holds."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    settings = get_settings()
    store = Path(tempfile.mkdtemp(prefix="vector_store_smoke_")) / "chroma"
    logger.info("Temp store: %s (data/chroma untouched)", store)
    logger.info(
        "Settings: model=%s local_files_only=%s chunk %d/%d collection=%s",
        settings.embedding_model_name,
        settings.embedding_local_files_only,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.chroma_collection_name,
    )

    model = load_embedding_model(
        settings.embedding_model_name, settings.embedding_local_files_only
    )
    collection = open_collection(
        store,
        settings.chroma_collection_name,
        settings.embedding_model_name,
        model_dimension(model),
    )

    pdfs = sorted(
        (p for p in settings.inbox_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.lower(),
    )
    prepared_docs: dict[str, PreparedDocument] = {}
    embeddings_by_doc = {}
    submitted = 0
    for pdf in pdfs:
        try:
            prepared = prepare_document(pdf, settings)
        except PdfExtractionError as exc:
            logger.info("SKIPPED %-45s %s", pdf.name, exc)
            continue
        started = time.perf_counter()
        embeddings = embed_chunks(
            prepared.chunks,
            model,
            settings.embedding_batch_size,
            model_name=settings.embedding_model_name,
        )
        stored = replace_document_chunks(
            collection, prepared.doc_id, prepared.chunks, embeddings
        )
        submitted += stored
        logger.info(
            "%-45s doc_id %s chunks %3d needs_ocr %-5s stored %3d in collection %3d (%.2f s)",
            pdf.name,
            prepared.doc_id[:12],
            len(prepared.chunks),
            prepared.needs_ocr,
            stored,
            count_document_chunks(collection, prepared.doc_id),
            time.perf_counter() - started,
        )
        if prepared.doc_id in prepared_docs:
            logger.info(
                "  duplicate of %s (same doc_id -> same chunk ids)",
                prepared_docs[prepared.doc_id].filename,
            )
        prepared_docs.setdefault(prepared.doc_id, prepared)
        embeddings_by_doc.setdefault(prepared.doc_id, embeddings)

    unique_total = sum(len(p.chunks) for p in prepared_docs.values())
    total = collection.count()
    logger.info(
        "Chunks submitted %d (incl. duplicates), unique documents %d, unique chunks %d, "
        "collection count %d",
        submitted,
        len(prepared_docs),
        unique_total,
        total,
    )

    query_vector = embed_query(QUERY, model)
    results = query_by_embedding(collection, query_vector, TOP_K)
    logger.info("Query %r top-%d:", QUERY, TOP_K)
    for rank, r in enumerate(results, start=1):
        logger.info(
            "  %d. %-45s page %3d similarity %.6f distance %.6f",
            rank,
            r.filename,
            r.page_number,
            r.similarity,
            r.distance,
        )

    rerun_doc = max(prepared_docs.values(), key=lambda p: len(p.chunks))
    before = count_document_chunks(collection, rerun_doc.doc_id)
    replace_document_chunks(
        collection, rerun_doc.doc_id, rerun_doc.chunks, embeddings_by_doc[rerun_doc.doc_id]
    )
    after = count_document_chunks(collection, rerun_doc.doc_id)
    logger.info(
        "Re-run replace for %s: %d -> %d chunks, collection %d -> %d",
        rerun_doc.filename,
        before,
        after,
        total,
        collection.count(),
    )

    top = results[0] if results else None
    checks = {
        f"chunks submitted (incl. duplicate copy) == {EXPECTED_TOTAL_CHUNKS}": submitted
        == EXPECTED_TOTAL_CHUNKS,
        "collection count == unique chunks (duplicate copy adds none)": total == unique_total
        and total < submitted,
        f"top-1 is {EXPECTED_TOP_FILE_PART} page {EXPECTED_TOP_PAGE}": top is not None
        and EXPECTED_TOP_FILE_PART in top.filename
        and top.page_number == EXPECTED_TOP_PAGE,
        f"top-1 similarity {EXPECTED_TOP_SIMILARITY} +/- {SIMILARITY_TOLERANCE}": top is not None
        and abs(top.similarity - EXPECTED_TOP_SIMILARITY) <= SIMILARITY_TOLERANCE,
        "re-run count unchanged": before == after and collection.count() == total,
    }
    for name, ok in checks.items():
        logger.info("%s: %s", "PASS" if ok else "FAIL", name)
    logger.info("Remove the temp store after this process exits: %s", store.parent)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
