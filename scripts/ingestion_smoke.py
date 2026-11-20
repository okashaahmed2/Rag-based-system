"""One-off smoke check: ingest_document on every inbox PDF, twice, then query.

Uses the real settings (``.env``, including ``local_files_only``) with the
Chroma folder and the SQLite file redirected to a NEW TEMPORARY directory, so
``data/chroma`` and ``data/status.db`` are never touched. No file is moved.

The temporary directory is not deleted here (Windows keeps the Chroma store
locked until the process exits); its path is logged for removal afterwards.

Usage::

    .venv\\Scripts\\python.exe scripts\\ingestion_smoke.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

from rag_pipeline.config import get_settings
from rag_pipeline.embeddings import embed_query
from rag_pipeline.ingestion import (
    IngestContext,
    IngestOutcome,
    IngestResult,
    build_ingest_context,
    ingest_document,
)
from rag_pipeline.vector_store import query_by_embedding

logger = logging.getLogger("ingestion_smoke")

QUERY = "What is a decision tree?"
EXPECTED_TOP_FILE_PART = "Lecture07"
EXPECTED_TOP_PAGE = 4
EXPECTED_TOP_SIMILARITY = 0.838
SIMILARITY_TOLERANCE = 1e-3
EXPECTED_CHROMA_TOTAL = 86
EXPECTED_SQLITE_ROWS = 7
#: Expected outcome and chunk count of run 1, by filename.
EXPECTED_RUN1: dict[str, tuple[IngestOutcome, int | None]] = {
    "LACE_research_paper_2col.pdf": (IngestOutcome.DONE, 24),
    "Lecture07_DecisionTrees.pdf": (IngestOutcome.DONE, 17),
    "not_a_pdf_test.pdf": (IngestOutcome.FAILED, None),
    "partly_blank_test.pdf": (IngestOutcome.DONE, 2),
    "password_protected_test.pdf": (IngestOutcome.FAILED, None),
    "scanned_notes_test.pdf": (IngestOutcome.NEEDS_OCR, 0),
    "SVM,K-mean,KNN.pdf": (IngestOutcome.DONE, 43),
    "svm_copy_renamed.pdf": (IngestOutcome.SKIPPED, 43),
}


def _log_table(title: str, results: list[IngestResult]) -> None:
    logger.info("%s", title)
    logger.info("  %-30s %-9s %6s %8s %7s  %s", "file", "outcome", "chunks", "attempts", "seconds", "error / skip reason")
    for r in results:
        detail = r.skip_reason or r.error_message or ""
        logger.info(
            "  %-30s %-9s %6s %8s %7.2f  %s",
            r.filename,
            r.outcome.value,
            "-" if r.chunk_count is None else r.chunk_count,
            "-" if r.attempts is None else r.attempts,
            r.duration_seconds,
            detail[:110],
        )


def _run(pdfs: list[Path], context: IngestContext) -> list[IngestResult]:
    return [ingest_document(pdf, context) for pdf in pdfs]


def main() -> int:
    """Run both loops and the query; return 0 if every expectation holds."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    base = get_settings()
    temp = Path(tempfile.mkdtemp(prefix="ingestion_smoke_"))
    settings = base.model_copy(
        update={"chroma_path": temp / "chroma", "status_db_path": temp / "status.db"}
    )
    logger.info("Temp dir: %s (data/chroma and data/status.db untouched)", temp)
    logger.info(
        "model=%s local_files_only=%s chunk %d/%d max_attempts=%d",
        settings.embedding_model_name,
        settings.embedding_local_files_only,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.max_ingest_attempts,
    )

    context = build_ingest_context(settings)
    pdfs = sorted(
        (p for p in settings.inbox_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: (p.name.casefold(), p.name),
    )

    run1 = _run(pdfs, context)
    _log_table("Run 1", run1)
    total1 = context.collection.count()
    rows1 = len(context.store.list_documents())
    logger.info("Run 1: Chroma total %d, SQLite rows %d", total1, rows1)

    run2 = _run(pdfs, context)
    _log_table("Run 2", run2)
    total2 = context.collection.count()
    rows2 = len(context.store.list_documents())
    logger.info("Run 2: Chroma total %d, SQLite rows %d", total2, rows2)

    results = query_by_embedding(context.collection, embed_query(QUERY, context.model), 5)
    logger.info("Query %r top-5:", QUERY)
    for rank, r in enumerate(results, start=1):
        logger.info("  %d. %-30s page %3d similarity %.6f", rank, r.filename, r.page_number, r.similarity)

    by_name1 = {r.filename: r for r in run1}
    run1_ok = set(by_name1) == set(EXPECTED_RUN1) and all(
        (by_name1[name].outcome, by_name1[name].chunk_count) == expected
        for name, expected in EXPECTED_RUN1.items()
    )
    run2_ok = all(
        (r.outcome is IngestOutcome.SKIPPED)
        or (r.outcome is IngestOutcome.FAILED and r.attempts == 2)
        for r in run2
    ) and sum(r.outcome is IngestOutcome.FAILED for r in run2) == 2
    top = results[0] if results else None
    checks = {
        "run 1 outcomes and chunk counts as expected": run1_ok,
        f"run 1 Chroma total {EXPECTED_CHROMA_TOTAL}": total1 == EXPECTED_CHROMA_TOTAL,
        f"run 1 SQLite rows {EXPECTED_SQLITE_ROWS}": rows1 == EXPECTED_SQLITE_ROWS,
        "run 2 all SKIPPED, the 2 failures FAILED again with attempts 2": run2_ok,
        f"run 2 Chroma total still {EXPECTED_CHROMA_TOTAL}": total2 == EXPECTED_CHROMA_TOTAL,
        f"run 2 SQLite rows still {EXPECTED_SQLITE_ROWS}": rows2 == EXPECTED_SQLITE_ROWS,
        f"top-1 {EXPECTED_TOP_FILE_PART} page {EXPECTED_TOP_PAGE}": top is not None
        and EXPECTED_TOP_FILE_PART in top.filename
        and top.page_number == EXPECTED_TOP_PAGE,
        f"top-1 similarity {EXPECTED_TOP_SIMILARITY} +/- {SIMILARITY_TOLERANCE}": top is not None
        and abs(top.similarity - EXPECTED_TOP_SIMILARITY) <= SIMILARITY_TOLERANCE,
    }
    for name, ok in checks.items():
        logger.info("%s: %s", "PASS" if ok else "FAIL", name)
    logger.info("Remove the temp dir after this process exits: %s", temp)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
