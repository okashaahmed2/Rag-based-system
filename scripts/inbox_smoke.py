"""One-off smoke check: ingest_inbox three times on a COPY of data/inbox.

Every PDF in the configured inbox is copied into a NEW TEMPORARY inbox; the
temporary processed / failed / needs_ocr folders, Chroma store and SQLite file
live in the same temporary directory. ``data/inbox`` itself is only read: its
file names, sizes and modification times are compared before and after.
The real model is loaded from settings (``local_files_only`` from ``.env``).

The temporary directory is not deleted here (Windows keeps the Chroma store
locked until the process exits); its path is logged for removal afterwards.

Usage::

    .venv\\Scripts\\python.exe scripts\\inbox_smoke.py
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from pathlib import Path

from rag_pipeline.config import get_settings
from rag_pipeline.embeddings import embed_query
from rag_pipeline.inbox_ingestion import ingest_inbox
from rag_pipeline.ingestion import IngestContext, build_ingest_context
from rag_pipeline.vector_store import query_by_embedding

logger = logging.getLogger("inbox_smoke")

QUERY = "What is a decision tree?"
EXPECTED_TOP_FILE_PART = "Lecture07"
EXPECTED_TOP_PAGE = 4
EXPECTED_TOP_SIMILARITY = 0.838
SIMILARITY_TOLERANCE = 1e-3
EXPECTED_CHROMA = 86
EXPECTED_ROWS = 7
TEXT_PDFS = [
    "LACE_research_paper_2col.pdf",
    "Lecture07_DecisionTrees.pdf",
    "partly_blank_test.pdf",
    "SVM,K-mean,KNN.pdf",
    "svm_copy_renamed.pdf",
]
BAD_PDFS = ["not_a_pdf_test.pdf", "password_protected_test.pdf"]
SCANNED = ["scanned_notes_test.pdf"]


def _fingerprint(folder: Path) -> list[tuple[str, int, int]]:
    """(name, size, mtime_ns) of every entry, sorted."""
    return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in folder.iterdir())


def _names(folder: Path) -> list[str]:
    return sorted((p.name for p in folder.iterdir()), key=str.casefold) if folder.exists() else []


def _log_folders(title: str, folders: dict[str, Path], context: IngestContext) -> None:
    logger.info("%s", title)
    for label, folder in folders.items():
        logger.info("  %-10s %s", label, ", ".join(_names(folder)) or "(empty)")
    attempts = {
        r.filename: r.attempts for r in context.store.list_documents() if r.status.value == "failed"
    }
    logger.info(
        "  Chroma %d chunks, SQLite %d rows, failed attempts %s",
        context.collection.count(),
        len(context.store.list_documents()),
        attempts or "{}",
    )


def main() -> int:
    """Run three inbox passes and the query; return 0 if every check passes."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    base = get_settings()
    real_inbox = base.inbox_dir
    before = _fingerprint(real_inbox)

    temp = Path(tempfile.mkdtemp(prefix="inbox_smoke_"))
    folders = {
        "inbox": temp / "inbox",
        "processed": temp / "processed",
        "needs_ocr": temp / "needs_ocr",
        "failed": temp / "failed",
    }
    folders["inbox"].mkdir()
    for pdf in real_inbox.iterdir():
        if pdf.is_file() and pdf.suffix.lower() == ".pdf":
            shutil.copy2(pdf, folders["inbox"] / pdf.name)
    settings = base.model_copy(
        update={
            "inbox_dir": folders["inbox"],
            "processed_dir": folders["processed"],
            "failed_dir": folders["failed"],
            "needs_ocr_dir": folders["needs_ocr"],
            "chroma_path": temp / "chroma",
            "status_db_path": temp / "status.db",
        }
    )
    logger.info("Temp dir: %s (data/ untouched)", temp)
    context = build_ingest_context(settings)

    snapshots: list[dict[str, list[str]]] = []
    counts: list[tuple[int, int]] = []
    for run_no in (1, 2, 3):
        report = ingest_inbox(
            context,
            settings.inbox_dir,
            settings.processed_dir,
            settings.failed_dir,
            settings.needs_ocr_dir,
        )
        logger.info(
            "Run %d: ingested %d, moved %d, left %d, move failures %d, scan errors %d",
            run_no,
            len(report.ingest_results),
            len(report.moves),
            len(report.left_in_inbox),
            len(report.move_failures),
            len(report.scan_errors),
        )
        _log_folders(f"Folders after run {run_no}", folders, context)
        snapshots.append({label: _names(folder) for label, folder in folders.items()})
        counts.append((context.collection.count(), len(context.store.list_documents())))

    results = query_by_embedding(context.collection, embed_query(QUERY, context.model), 5)
    logger.info("Query %r top-5:", QUERY)
    for rank, r in enumerate(results, start=1):
        logger.info("  %d. %-30s page %3d similarity %.6f", rank, r.filename, r.page_number, r.similarity)

    after = _fingerprint(real_inbox)
    logger.info("data/inbox before: %s", before)
    logger.info("data/inbox after:  %s", after)

    run1, run3 = snapshots[0], snapshots[2]
    top = results[0] if results else None
    by_case = sorted(TEXT_PDFS, key=str.casefold)
    checks = {
        "run 1 processed = 4 text PDFs + svm_copy_renamed": run1["processed"] == by_case,
        "run 1 needs_ocr = scanned": run1["needs_ocr"] == SCANNED,
        "run 1 inbox = invalid + password": run1["inbox"] == BAD_PDFS,
        f"run 1 Chroma {EXPECTED_CHROMA}": counts[0][0] == EXPECTED_CHROMA,
        "run 3 failed = invalid + password": run3["failed"] == BAD_PDFS,
        "run 3 inbox empty": run3["inbox"] == [],
        f"run 3 Chroma {EXPECTED_CHROMA}, SQLite {EXPECTED_ROWS}": counts[2]
        == (EXPECTED_CHROMA, EXPECTED_ROWS),
        f"top-1 {EXPECTED_TOP_FILE_PART} page {EXPECTED_TOP_PAGE}": top is not None
        and EXPECTED_TOP_FILE_PART in top.filename
        and top.page_number == EXPECTED_TOP_PAGE,
        f"top-1 similarity {EXPECTED_TOP_SIMILARITY} +/- {SIMILARITY_TOLERANCE}": top is not None
        and abs(top.similarity - EXPECTED_TOP_SIMILARITY) <= SIMILARITY_TOLERANCE,
        "data/inbox unchanged (names, sizes, mtimes)": before == after,
    }
    for name, ok in checks.items():
        logger.info("%s: %s", "PASS" if ok else "FAIL", name)
    logger.info("Remove the temp dir after this process exits: %s", temp)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
