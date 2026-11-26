"""Background job runners for ``POST /ingest/scan`` and ``POST /ingest``, and their summary.

:func:`run_job` (scan) and :func:`run_ingest_job` (one inbox file) run in
FastAPI's worker threadpool (plain ``def`` added with
``BackgroundTasks.add_task``), after the 202 response was sent. Both share
the same lifecycle and error handling::

    queued -> running -> work (runner) -> succeeded (summary)
                                       -> failed ("ExceptionType: message")

A failure to record the job's own state (:class:`StatusStoreError` from the
store) is logged as CRITICAL and never re-raised: there is no caller left to
tell. If recording success fails, the job is marked ``failed`` instead, so it
does not stay ``running`` and block every later scan. A job interrupted by a
process stop stays active in the database and is marked ``failed`` by the
startup recovery of the next process.

Logs contain job ids, file names, counts and error messages only; never
document, chunk or question text.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from rag_pipeline.config import Settings
from rag_pipeline.inbox_ingestion import InboxRunReport
from rag_pipeline.ingestion import IngestContext, IngestOutcome, IngestResult
from rag_pipeline.status_db import StatusStoreError

logger = logging.getLogger(__name__)

#: ``inbox_runner(context, inbox_dir, processed_dir, failed_dir, needs_ocr_dir)``.
InboxRunner = Callable[[IngestContext, Path, Path, Path, Path], InboxRunReport]


class SingleFileRunner(Protocol):
    """``single_file_runner(context, path, processed, failed, needs_ocr, *, inbox_dir)``."""

    def __call__(
        self,
        context: IngestContext,
        path: Path,
        processed_dir: Path,
        failed_dir: Path,
        needs_ocr_dir: Path,
        *,
        inbox_dir: Path,
    ) -> InboxRunReport: ...

#: Most per-file entries kept in a summary (``files_truncated`` says if more existed).
MAX_SUMMARY_FILES = 200
#: Longest per-file error string kept in a summary.
MAX_SUMMARY_ERROR_CHARS = 300


def _trim(text: str, limit: int = MAX_SUMMARY_ERROR_CHARS) -> str:
    return text.strip()[:limit]


def _path_variants(path: Path) -> list[str]:
    """Spellings of ``path`` and its folder to hide, longest first.

    ``str(path)`` and its resolved form, plus their parent folders when those
    are absolute (a relative parent such as ``.`` or ``inbox`` could match
    ordinary words in the message).
    """
    variants = {str(path)}
    parents = [path.parent]
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        logger.warning("Could not resolve %s for the job summary: %s", path.name, exc)
    else:
        variants.add(str(resolved))
        parents.append(resolved.parent)
    variants.update(str(parent) for parent in parents if parent.is_absolute())
    variants.discard(path.name)
    variants.discard("")
    return sorted(variants, key=len, reverse=True)


def _without_paths(text: str, path: Path) -> str:
    """``text`` with every full path / folder spelling of ``path`` replaced by its file name."""
    for variant in _path_variants(path):
        text = text.replace(variant, path.name)
    return text


def _file_entry(result: IngestResult) -> dict[str, Any]:
    """``{filename, outcome, attempts}`` plus ``skip_reason``/``error`` when set.

    The error has the file's full path (and folder) replaced by the file name
    before trimming, so ``/status`` never shows local paths.
    """
    entry: dict[str, Any] = {
        "filename": result.filename,
        "outcome": result.outcome.value,
        "attempts": result.attempts,
    }
    if result.skip_reason is not None:
        entry["skip_reason"] = result.skip_reason
    if result.error_message is not None:
        entry["error"] = _trim(_without_paths(result.error_message, result.path))
    return entry


def summary_from_report(report: InboxRunReport, duration_seconds: float) -> dict[str, Any]:
    """JSON-ready summary of one inbox run (pure function).

    Keys: ``outcomes`` (count per :class:`IngestOutcome`, every outcome
    present), ``moved``, ``left_in_inbox``, ``move_failures``, ``scan_errors``,
    ``duration_seconds`` (rounded to ms), ``files`` (one entry per ingested
    file, in order, at most :data:`MAX_SUMMARY_FILES`) and ``files_truncated``.
    """
    counts = Counter(result.outcome for result in report.ingest_results)
    files = [_file_entry(result) for result in report.ingest_results[:MAX_SUMMARY_FILES]]
    return {
        "outcomes": {outcome.value: counts.get(outcome, 0) for outcome in IngestOutcome},
        "moved": len(report.moves),
        "left_in_inbox": len(report.left_in_inbox),
        "move_failures": len(report.move_failures),
        "scan_errors": len(report.scan_errors),
        "duration_seconds": round(max(duration_seconds, 0.0), 3),
        "files": files,
        "files_truncated": len(report.ingest_results) > MAX_SUMMARY_FILES,
    }


def _mark_failed(context: IngestContext, job_id: str, message: str) -> None:
    try:
        context.store.mark_job_failed(job_id, message)
    except StatusStoreError as exc:
        logger.critical(
            "Job %s: could not record failure (%s); it stays active until the next restart: %s",
            job_id,
            message,
            exc,
        )


def _run_work(
    job_id: str,
    context: IngestContext,
    label: str,
    where: str,
    work: Callable[[], InboxRunReport],
) -> None:
    """The shared job lifecycle: running -> ``work()`` -> succeeded / failed.

    ``label`` names the work in logs ("inbox scan", "ingest"); ``where`` is a
    folder or file name. Never raises for an ordinary exception.
    """
    try:
        context.store.mark_job_running(job_id)
    except StatusStoreError as exc:
        logger.critical("Job %s: could not mark it running, not started: %s", job_id, exc)
        return

    started = time.perf_counter()
    logger.info("Job %s: %s started on %s", job_id, label, where)
    try:
        report = work()
        summary = summary_from_report(report, time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - recorded as the job's failure
        logger.exception(
            "Job %s: %s crashed after %.2f s", job_id, label, time.perf_counter() - started
        )
        _mark_failed(context, job_id, f"{type(exc).__name__}: {exc}")
        return

    try:
        context.store.mark_job_succeeded(job_id, summary)
    except StatusStoreError as exc:
        logger.critical("Job %s: finished but could not record success: %s", job_id, exc)
        _mark_failed(context, job_id, f"could not record success: {exc}")
        return
    logger.info(
        "Job %s: succeeded in %.2f s, outcomes %s",
        job_id,
        summary["duration_seconds"],
        summary["outcomes"],
    )


def run_job(
    job_id: str,
    context: IngestContext,
    settings: Settings,
    inbox_runner: InboxRunner,
) -> None:
    """Run one scan job to completion; never raises for an ordinary exception."""
    _run_work(
        job_id,
        context,
        "inbox scan",
        str(settings.inbox_dir),
        lambda: inbox_runner(
            context,
            settings.inbox_dir,
            settings.processed_dir,
            settings.failed_dir,
            settings.needs_ocr_dir,
        ),
    )


def run_ingest_job(
    job_id: str,
    context: IngestContext,
    settings: Settings,
    single_file_runner: SingleFileRunner,
    filename: str,
) -> None:
    """Run one single-file ingest job (``filename`` in the inbox); never raises for an
    ordinary exception."""
    _run_work(
        job_id,
        context,
        "ingest",
        filename,
        lambda: single_file_runner(
            context,
            settings.inbox_dir / filename,
            settings.processed_dir,
            settings.failed_dir,
            settings.needs_ocr_dir,
            inbox_dir=settings.inbox_dir,
        ),
    )
