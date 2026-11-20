"""Ingest the inbox: scan -> ingest new files -> place every file by its status.

:func:`ingest_inbox` scans the inbox with the status store's known ids,
ingests the NEW files with :func:`rag_pipeline.ingestion.ingest_document`, and
then places every scanned file that has a ``doc_id`` according to its
document's CURRENT status (:func:`destination_for`)::

    done                               -> processed_dir
    needs_ocr                          -> needs_ocr_dir
    failed, attempts >= max_attempts   -> failed_dir
    failed (retryable), pending,
    processing, not registered         -> stays in the inbox

A duplicate copy follows its document's status like the original. Because
placement depends only on status, a move that failed (file open in a viewer,
permission problem) is simply retried on the next run, without re-ingesting.

Files are never overwritten and never deleted: :func:`safe_move` picks the
first free name ("name.pdf", "name (1).pdf", "name (2).pdf", ...) and uses a
rename that refuses to replace an existing file.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rag_pipeline.inbox import ScannedFile, ScanStatus, scan_inbox
from rag_pipeline.ingestion import IngestContext, IngestResult, ingest_document
from rag_pipeline.preparation import LOG_DOC_ID_CHARS
from rag_pipeline.status_db import DocumentRecord, DocumentStatus

logger = logging.getLogger(__name__)

#: Highest " (n)" suffix tried before a move gives up.
MAX_NAME_SUFFIX = 9999
REASON_NOT_REGISTERED = "not registered"


@dataclass(frozen=True)
class DestinationFolders:
    """Where files go once their document reaches a final status."""

    processed_dir: Path
    failed_dir: Path
    needs_ocr_dir: Path


@dataclass(frozen=True)
class FileMove:
    """A file moved out of the inbox."""

    source: Path
    destination: Path


@dataclass(frozen=True)
class LeftInInbox:
    """A file that stays in the inbox, and why (status or scan error)."""

    path: Path
    reason: str


@dataclass(frozen=True)
class MoveFailure:
    """A file that should have moved but could not; it stays in the inbox."""

    path: Path
    destination_dir: Path
    error: str


@dataclass(frozen=True)
class InboxRunReport:
    """Everything one :func:`ingest_inbox` run did."""

    ingest_results: tuple[IngestResult, ...]
    moves: tuple[FileMove, ...]
    left_in_inbox: tuple[LeftInInbox, ...]
    move_failures: tuple[MoveFailure, ...]
    scan_errors: tuple[LeftInInbox, ...]


def destination_for(
    record: DocumentRecord | None, max_attempts: int, folders: DestinationFolders
) -> Path | None:
    """Folder a file belongs in for its document's status; ``None`` = stay in inbox."""
    if record is None:
        return None
    if record.status is DocumentStatus.DONE:
        return folders.processed_dir
    if record.status is DocumentStatus.NEEDS_OCR:
        return folders.needs_ocr_dir
    if record.status is DocumentStatus.FAILED and record.attempts >= max_attempts:
        return folders.failed_dir
    return None


def _rename_no_overwrite(source: Path, target: Path) -> None:
    """Rename ``source`` to ``target``; raise FileExistsError if ``target`` exists.

    On Windows ``os.rename`` never replaces an existing file. Elsewhere a hard
    link (which fails if the target exists) followed by removing the old name
    gives the same guarantee; the file's data is never deleted.
    """
    if os.name == "nt":
        os.rename(source, target)
    else:  # pragma: no cover - the project runs on Windows
        os.link(source, target)
        os.unlink(source)


def _candidate(source: Path, dest_dir: Path, n: int) -> Path:
    name = source.name if n == 0 else f"{source.stem} ({n}){source.suffix}"
    return dest_dir / name


def safe_move(source: Path, dest_dir: Path) -> Path:
    """Move ``source`` into ``dest_dir`` under the first free name; return the new path.

    Never overwrites: a name that exists (checked first, and again by the
    rename itself, which raises ``FileExistsError``) is skipped for the next
    " (n)" suffix. The source must be on the same volume as ``dest_dir``.

    Raises:
        OSError: the folder cannot be created, the rename fails for another
            reason (e.g. ``PermissionError`` for a file open elsewhere), or no
            free name was found up to ``MAX_NAME_SUFFIX``. The source is then
            left where it was.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for n in range(MAX_NAME_SUFFIX + 1):
        target = _candidate(source, dest_dir, n)
        if target.exists():
            continue
        try:
            _rename_no_overwrite(source, target)
        except FileExistsError:
            continue  # created between the check and the rename: try the next name
        return target
    raise OSError(f"No free name for {source.name} in {dest_dir} up to ({MAX_NAME_SUFFIX})")


def _place(
    scanned: ScannedFile,
    context: IngestContext,
    folders: DestinationFolders,
    moves: list[FileMove],
    left: list[LeftInInbox],
    failures: list[MoveFailure],
) -> None:
    """Move one scanned file according to its document's status (or leave it)."""
    doc_id = scanned.doc_id or ""
    record = context.store.get_document(doc_id)
    dest_dir = destination_for(record, context.max_ingest_attempts, folders)
    if dest_dir is None:
        reason = REASON_NOT_REGISTERED if record is None else record.status.value
        if record is not None and record.status is DocumentStatus.FAILED:
            reason = f"failed, attempts {record.attempts} of {context.max_ingest_attempts}"
        left.append(LeftInInbox(scanned.path, reason))
        return
    try:
        new_path = safe_move(scanned.path, dest_dir)
    except OSError as exc:
        message = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Could not move %s (%s) to %s, it stays in the inbox and is retried next run: %s",
            scanned.filename,
            doc_id[:LOG_DOC_ID_CHARS],
            dest_dir,
            message,
        )
        failures.append(MoveFailure(scanned.path, dest_dir, message))
        return
    logger.info("Moved %s -> %s", scanned.path, new_path)
    moves.append(FileMove(scanned.path, new_path))


def ingest_inbox(
    context: IngestContext,
    inbox_dir: Path,
    processed_dir: Path,
    failed_dir: Path,
    needs_ocr_dir: Path,
) -> InboxRunReport:
    """Scan the inbox, ingest NEW files, then place every file by status.

    NEW files are ingested first (in scan order); duplicates and already-known
    files are not ingested. Placement happens after all ingestion, so a
    duplicate copy follows the status its original reached in this run.

    Raises:
        ValueError: a destination folder is the inbox itself.
        InboxScanError: the inbox is missing or cannot be listed.
        StatusStoreError: the status store failed (as in ingestion).
    """
    folders = DestinationFolders(processed_dir, failed_dir, needs_ocr_dir)
    for folder in (processed_dir, failed_dir, needs_ocr_dir):
        if folder.resolve() == inbox_dir.resolve():
            raise ValueError(f"Destination folder {folder} must not be the inbox")

    known = context.store.known_doc_ids(context.max_ingest_attempts)
    scan = scan_inbox(inbox_dir, known_doc_ids=known)

    results = [ingest_document(file.path, context) for file in scan.new_files]

    moves: list[FileMove] = []
    left: list[LeftInInbox] = []
    failures: list[MoveFailure] = []
    for scanned in scan.files:
        if scanned.status is ScanStatus.ERROR or scanned.doc_id is None:
            continue
        _place(scanned, context, folders, moves, left, failures)
    scan_errors = tuple(
        LeftInInbox(file.path, f"scan error: {file.error}") for file in scan.errors
    )

    report = InboxRunReport(
        ingest_results=tuple(results),
        moves=tuple(moves),
        left_in_inbox=tuple(left) + scan_errors,
        move_failures=tuple(failures),
        scan_errors=scan_errors,
    )
    outcomes = Counter(r.outcome.value for r in results)
    logger.info(
        "Inbox run on %s: %d file(s) scanned, ingested %s, moved %d, left %d, "
        "move failures %d, scan errors %d",
        inbox_dir,
        len(scan.files),
        dict(sorted(outcomes.items())) or "{}",
        len(moves),
        len(report.left_in_inbox),
        len(failures),
        len(scan_errors),
    )
    return report
