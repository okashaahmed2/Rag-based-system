"""Inbox scanning with duplicate detection (detection only).

:func:`scan_inbox` lists the PDFs in the inbox folder, hashes each one into
its ``doc_id`` and classifies it as new, a duplicate within the scan, already
known, or unreadable. It never extracts text and never moves, renames, copies
or deletes files; acting on the result is the caller's job.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from rag_pipeline.extraction import compute_file_doc_id

logger = logging.getLogger(__name__)

PDF_SUFFIX = ".pdf"


class InboxScanError(Exception):
    """Raised when the inbox folder is missing or cannot be listed."""


class ScanStatus(str, Enum):
    """Classification of one PDF found in the inbox."""

    NEW = "NEW"
    DUPLICATE_IN_SCAN = "DUPLICATE_IN_SCAN"
    ALREADY_KNOWN = "ALREADY_KNOWN"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ScannedFile:
    """Result for one PDF in the inbox.

    Attributes:
        path: Full path of the file.
        filename: File name only.
        status: Classification of the file.
        doc_id: SHA-256 hex digest of the file bytes; ``None`` for ``ERROR``.
        duplicate_of: For ``DUPLICATE_IN_SCAN``, the filename of the earlier
            file in this scan with the same ``doc_id``; otherwise ``None``.
        error: For ``ERROR``, the error message; otherwise ``None``.
    """

    path: Path
    filename: str
    status: ScanStatus
    doc_id: str | None = None
    duplicate_of: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class InboxScanResult:
    """Result of scanning the inbox folder."""

    files: tuple[ScannedFile, ...]
    ignored: tuple[str, ...]

    def _with_status(self, status: ScanStatus) -> tuple[ScannedFile, ...]:
        return tuple(file for file in self.files if file.status is status)

    @property
    def new_files(self) -> tuple[ScannedFile, ...]:
        """Files whose ``doc_id`` was not seen before."""
        return self._with_status(ScanStatus.NEW)

    @property
    def duplicates(self) -> tuple[ScannedFile, ...]:
        """Files with the same bytes as an earlier file in this scan."""
        return self._with_status(ScanStatus.DUPLICATE_IN_SCAN)

    @property
    def already_known(self) -> tuple[ScannedFile, ...]:
        """Files whose ``doc_id`` is in ``known_doc_ids``."""
        return self._with_status(ScanStatus.ALREADY_KNOWN)

    @property
    def errors(self) -> tuple[ScannedFile, ...]:
        """Files that could not be read."""
        return self._with_status(ScanStatus.ERROR)


def scan_inbox(
    inbox_dir: Path, known_doc_ids: Collection[str] = ()
) -> InboxScanResult:
    """Scan ``inbox_dir`` (non-recursively) and classify every PDF in it.

    Only regular files with a ``.pdf`` suffix (any case) are scanned; all
    other entries (other files, sub-folders) are listed in ``ignored``. PDFs
    are processed in filename order, case-insensitive, so "earlier file"
    for duplicates is the alphabetically first one.

    Each PDF gets exactly one status, checked in this order:

    * ``ERROR``: the file could not be read.
    * ``ALREADY_KNOWN``: its ``doc_id`` is in ``known_doc_ids``.
    * ``DUPLICATE_IN_SCAN``: an earlier file in this scan has the same
      ``doc_id``.
    * ``NEW``: otherwise.

    Args:
        inbox_dir: Folder to scan.
        known_doc_ids: ``doc_id`` values that are already ingested. This is
            a placeholder for the SQLite lookup planned for Phase 5; until
            then callers pass the ids explicitly (default: none).

    Raises:
        InboxScanError: if ``inbox_dir`` does not exist, is not a folder or
            cannot be listed.
    """
    pdf_paths, ignored = _list_inbox(inbox_dir)
    known = frozenset(known_doc_ids)
    first_file_by_doc_id: dict[str, str] = {}
    files: list[ScannedFile] = []

    for path in pdf_paths:
        scanned = _classify(path, known, first_file_by_doc_id)
        if scanned.status is ScanStatus.DUPLICATE_IN_SCAN:
            logger.warning(
                "Duplicate in inbox: %s -> %s (doc_id=%s)",
                scanned.filename,
                scanned.duplicate_of,
                (scanned.doc_id or "")[:12],
            )
        files.append(scanned)

    result = InboxScanResult(files=tuple(files), ignored=ignored)
    logger.info(
        "Inbox scan of %s: %d PDF(s): new=%d, duplicate_in_scan=%d, "
        "already_known=%d, error=%d; ignored=%d",
        inbox_dir,
        len(result.files),
        len(result.new_files),
        len(result.duplicates),
        len(result.already_known),
        len(result.errors),
        len(result.ignored),
    )
    return result


def _list_inbox(inbox_dir: Path) -> tuple[list[Path], tuple[str, ...]]:
    """Return the sorted PDF paths and the sorted names of ignored entries."""
    if not inbox_dir.is_dir():
        reason = "does not exist" if not inbox_dir.exists() else "is not a folder"
        logger.error("Inbox %s %s", inbox_dir, reason)
        raise InboxScanError(f"Inbox {inbox_dir} {reason}")
    try:
        entries = list(inbox_dir.iterdir())
    except OSError as exc:
        logger.error("Cannot list inbox %s: %s", inbox_dir, exc)
        raise InboxScanError(f"Cannot list inbox {inbox_dir}: {exc}") from exc

    pdf_paths: list[Path] = []
    ignored: list[str] = []
    for entry in entries:
        if entry.suffix.lower() == PDF_SUFFIX and entry.is_file():
            pdf_paths.append(entry)
        else:
            ignored.append(entry.name)

    pdf_paths.sort(key=lambda p: _sort_key(p.name))
    ignored.sort(key=_sort_key)
    return pdf_paths, tuple(ignored)


def _sort_key(name: str) -> tuple[str, str]:
    """Case-insensitive sort key; the original name breaks ties."""
    return (name.casefold(), name)


def _classify(
    path: Path, known: frozenset[str], first_file_by_doc_id: dict[str, str]
) -> ScannedFile:
    """Hash one PDF and classify it; records new ids in ``first_file_by_doc_id``."""
    try:
        doc_id = compute_file_doc_id(path)
    except OSError as exc:
        logger.error("Cannot read inbox file %s: %s", path, exc)
        return ScannedFile(
            path=path, filename=path.name, status=ScanStatus.ERROR, error=str(exc)
        )

    if doc_id in known:
        return ScannedFile(
            path=path, filename=path.name, status=ScanStatus.ALREADY_KNOWN, doc_id=doc_id
        )
    original = first_file_by_doc_id.get(doc_id)
    if original is not None:
        return ScannedFile(
            path=path,
            filename=path.name,
            status=ScanStatus.DUPLICATE_IN_SCAN,
            doc_id=doc_id,
            duplicate_of=original,
        )
    first_file_by_doc_id[doc_id] = path.name
    return ScannedFile(path=path, filename=path.name, status=ScanStatus.NEW, doc_id=doc_id)
