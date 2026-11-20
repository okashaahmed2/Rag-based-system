"""Document status register in SQLite (stdlib ``sqlite3``).

One row per document (``doc_id`` = SHA-256 of the file bytes, so no separate
file hash is stored) records where the document is in its lifecycle::

    (new)            -> pending
    pending | failed -> processing   (attempts += 1, error cleared)
    processing       -> done (chunk_count) | needs_ocr (chunk_count 0)
                        | failed (error_message)

Any other transition raises :class:`StatusStoreError` and changes nothing:
every transition is a single conditional ``UPDATE ... WHERE doc_id = ? AND
status IN (...)`` inside an immediate transaction.

Every method opens its own short-lived connection, closed with
:func:`contextlib.closing` (``with sqlite3.connect()`` only ends the
transaction, it does not close the connection, which keeps the file locked on
Windows). Only parameterized queries are used.

The system is single-process: a document still ``processing`` at startup was
interrupted, and :meth:`StatusStore.recover_interrupted` marks it ``failed``.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

#: Schema version stored in ``PRAGMA user_version``.
SCHEMA_VERSION = 1
#: Seconds a connection waits for a lock held by another connection.
BUSY_TIMEOUT_SECONDS = 5.0
#: Longest error message stored; longer messages are trimmed.
MAX_ERROR_MESSAGE_CHARS = 2000
#: Error message written by :meth:`StatusStore.recover_interrupted`.
INTERRUPTED_MESSAGE = "interrupted: process stopped during processing"
#: Number of leading ``doc_id`` characters shown in log lines.
LOG_DOC_ID_CHARS = 12

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        status TEXT NOT NULL
            CHECK (status IN ('pending', 'processing', 'done', 'failed', 'needs_ocr')),
        error_message TEXT,
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        chunk_count INTEGER CHECK (chunk_count IS NULL OR chunk_count >= 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status)",
)
_SET_SCHEMA_VERSION = "PRAGMA user_version = 1"

# Every statement is a complete literal; values are always bound parameters.
_SELECT_ONE = (
    "SELECT doc_id, filename, status, error_message, attempts, chunk_count, "
    "created_at, updated_at FROM documents WHERE doc_id = ?"
)
_SELECT_ALL = (
    "SELECT doc_id, filename, status, error_message, attempts, chunk_count, "
    "created_at, updated_at FROM documents ORDER BY created_at, doc_id"
)
_SELECT_BY_STATUS = (
    "SELECT doc_id, filename, status, error_message, attempts, chunk_count, "
    "created_at, updated_at FROM documents WHERE status = ? ORDER BY created_at, doc_id"
)
_INSERT_PENDING = (
    "INSERT INTO documents (doc_id, filename, status, attempts, created_at, updated_at) "
    "VALUES (?, ?, 'pending', 0, ?, ?) ON CONFLICT (doc_id) DO NOTHING"
)
_UPDATE_TO_PROCESSING = (
    "UPDATE documents SET status = 'processing', attempts = attempts + 1, "
    "error_message = NULL, chunk_count = NULL, updated_at = ? "
    "WHERE doc_id = ? AND status IN ('pending', 'failed')"
)
_UPDATE_TO_DONE = (
    "UPDATE documents SET status = 'done', chunk_count = ?, error_message = NULL, "
    "updated_at = ? WHERE doc_id = ? AND status IN ('processing')"
)
_UPDATE_TO_NEEDS_OCR = (
    "UPDATE documents SET status = 'needs_ocr', chunk_count = 0, error_message = NULL, "
    "updated_at = ? WHERE doc_id = ? AND status IN ('processing')"
)
_UPDATE_TO_FAILED = (
    "UPDATE documents SET status = 'failed', error_message = ?, chunk_count = NULL, "
    "updated_at = ? WHERE doc_id = ? AND status IN ('processing')"
)
_SELECT_PROCESSING = "SELECT doc_id, filename FROM documents WHERE status = 'processing'"
_RECOVER_PROCESSING = (
    "UPDATE documents SET status = 'failed', error_message = ?, chunk_count = NULL, "
    "updated_at = ? WHERE status = 'processing'"
)
_SELECT_KNOWN = (
    "SELECT doc_id FROM documents WHERE status IN ('done', 'needs_ocr', 'processing') "
    "OR (status = 'failed' AND attempts >= ?)"
)


class StatusStoreError(Exception):
    """Raised for invalid transitions, invalid input or database failures."""


class DocumentStatus(str, Enum):
    """Lifecycle status of one document."""

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    NEEDS_OCR = "needs_ocr"


@dataclass(frozen=True)
class DocumentRecord:
    """One row of the ``documents`` table.

    Attributes:
        doc_id: SHA-256 hex digest of the PDF bytes.
        filename: Base name of the PDF when it was registered.
        status: Current lifecycle status.
        error_message: Last failure reason (``failed`` only), else ``None``.
        attempts: Number of times processing was started.
        chunk_count: Chunks stored (``done``; 0 for ``needs_ocr``), else ``None``.
        created_at: Registration time (UTC, timezone-aware).
        updated_at: Time of the last change (UTC, timezone-aware).
    """

    doc_id: str
    filename: str
    status: DocumentStatus
    error_message: str | None
    attempts: int
    chunk_count: int | None
    created_at: datetime
    updated_at: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise StatusStoreError(f"Stored timestamp without offset: {value!r}")
    return parsed.astimezone(timezone.utc)


def _to_record(row: tuple[object, ...]) -> DocumentRecord:
    doc_id, filename, status, error, attempts, chunks, created, updated = row
    return DocumentRecord(
        doc_id=str(doc_id),
        filename=str(filename),
        status=DocumentStatus(status),
        error_message=None if error is None else str(error),
        attempts=int(attempts),  # type: ignore[call-overload]
        chunk_count=None if chunks is None else int(chunks),  # type: ignore[call-overload]
        created_at=_parse_time(str(created)),
        updated_at=_parse_time(str(updated)),
    )


class StatusStore:
    """Document status register backed by one SQLite file.

    Creating the object touches nothing; call :meth:`initialize` once before
    use. ``clock`` returns the current time (timezone-aware) and is meant for
    tests; the default is the real UTC time.
    """

    def __init__(self, path: Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self._path = path
        self._clock = clock if clock is not None else _utc_now

    @property
    def path(self) -> Path:
        """Location of the SQLite file."""
        return self._path

    # --- connection handling ---------------------------------------------------

    def _now(self) -> str:
        now = self._clock()
        if now.tzinfo is None:
            raise StatusStoreError("clock returned a naive datetime (no timezone)")
        return now.astimezone(timezone.utc).isoformat(timespec="microseconds")

    @contextmanager
    def _transaction(self, *, create: bool = False) -> Iterator[sqlite3.Connection]:
        """Open a connection, run one immediate transaction, always close it.

        Without ``create`` the file must already exist (it is opened read-write
        via a URI), so a store that was never initialized raises instead of
        silently creating an empty database.
        """
        try:
            if create:
                conn = sqlite3.connect(
                    self._path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None
                )
            else:
                conn = sqlite3.connect(
                    self._path.resolve().as_uri() + "?mode=rw",
                    timeout=BUSY_TIMEOUT_SECONDS,
                    isolation_level=None,
                    uri=True,
                )
        except sqlite3.Error as exc:
            logger.error("Cannot open status database %s: %s", self._path, exc)
            raise StatusStoreError(
                f"Cannot open status database {self._path} "
                f"(not initialized or not accessible): {exc}"
            ) from exc

        with closing(conn):
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except BaseException as exc:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                if isinstance(exc, sqlite3.Error):
                    logger.error("Status database error in %s: %s", self._path, exc)
                    raise StatusStoreError(
                        f"Status database error in {self._path}: {exc}"
                    ) from exc
                raise

    @staticmethod
    def _fetch(conn: sqlite3.Connection, doc_id: str) -> DocumentRecord | None:
        row = conn.execute(_SELECT_ONE, (doc_id,)).fetchone()
        return None if row is None else _to_record(row)

    # --- schema ----------------------------------------------------------------

    def initialize(self) -> None:
        """Create the folder, table, index and schema version (idempotent).

        Raises:
            StatusStoreError: the database has a newer schema version, or the
                folder/file cannot be created or opened.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StatusStoreError(
                f"Cannot create folder for status database {self._path}: {exc}"
            ) from exc
        with self._transaction(create=True) as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise StatusStoreError(
                    f"Status database {self._path} has schema version {version}, "
                    f"this code supports up to {SCHEMA_VERSION}"
                )
            for statement in _SCHEMA:
                conn.execute(statement)
            if version < SCHEMA_VERSION:
                conn.execute(_SET_SCHEMA_VERSION)
        logger.info("Status database ready at %s (schema version %d)", self._path, SCHEMA_VERSION)

    # --- writes ----------------------------------------------------------------

    def register_document(self, doc_id: str, filename: str) -> DocumentRecord:
        """Add a new document as ``pending``; return an existing one unchanged.

        Raises:
            StatusStoreError: empty ``doc_id`` or ``filename``, or database error.
        """
        if not doc_id:
            raise StatusStoreError("doc_id must not be empty")
        if not filename.strip():
            raise StatusStoreError("filename must not be empty")
        now = self._now()
        with self._transaction() as conn:
            cursor = conn.execute(_INSERT_PENDING, (doc_id, filename, now, now))
            record = self._fetch(conn, doc_id)
        if record is None:
            raise StatusStoreError(f"Document {doc_id} missing right after insert")
        if cursor.rowcount == 1:
            logger.info(
                "Document %s (%s): new -> pending", doc_id[:LOG_DOC_ID_CHARS], filename
            )
        else:
            logger.info(
                "Document %s (%s) already registered as %s",
                doc_id[:LOG_DOC_ID_CHARS],
                record.filename,
                record.status.value,
            )
        return record

    def _transition(
        self,
        doc_id: str,
        allowed_from: tuple[DocumentStatus, ...],
        new_status: DocumentStatus,
        sql: str,
        values: tuple[object, ...],
    ) -> DocumentRecord:
        """Run one conditional UPDATE; raise if the current status does not allow it.

        ``sql`` is one of this module's literal ``_UPDATE_TO_*`` statements;
        ``values`` are bound in order and end with ``updated_at, doc_id``.
        ``allowed_from`` must match the statement's ``status IN (...)`` list; it
        is used for the error message only.
        """
        with self._transaction() as conn:
            before = self._fetch(conn, doc_id)
            cursor = conn.execute(sql, values)
            if cursor.rowcount != 1:
                current = "unknown doc_id" if before is None else f"status {before.status.value}"
                raise StatusStoreError(
                    f"Cannot move document {doc_id} to {new_status.value}: {current} "
                    f"(allowed from {', '.join(s.value for s in allowed_from)})"
                )
            after = self._fetch(conn, doc_id)
        if before is None or after is None:
            raise StatusStoreError(f"Document {doc_id} vanished during a transition")
        logger.info(
            "Document %s (%s): %s -> %s (attempts %d)",
            doc_id[:LOG_DOC_ID_CHARS],
            after.filename,
            before.status.value,
            after.status.value,
            after.attempts,
        )
        return after

    def mark_processing(self, doc_id: str) -> DocumentRecord:
        """``pending``/``failed`` -> ``processing``; attempts + 1, error cleared."""
        return self._transition(
            doc_id,
            (DocumentStatus.PENDING, DocumentStatus.FAILED),
            DocumentStatus.PROCESSING,
            _UPDATE_TO_PROCESSING,
            (self._now(), doc_id),
        )

    def mark_done(self, doc_id: str, chunk_count: int) -> DocumentRecord:
        """``processing`` -> ``done`` with the number of stored chunks."""
        if isinstance(chunk_count, bool) or not isinstance(chunk_count, int) or chunk_count < 0:
            raise StatusStoreError(f"chunk_count must be an int >= 0, got {chunk_count!r}")
        return self._transition(
            doc_id,
            (DocumentStatus.PROCESSING,),
            DocumentStatus.DONE,
            _UPDATE_TO_DONE,
            (chunk_count, self._now(), doc_id),
        )

    def mark_needs_ocr(self, doc_id: str) -> DocumentRecord:
        """``processing`` -> ``needs_ocr`` (chunk_count 0)."""
        return self._transition(
            doc_id,
            (DocumentStatus.PROCESSING,),
            DocumentStatus.NEEDS_OCR,
            _UPDATE_TO_NEEDS_OCR,
            (self._now(), doc_id),
        )

    def mark_failed(self, doc_id: str, error_message: str) -> DocumentRecord:
        """``processing`` -> ``failed``; the message is trimmed to 2000 characters.

        Raises:
            StatusStoreError: empty message or invalid transition.
        """
        message = error_message.strip()
        if not message:
            raise StatusStoreError("error_message must not be empty")
        message = message[:MAX_ERROR_MESSAGE_CHARS]
        record = self._transition(
            doc_id,
            (DocumentStatus.PROCESSING,),
            DocumentStatus.FAILED,
            _UPDATE_TO_FAILED,
            (message, self._now(), doc_id),
        )
        logger.info("Document %s failed: %s", doc_id[:LOG_DOC_ID_CHARS], message)
        return record

    def recover_interrupted(self) -> int:
        """Mark every ``processing`` document ``failed`` (attempts unchanged).

        Run once at startup: in this single-process system, a document still
        ``processing`` then was interrupted. Returns the number of documents
        changed.
        """
        now = self._now()
        with self._transaction() as conn:
            stuck = conn.execute(_SELECT_PROCESSING).fetchall()
            cursor = conn.execute(_RECOVER_PROCESSING, (INTERRUPTED_MESSAGE, now))
            changed = cursor.rowcount
        for doc_id, filename in stuck:
            logger.warning(
                "Document %s (%s): processing -> failed (%s)",
                str(doc_id)[:LOG_DOC_ID_CHARS],
                filename,
                INTERRUPTED_MESSAGE,
            )
        logger.info("Recovered %d interrupted document(s)", changed)
        return changed

    # --- reads -----------------------------------------------------------------

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        """The record for ``doc_id``, or ``None`` if it is not registered."""
        with self._transaction() as conn:
            return self._fetch(conn, doc_id)

    def list_documents(self, status: DocumentStatus | None = None) -> list[DocumentRecord]:
        """All records (optionally of one status), oldest registration first."""
        with self._transaction() as conn:
            if status is None:
                rows = conn.execute(_SELECT_ALL).fetchall()
            else:
                rows = conn.execute(
                    _SELECT_BY_STATUS, (DocumentStatus(status).value,)
                ).fetchall()
        return [_to_record(row) for row in rows]

    def known_doc_ids(self, max_attempts: int) -> set[str]:
        """``doc_id`` values the inbox scan must skip.

        Known: ``done``, ``needs_ocr``, ``processing`` (skipped at runtime,
        recovered at startup) and ``failed`` with ``attempts >= max_attempts``.
        Retryable (not returned): ``pending`` and ``failed`` with fewer attempts.

        Raises:
            StatusStoreError: ``max_attempts < 1`` or database error.
        """
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise StatusStoreError(f"max_attempts must be an int >= 1, got {max_attempts!r}")
        with self._transaction() as conn:
            rows = conn.execute(_SELECT_KNOWN, (max_attempts,)).fetchall()
        return {str(row[0]) for row in rows}
