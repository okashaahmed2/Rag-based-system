"""Validation of a file name sent to ``POST /ingest`` (pure, no store access).

:func:`validate_inbox_filename` accepts only a plain PDF file name that names
an existing regular file directly inside the inbox. Two distinct errors:

* :class:`InboxFilenameError` (HTTP 422): the name itself is not acceptable.
* :class:`InboxFileNotFoundError` (HTTP 404): the name is fine, but no such
  regular file is directly in the inbox.

Error messages never contain the inbox path or the rejected name, so they can
be returned to the client as they are.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

#: Longest accepted file name (NTFS limit per path component).
MAX_FILENAME_CHARS = 255
PDF_SUFFIX = ".pdf"
#: Characters that would make the name a path, a drive or an NTFS stream.
FORBIDDEN_CHARS = frozenset({"/", "\\", ":", "\u0000"})
#: Windows device names; reserved with any extension and in any case.
WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
FILE_NOT_FOUND = "file not found in inbox"


class InboxFilenameError(ValueError):
    """The file name is not a plain, acceptable PDF file name."""


class InboxFileNotFoundError(LookupError):
    """No regular file with that name is directly in the inbox."""


def check_filename_format(name: str) -> None:
    """Raise :class:`InboxFilenameError` if ``name`` is not an acceptable file name."""
    if not name:
        raise InboxFilenameError("filename must not be empty")
    if len(name) > MAX_FILENAME_CHARS:
        raise InboxFilenameError(f"filename must be at most {MAX_FILENAME_CHARS} characters")
    if name != name.strip():
        raise InboxFilenameError("filename must not start or end with whitespace")
    if any(char in FORBIDDEN_CHARS for char in name):
        raise InboxFilenameError("filename must not contain '/', '\\', ':' or NUL")
    if any(unicodedata.category(char) == "Cc" for char in name):
        raise InboxFilenameError("filename must not contain control characters")
    if name in (".", ".."):
        raise InboxFilenameError("filename must be a file name, not '.' or '..'")
    if Path(name).name != name:
        raise InboxFilenameError("filename must be a plain file name without folders")
    if Path(name).suffix.lower() != PDF_SUFFIX:
        raise InboxFilenameError("filename must end with .pdf")
    # Windows also ignores trailing spaces before the extension ("CON .pdf").
    stem = name.split(".", 1)[0].rstrip(" ")
    if stem.upper() in WINDOWS_RESERVED_NAMES:
        raise InboxFilenameError("filename must not be a reserved Windows device name")


def validate_inbox_filename(name: str, inbox_dir: Path) -> Path:
    """The inbox path for ``name``; see the module docstring.

    The returned path is ``inbox_dir / <name as stored on disk>`` (on Windows
    the on-disk spelling can differ from ``name`` in case).

    Raises:
        InboxFilenameError: the name is not acceptable (format).
        InboxFileNotFoundError: the name does not resolve to an existing
            regular file directly inside ``inbox_dir``.
    """
    check_filename_format(name)
    candidate = inbox_dir / name
    try:
        resolved = candidate.resolve()
        inside = resolved.parent == inbox_dir.resolve()
        is_file = inside and candidate.is_file() and resolved.is_file()
    except OSError as exc:  # e.g. a name Windows cannot represent
        raise InboxFileNotFoundError(FILE_NOT_FOUND) from exc
    if not is_file:
        raise InboxFileNotFoundError(FILE_NOT_FOUND)
    return inbox_dir / resolved.name
