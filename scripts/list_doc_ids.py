"""Print ``filename<TAB>doc_id`` for every PDF in a folder (read-only).

Used to fill the ``doc_id`` values of the golden evaluation set. Every
``*.pdf`` directly in the folder (case-insensitive, not recursive) is hashed
with :func:`compute_file_doc_id`; files are only read, never moved, renamed
or changed. Lines are sorted by file name. A file that cannot be read is
reported on stderr and the exit code is 1; the other files are still listed.

Usage::

    .venv\\Scripts\\python.exe scripts\\list_doc_ids.py data\\processed
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_pipeline.extraction import compute_file_doc_id


def pdf_files(folder: Path) -> list[Path]:
    """``*.pdf`` files directly in ``folder`` (case-insensitive), sorted by name."""
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.lower(),
    )


def list_doc_ids(folder: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return ``(name, doc_id)`` pairs and ``(name, error)`` pairs for ``folder``."""
    listed: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []
    for path in pdf_files(folder):
        try:
            listed.append((path.name, compute_file_doc_id(path)))
        except OSError as exc:
            errors.append((path.name, f"{type(exc).__name__}: {exc}"))
    return listed, errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="folder holding the PDFs")
    args = parser.parse_args(argv)
    folder: Path = args.folder
    if not folder.is_dir():
        sys.stderr.write(f"Not a folder: {folder}\n")
        return 2

    listed, errors = list_doc_ids(folder)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # cp1252 console vs. Unicode names
    for name, doc_id in listed:
        sys.stdout.write(f"{name}\t{doc_id}\n")
    for name, error in errors:
        sys.stderr.write(f"Cannot read {name}: {error}\n")
    if not listed and not errors:
        sys.stderr.write(f"No PDF files in {folder}\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
