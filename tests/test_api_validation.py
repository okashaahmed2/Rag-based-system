"""Tests for rag_pipeline.api.validation (pure; tmp folders only)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_pipeline.api.validation import (
    FILE_NOT_FOUND,
    MAX_FILENAME_CHARS,
    InboxFilenameError,
    InboxFileNotFoundError,
    check_filename_format,
    validate_inbox_filename,
)

ACCEPTED = [
    "a.pdf",
    "Lecture07_DecisionTrees.pdf",
    "my notes v2.pdf",  # spaces inside
    "kal ki class ke notes.pdf",  # Roman Urdu
    "Mera Sabaq - Hissa Awwal.pdf",  # Roman Urdu
    "\u0633\u0628\u0642.pdf",  # Urdu script
    "SVM,K-mean,KNN.pdf",
    "SCAN.PDF",
    "report.v2.final.Pdf",
    "CONsole.pdf",  # only the exact device name is reserved
    "COM10.pdf",
    "x" * (MAX_FILENAME_CHARS - 4) + ".pdf",  # exactly 255
]

REJECTED = {
    "empty": "",
    "leading-space": " a.pdf",
    "trailing-space": "a.pdf ",
    "trailing-tab": "a.pdf\t",
    "dotdot-slash": "../a.pdf",
    "dotdot-backslash": "..\\a.pdf",
    "sub-slash": "sub/a.pdf",
    "sub-backslash": "sub\\a.pdf",
    "absolute": "C:\\x\\a.pdf",
    "drive-relative": "C:a.pdf",
    "stream": "a.pdf:stream",
    "unc": "\\\\server\\share\\a.pdf",
    "txt": "a.txt",
    "no-suffix": "a",
    "only-suffix": ".pdf",
    "trailing-dot": "a.pdf.",
    "dot": ".",
    "dotdot": "..",
    "CON": "CON.pdf",
    "nul-lower": "nul.PDF",
    "com1": "com1.pdf",
    "lpt9": "LPT9.pdf",
    "prn-two-dots": "PRN.tar.pdf",
    "aux-space-before-dot": "AUX .pdf",
    "nul-char": "a\u0000b.pdf",
    "newline": "a\nb.pdf",
    "del-char": "a\u007fb.pdf",
    "c1-control": "a\u0085b.pdf",
    "256-chars": "x" * (MAX_FILENAME_CHARS - 3) + ".pdf",
}


@pytest.mark.parametrize("name", ACCEPTED)
def test_accepted_names_return_the_inbox_path(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_bytes(b"%PDF-1.7")

    path = validate_inbox_filename(name, tmp_path)

    assert path == tmp_path / name
    assert path.parent == tmp_path


@pytest.mark.parametrize("name", list(REJECTED.values()), ids=list(REJECTED))
def test_rejected_names_raise_the_format_error(tmp_path: Path, name: str) -> None:
    with pytest.raises(InboxFilenameError) as info:
        validate_inbox_filename(name, tmp_path)

    message = str(info.value)
    assert message and str(tmp_path) not in message
    if len(name) > 5:  # the message never echoes the rejected name (".pdf" is in the hint)
        assert name not in message


def test_format_error_is_raised_before_any_file_system_access(tmp_path: Path) -> None:
    missing_inbox = tmp_path / "does-not-exist"

    with pytest.raises(InboxFilenameError):
        validate_inbox_filename("../a.pdf", missing_inbox)


def test_check_filename_format_accepts_a_plain_name() -> None:
    check_filename_format("a.pdf")  # no exception


def test_missing_file_is_not_found(tmp_path: Path) -> None:
    with pytest.raises(InboxFileNotFoundError) as info:
        validate_inbox_filename("missing.pdf", tmp_path)

    assert str(info.value) == FILE_NOT_FOUND


def test_directory_named_like_a_pdf_is_not_found(tmp_path: Path) -> None:
    (tmp_path / "d.pdf").mkdir()

    with pytest.raises(InboxFileNotFoundError):
        validate_inbox_filename("d.pdf", tmp_path)


def test_missing_inbox_is_not_found(tmp_path: Path) -> None:
    with pytest.raises(InboxFileNotFoundError):
        validate_inbox_filename("a.pdf", tmp_path / "no-inbox")


def test_file_in_a_subfolder_cannot_be_named(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.pdf").write_bytes(b"%PDF-1.7")

    with pytest.raises(InboxFileNotFoundError):
        validate_inbox_filename("a.pdf", tmp_path)
    with pytest.raises(InboxFilenameError):
        validate_inbox_filename("sub/a.pdf", tmp_path)


def test_errors_are_distinct_types() -> None:
    assert not issubclass(InboxFilenameError, InboxFileNotFoundError)
    assert not issubclass(InboxFileNotFoundError, InboxFilenameError)


@pytest.mark.skipif(os.name != "nt", reason="case-insensitive file names are a Windows feature")
def test_windows_returns_the_on_disk_spelling(tmp_path: Path) -> None:
    (tmp_path / "Report.pdf").write_bytes(b"%PDF-1.7")

    path = validate_inbox_filename("report.PDF", tmp_path)

    assert path == tmp_path / "Report.pdf"
    assert path.name == "Report.pdf"
