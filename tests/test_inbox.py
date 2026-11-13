"""Tests for rag_pipeline.inbox."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from rag_pipeline import inbox
from rag_pipeline.extraction import compute_doc_id
from rag_pipeline.inbox import InboxScanError, ScanStatus, scan_inbox


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_identical_files_one_new_one_duplicate_of_alphabetical_first(
    tmp_path: Path,
) -> None:
    # Created in reverse order to show that order comes from the names.
    _write(tmp_path / "b_copy.pdf", b"same bytes")
    _write(tmp_path / "a_original.pdf", b"same bytes")

    result = scan_inbox(tmp_path)

    assert [(f.filename, f.status) for f in result.files] == [
        ("a_original.pdf", ScanStatus.NEW),
        ("b_copy.pdf", ScanStatus.DUPLICATE_IN_SCAN),
    ]
    duplicate = result.duplicates[0]
    assert duplicate.duplicate_of == "a_original.pdf"
    assert duplicate.doc_id == compute_doc_id(b"same bytes")
    assert result.new_files[0].duplicate_of is None


def test_known_doc_id_is_already_known(tmp_path: Path) -> None:
    _write(tmp_path / "known.pdf", b"known bytes")
    _write(tmp_path / "fresh.pdf", b"fresh bytes")

    result = scan_inbox(tmp_path, known_doc_ids={compute_doc_id(b"known bytes")})

    statuses = {f.filename: f.status for f in result.files}
    assert statuses == {
        "fresh.pdf": ScanStatus.NEW,
        "known.pdf": ScanStatus.ALREADY_KNOWN,
    }
    assert [f.filename for f in result.already_known] == ["known.pdf"]


def test_known_doc_id_takes_precedence_over_duplicate(tmp_path: Path) -> None:
    _write(tmp_path / "a.pdf", b"known bytes")
    _write(tmp_path / "b.pdf", b"known bytes")

    result = scan_inbox(tmp_path, known_doc_ids=[compute_doc_id(b"known bytes")])

    assert [f.status for f in result.files] == [
        ScanStatus.ALREADY_KNOWN,
        ScanStatus.ALREADY_KNOWN,
    ]


def test_uppercase_suffix_scanned_and_non_pdfs_ignored(tmp_path: Path) -> None:
    _write(tmp_path / "UPPER.PDF", b"upper")
    _write(tmp_path / "lower.pdf", b"lower")
    _write(tmp_path / ".gitkeep", b"")
    _write(tmp_path / "notes.txt", b"text")
    (tmp_path / "subfolder").mkdir()
    _write(tmp_path / "subfolder" / "nested.pdf", b"nested")
    (tmp_path / "folder.pdf").mkdir()  # a directory with a .pdf name

    result = scan_inbox(tmp_path)

    assert [f.filename for f in result.files] == ["lower.pdf", "UPPER.PDF"]
    assert all(f.status is ScanStatus.NEW for f in result.files)
    assert result.ignored == (".gitkeep", "folder.pdf", "notes.txt", "subfolder")


def test_empty_inbox_gives_empty_result(tmp_path: Path) -> None:
    result = scan_inbox(tmp_path)

    assert result.files == ()
    assert result.ignored == ()
    assert result.new_files == ()
    assert result.duplicates == ()
    assert result.already_known == ()
    assert result.errors == ()


def test_missing_inbox_raises_inbox_scan_error(tmp_path: Path) -> None:
    with pytest.raises(InboxScanError, match="does not exist"):
        scan_inbox(tmp_path / "missing")


def test_inbox_that_is_a_file_raises_inbox_scan_error(tmp_path: Path) -> None:
    not_a_dir = _write(tmp_path / "inbox.pdf", b"x")

    with pytest.raises(InboxScanError, match="is not a folder"):
        scan_inbox(not_a_dir)


@pytest.mark.parametrize(
    "creation_order",
    [
        ["c.pdf", "B.pdf", "a.pdf"],
        ["a.pdf", "B.pdf", "c.pdf"],
        ["B.pdf", "c.pdf", "a.pdf"],
    ],
)
def test_order_is_case_insensitive_alphabetical_regardless_of_creation(
    tmp_path: Path, creation_order: list[str]
) -> None:
    for name in creation_order:
        _write(tmp_path / name, name.encode())

    result = scan_inbox(tmp_path)

    assert [f.filename for f in result.files] == ["a.pdf", "B.pdf", "c.pdf"]


def test_unreadable_file_is_error_and_scan_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "bad.pdf", b"bad")
    _write(tmp_path / "good.pdf", b"good")
    real_hash = inbox.compute_file_doc_id

    def fake_hash(path: Path, chunk_size: int = 1_048_576) -> str:
        if path.name == "bad.pdf":
            raise PermissionError("access denied")
        return real_hash(path, chunk_size)

    monkeypatch.setattr(inbox, "compute_file_doc_id", fake_hash)

    result = scan_inbox(tmp_path)

    bad, good = result.files
    assert bad.status is ScanStatus.ERROR
    assert bad.doc_id is None
    assert bad.error == "access denied"
    assert good.status is ScanStatus.NEW
    assert result.errors == (bad,)


def test_scan_does_not_modify_files(tmp_path: Path) -> None:
    _write(tmp_path / "a.pdf", b"same")
    _write(tmp_path / "b.pdf", b"same")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    scan_inbox(tmp_path)

    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_logs_warning_per_duplicate_and_info_summary(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(tmp_path / "a.pdf", b"same")
    _write(tmp_path / "b.pdf", b"same")
    _write(tmp_path / "c.pdf", b"same")

    with caplog.at_level(logging.INFO, logger="rag_pipeline.inbox"):
        scan_inbox(tmp_path)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    short_id = compute_doc_id(b"same")[:12]
    assert warnings == [
        f"Duplicate in inbox: b.pdf -> a.pdf (doc_id={short_id})",
        f"Duplicate in inbox: c.pdf -> a.pdf (doc_id={short_id})",
    ]
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1
    assert "new=1, duplicate_in_scan=2, already_known=0, error=0" in infos[0]


def test_status_enum_is_str_based() -> None:
    assert isinstance(ScanStatus.NEW, str)
    assert ScanStatus.DUPLICATE_IN_SCAN == "DUPLICATE_IN_SCAN"
