"""Tests for scripts/list_doc_ids.py (read-only filename -> doc_id listing)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pymupdf
import pytest

from rag_pipeline.config import PROJECT_ROOT
from rag_pipeline.extraction import compute_file_doc_id


def _load_script() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "list_doc_ids.py"
    spec = importlib.util.spec_from_file_location("list_doc_ids", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("list_doc_ids", module)
    spec.loader.exec_module(module)
    return module


list_doc_ids = _load_script()


def write_pdf(path: Path, text: str) -> Path:
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), text)
        pdf.save(path)
    return path


def snapshot(folder: Path) -> list[tuple[str, int, int]]:
    return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in folder.iterdir())


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    write_pdf(pdfs / "Lecture07.pdf", "decision trees")
    shutil.copy2(pdfs / "Lecture07.pdf", pdfs / "Renamed copy.PDF")
    write_pdf(pdfs / "paper.pdf", "a research paper")
    (pdfs / "notes.txt").write_text("not a pdf", encoding="utf-8")
    (pdfs / "sub").mkdir()
    write_pdf(pdfs / "sub" / "nested.pdf", "nested")
    return pdfs


def test_renamed_copy_has_same_doc_id(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = snapshot(folder)

    exit_code = list_doc_ids.main([str(folder)])

    out = capsys.readouterr().out
    rows = [line.split("\t") for line in out.splitlines()]
    assert exit_code == 0
    assert [name for name, _ in rows] == ["Lecture07.pdf", "paper.pdf", "Renamed copy.PDF"]
    by_name = dict(rows)
    assert by_name["Lecture07.pdf"] == by_name["Renamed copy.PDF"]
    assert by_name["Lecture07.pdf"] != by_name["paper.pdf"]
    assert by_name["paper.pdf"] == compute_file_doc_id(folder / "paper.pdf")
    assert all(len(doc_id) == 64 for doc_id in by_name.values())
    assert snapshot(folder) == before  # read-only: nothing moved, renamed or changed


def test_missing_folder_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert list_doc_ids.main([str(tmp_path / "missing")]) == 2
    assert "Not a folder" in capsys.readouterr().err


def test_folder_without_pdfs_reports_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert list_doc_ids.main([str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No PDF files" in captured.err


def test_unreadable_file_is_reported_and_others_listed(
    folder: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = list_doc_ids.compute_file_doc_id

    def flaky(path: Path) -> str:
        if path.name == "paper.pdf":
            raise PermissionError("locked")
        return real(path)

    monkeypatch.setattr(list_doc_ids, "compute_file_doc_id", flaky)

    assert list_doc_ids.main([str(folder)]) == 1
    captured = capsys.readouterr()
    assert "paper.pdf" not in captured.out
    assert len(captured.out.splitlines()) == 2
    assert "Cannot read paper.pdf: PermissionError: locked" in captured.err
