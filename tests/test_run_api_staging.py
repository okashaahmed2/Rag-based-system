"""Tests for scripts/run_api_staging.py (tmp_path project roots only, no server)."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pytest

from rag_pipeline.config import PROJECT_ROOT


def _load_script() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "run_api_staging.py"
    spec = importlib.util.spec_from_file_location("run_api_staging", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_api_staging", module)
    spec.loader.exec_module(module)
    return module


staging = _load_script()

NOW = datetime(2026, 11, 27, 18, 30, 5)
RUN = "20261127-183005"
PATH_ENV_NAMES = {
    "RAG_INBOX_DIR",
    "RAG_PROCESSED_DIR",
    "RAG_FAILED_DIR",
    "RAG_NEEDS_OCR_DIR",
    "RAG_CHROMA_PATH",
    "RAG_STATUS_DB_PATH",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(folder: Path) -> list[tuple[str, int, int]]:
    return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in folder.iterdir())


def tree(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*")}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A fake project root with a live inbox holding PDFs and non-PDFs."""
    root = tmp_path / "project"
    inbox = root / "data" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "a.pdf").write_bytes(b"%PDF-1.7 alpha")
    (inbox / "B.PDF").write_bytes(b"%PDF-1.7 bravo bytes")
    (inbox / "notes.txt").write_text("not a pdf", encoding="utf-8")
    (inbox / ".gitkeep").write_bytes(b"")
    (inbox / "sub").mkdir()
    (inbox / "sub" / "nested.pdf").write_bytes(b"%PDF nested")
    old = 1_700_000_000
    for p in inbox.iterdir():
        if p.is_file():
            os.utime(p, (old, old))
    return root


class FakeRunner:
    def __init__(self, code: int = 0) -> None:
        self.code = code
        self.calls: list[tuple[list[str], dict[str, str], Path]] = []

    def __call__(self, argv: Sequence[str], env: Mapping[str, str], cwd: Path) -> int:
        self.calls.append((list(argv), dict(env), cwd))
        return self.code


def run_main(project: Path, *argv: str, **kwargs: object) -> int:
    return staging.main(
        list(argv),
        project_root=project,
        live_inbox=project / "data" / "inbox",
        now=lambda: NOW,
        **kwargs,
    )


# --- prepare -------------------------------------------------------------------


def test_prepare_copies_only_pdfs_and_keeps_bytes_and_mtimes(project: Path) -> None:
    inbox = project / "data" / "inbox"
    before = snapshot(inbox)
    before_nested = snapshot(inbox / "sub")

    assert run_main(project, "prepare") == 0

    run_dir = project / "data" / "staging" / RUN
    assert sorted(p.name for p in run_dir.iterdir()) == sorted(staging.RUN_SUBFOLDERS)
    copied = run_dir / "inbox"
    assert sorted(p.name for p in copied.iterdir()) == ["B.PDF", "a.pdf"]
    for name in ("a.pdf", "B.PDF"):
        assert sha256(copied / name) == sha256(inbox / name)
        assert (copied / name).stat().st_mtime_ns == (inbox / name).stat().st_mtime_ns
    for name in ("processed", "failed", "needs_ocr"):
        assert list((run_dir / name).iterdir()) == []
    assert not (run_dir / "chroma").exists()
    assert not (run_dir / "status.db").exists()
    assert snapshot(inbox) == before
    assert snapshot(inbox / "sub") == before_nested


def test_prepare_prints_run_folder_and_copied_names(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_main(project, "prepare") == 0
    out = capsys.readouterr().out
    assert f"Run folder: {project / 'data' / 'staging' / RUN}" in out
    assert "copied: a.pdf" in out
    assert "copied: B.PDF" in out
    assert "Copied 2 PDF(s)" in out
    assert "notes.txt" not in out


def test_prepare_refuses_existing_run_folder(project: Path) -> None:
    run_dir = project / "data" / "staging" / RUN
    run_dir.mkdir(parents=True)
    marker = run_dir / "keep.txt"
    marker.write_text("x", encoding="utf-8")

    assert run_main(project, "prepare") == 1
    assert tree(run_dir) == {"keep.txt"}
    assert marker.read_text(encoding="utf-8") == "x"


def test_prepare_twice_same_second_refused_and_nothing_removed(project: Path) -> None:
    assert run_main(project, "prepare") == 0
    before = tree(project)
    assert run_main(project, "prepare") == 1
    assert tree(project) == before


def test_prepare_missing_live_inbox_exits_1_and_creates_nothing(tmp_path: Path) -> None:
    root = tmp_path / "empty_project"
    root.mkdir()
    assert run_main(root, "prepare") == 1
    assert list(root.iterdir()) == []


def test_prepare_empty_inbox_creates_run_with_zero_copies(tmp_path: Path) -> None:
    root = tmp_path / "p"
    (root / "data" / "inbox").mkdir(parents=True)
    assert run_main(root, "prepare") == 0
    assert list((root / "data" / "staging" / RUN / "inbox").iterdir()) == []


# --- serve: run selection ------------------------------------------------------


def make_runs(project: Path, *names: str) -> None:
    for name in names:
        (project / "data" / "staging" / name).mkdir(parents=True)


def test_serve_without_run_exits_1(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runner = FakeRunner()
    assert run_main(project, "serve", "--print-only", runner=runner) == 1
    assert "run 'prepare' first" in capsys.readouterr().err
    assert runner.calls == []


def test_serve_with_empty_staging_root_exits_1(project: Path) -> None:
    (project / "data" / "staging").mkdir(parents=True)
    assert run_main(project, "serve", "--print-only") == 1


def test_serve_picks_newest_run(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_runs(project, "20261127-100000", "20261127-180000", "20261126-235959", "zz-notes")
    assert run_main(project, "serve", "--print-only") == 0
    out = capsys.readouterr().out
    assert "STAGING RUN: 20261127-180000" in out


def test_serve_run_option_chooses_named_run(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_runs(project, "20261127-100000", "20261127-180000")
    assert run_main(project, "serve", "--run", "20261127-100000", "--print-only") == 0
    out = capsys.readouterr().out
    assert "STAGING RUN: 20261127-100000" in out
    run_dir = project / "data" / "staging" / "20261127-100000"
    assert f"RAG_INBOX_DIR={run_dir / 'inbox'}" in out


@pytest.mark.parametrize("name", ["missing", "..", "../..", "20261127-100000/inbox"])
def test_serve_unknown_or_escaping_run_exits_1(project: Path, name: str) -> None:
    make_runs(project, "20261127-100000")
    (project / "data" / "staging" / "20261127-100000" / "inbox").mkdir()
    assert run_main(project, "serve", "--run", name, "--print-only") == 1


# --- serve: overrides and safety ----------------------------------------------


def test_overrides_are_exactly_the_expected_rag_values(tmp_path: Path) -> None:
    run_dir = tmp_path / "data" / "staging" / RUN
    overrides = staging.build_overrides(run_dir)
    assert overrides == {
        "RAG_INBOX_DIR": str(run_dir / "inbox"),
        "RAG_PROCESSED_DIR": str(run_dir / "processed"),
        "RAG_FAILED_DIR": str(run_dir / "failed"),
        "RAG_NEEDS_OCR_DIR": str(run_dir / "needs_ocr"),
        "RAG_CHROMA_PATH": str(run_dir / "chroma"),
        "RAG_STATUS_DB_PATH": str(run_dir / "status.db"),
        "RAG_API_HOST": "127.0.0.1",
        "RAG_API_PORT": "8000",
    }


def test_override_env_names_match_settings_fields() -> None:
    from rag_pipeline.config import Settings

    for field, _ in staging.PATH_OVERRIDES:
        assert field in Settings.model_fields
    assert {staging.env_name(f) for f, _ in staging.PATH_OVERRIDES} == PATH_ENV_NAMES


def test_overrides_are_read_by_settings(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_pipeline.config import Settings

    run_dir = project / "data" / "staging" / RUN
    for name, value in staging.build_overrides(run_dir).items():
        monkeypatch.setenv(name, value)
    settings = Settings(_env_file=None)
    assert settings.inbox_dir == run_dir / "inbox"
    assert settings.processed_dir == run_dir / "processed"
    assert settings.failed_dir == run_dir / "failed"
    assert settings.needs_ocr_dir == run_dir / "needs_ocr"
    assert settings.chroma_path == run_dir / "chroma"
    assert settings.status_db_path == run_dir / "status.db"
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8000


def test_child_env_is_base_plus_overrides_only(project: Path) -> None:
    make_runs(project, RUN)
    runner = FakeRunner(code=0)
    base = {"PATH": "C:\\bin", "SYSTEMROOT": "C:\\Windows"}
    assert run_main(project, "serve", runner=runner, base_env=base) == 0
    (_, env, _) = runner.calls[0]
    run_dir = project / "data" / "staging" / RUN
    assert env == {**base, **staging.build_overrides(run_dir)}
    assert {k for k in env if k.startswith("RAG_")} == PATH_ENV_NAMES | {
        "RAG_API_HOST",
        "RAG_API_PORT",
    }


def test_child_env_overrides_win_over_inherited_rag_values(project: Path) -> None:
    make_runs(project, RUN)
    runner = FakeRunner()
    live = str(project / "data" / "inbox")
    assert run_main(project, "serve", runner=runner, base_env={"RAG_INBOX_DIR": live}) == 0
    (_, env, _) = runner.calls[0]
    assert env["RAG_INBOX_DIR"] == str(project / "data" / "staging" / RUN / "inbox")


def test_check_overrides_accepts_staging_paths(project: Path) -> None:
    run_dir = project / "data" / "staging" / RUN
    live = staging.live_default_paths(project)
    assert staging.check_overrides(staging.build_overrides(run_dir), run_dir, live) == []


def test_live_default_paths_come_from_settings_defaults(project: Path) -> None:
    live = staging.live_default_paths(project)
    data = (project / "data").resolve()
    assert live == {
        "inbox_dir": data / "inbox",
        "processed_dir": data / "processed",
        "failed_dir": data / "failed",
        "needs_ocr_dir": data / "needs_ocr",
        "chroma_path": data / "chroma",
        "status_db_path": data / "status.db",
    }


@pytest.mark.parametrize(
    ("name", "relative"),
    [
        ("RAG_INBOX_DIR", "data/inbox"),
        ("RAG_CHROMA_PATH", "data/chroma/sub"),
        ("RAG_STATUS_DB_PATH", "data/status.db"),
        ("RAG_FAILED_DIR", "data/staging/" + RUN + "/../../failed"),
    ],
)
def test_live_path_override_exits_2_and_never_runs(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    relative: str,
) -> None:
    make_runs(project, RUN)
    real = staging.build_overrides

    def bad_overrides(run_dir: Path) -> dict[str, str]:
        overrides = real(run_dir)
        overrides[name] = str(project / relative)
        return overrides

    monkeypatch.setattr(staging, "build_overrides", bad_overrides)
    runner = FakeRunner()
    assert run_main(project, "serve", runner=runner) == 2
    assert run_main(project, "serve", "--print-only", runner=runner) == 2
    assert runner.calls == []
    assert "REFUSED" in capsys.readouterr().err


def test_override_outside_run_but_not_live_exits_2(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    make_runs(project, RUN, "20261101-000000")
    real = staging.build_overrides

    def other_run(run_dir: Path) -> dict[str, str]:
        overrides = real(run_dir)
        overrides["RAG_CHROMA_PATH"] = str(project / "data" / "staging" / "20261101-000000")
        return overrides

    monkeypatch.setattr(staging, "build_overrides", other_run)
    assert run_main(project, "serve", "--print-only") == 2


def test_missing_override_is_a_problem(project: Path) -> None:
    run_dir = project / "data" / "staging" / RUN
    overrides = staging.build_overrides(run_dir)
    del overrides["RAG_STATUS_DB_PATH"]
    problems = staging.check_overrides(overrides, run_dir, staging.live_default_paths(project))
    assert problems == ["RAG_STATUS_DB_PATH is missing"]


def test_override_equal_to_run_folder_is_a_problem(project: Path) -> None:
    run_dir = project / "data" / "staging" / RUN
    overrides = staging.build_overrides(run_dir)
    overrides["RAG_CHROMA_PATH"] = str(run_dir)
    problems = staging.check_overrides(overrides, run_dir, staging.live_default_paths(project))
    assert len(problems) == 1 and "RAG_CHROMA_PATH" in problems[0]


# --- serve: printing and command --------------------------------------------------


def test_print_only_prints_banner_and_overrides_and_starts_nothing(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_runs(project, RUN)
    runner = FakeRunner()
    assert run_main(project, "serve", "--print-only", runner=runner) == 0
    assert runner.calls == []
    out = capsys.readouterr().out
    assert f"STAGING RUN: {RUN}" in out
    for name, value in staging.build_overrides(project / "data" / "staging" / RUN).items():
        assert f"{name}={value}" in out
    assert "server not started" in out


def test_uvicorn_argv_is_correct() -> None:
    assert staging.build_uvicorn_argv("C:\\py\\python.exe") == [
        "C:\\py\\python.exe",
        "-m",
        "uvicorn",
        "rag_pipeline.api.app:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]


def test_serve_passes_argv_cwd_and_returns_runner_exit_code(project: Path) -> None:
    make_runs(project, RUN)
    runner = FakeRunner(code=3)
    assert run_main(project, "serve", runner=runner, base_env={}) == 3
    assert len(runner.calls) == 1
    argv, _, cwd = runner.calls[0]
    assert argv == staging.build_uvicorn_argv(sys.executable)
    assert cwd == project


def test_run_server_waits_again_after_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.waits = 0

        def wait(self) -> int:
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return 0

    created: list[FakeProcess] = []

    def fake_popen(argv: list[str], env: dict[str, str], cwd: Path) -> FakeProcess:
        created.append(FakeProcess())
        return created[-1]

    monkeypatch.setattr(staging.subprocess, "Popen", fake_popen)
    assert staging.run_server(["x"], {}, Path(".")) == 0
    assert created[0].waits == 2


# --- never deletes --------------------------------------------------------------


def test_no_code_path_deletes_anything(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def grows(action: list[str], **kwargs: object) -> None:
        before = tree(project)
        run_main(project, *action, **kwargs)
        after = tree(project)
        assert before <= after, f"{action} removed {before - after}"

    grows(["serve", "--print-only"])  # no run yet
    grows(["prepare"])
    grows(["prepare"])  # refused: run exists
    grows(["serve", "--print-only"])
    grows(["serve"], runner=FakeRunner(), base_env={})
    grows(["serve", "--run", "missing"])
    monkeypatch.setattr(
        staging,
        "build_overrides",
        lambda run_dir: {"RAG_INBOX_DIR": str(project / "data" / "inbox")},
    )
    grows(["serve"], runner=FakeRunner(), base_env={})
