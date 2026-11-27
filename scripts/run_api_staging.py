"""Run the real FastAPI app against an isolated staging copy of the live inbox.

While n8n workflows are being built, the API must never write the live data
(``data/inbox`` contents, ``data/processed``, ``data/failed``,
``data/needs_ocr``, ``data/chroma``, ``data/status.db``). This script:

* ``prepare``: creates ``data/staging/<YYYYMMDD-HHMMSS>/`` with ``inbox/``,
  ``processed/``, ``failed/`` and ``needs_ocr/`` and copies every ``*.pdf``
  (case-insensitive, non-recursive) of the configured live inbox into its
  ``inbox/`` with :func:`shutil.copy2`. An existing run folder is refused.
  ``chroma/`` and ``status.db`` are created later by the app itself.
* ``serve``: points the ``RAG_*`` path settings at one run folder (default: the
  newest timestamp-named run), refuses to start if any path is outside that run
  or inside a live default path, prints the overrides and starts
  ``python -m uvicorn rag_pipeline.api.app:create_app --factory`` on
  127.0.0.1:8000. ``--print-only`` prints and exits without starting anything.

Nothing is ever deleted or overwritten by this script.

Usage (PowerShell, from the project root)::

    .venv\\Scripts\\python.exe scripts\\run_api_staging.py prepare
    .venv\\Scripts\\python.exe scripts\\run_api_staging.py serve --print-only
    .venv\\Scripts\\python.exe scripts\\run_api_staging.py serve
    .venv\\Scripts\\python.exe scripts\\run_api_staging.py serve --run 20261127-180000

Exit codes: 0 ok, 1 usage/state error (no run, run exists, inbox missing),
2 safety check refused, otherwise the exit code of uvicorn.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rag_pipeline.config import PROJECT_ROOT, Settings

logger = logging.getLogger(__name__)

#: Staging root, relative to the project root.
STAGING_RELATIVE = Path("data") / "staging"
#: Run folder names written by ``prepare`` (local time).
RUN_NAME_FORMAT = "%Y%m%d-%H%M%S"
RUN_NAME_PATTERN = re.compile(r"^\d{8}-\d{6}$")
#: Sub-folders created by ``prepare``.
RUN_SUBFOLDERS: tuple[str, ...] = ("inbox", "processed", "failed", "needs_ocr")
#: Host and port the staging server binds to (this machine only).
STAGING_HOST = "127.0.0.1"
STAGING_PORT = 8000
#: Settings field -> path inside the run folder (env name = "RAG_" + field upper-case).
PATH_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("inbox_dir", "inbox"),
    ("processed_dir", "processed"),
    ("failed_dir", "failed"),
    ("needs_ocr_dir", "needs_ocr"),
    ("chroma_path", "chroma"),
    ("status_db_path", "status.db"),
)
APP_FACTORY = "rag_pipeline.api.app:create_app"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_UNSAFE = 2

#: Starts the server: ``(argv, env, cwd) -> exit code``.
Runner = Callable[[Sequence[str], Mapping[str, str], Path], int]


class StagingError(Exception):
    """A state problem that stops the command with exit code 1."""


@dataclass(frozen=True)
class PrepareResult:
    """Outcome of :func:`prepare`."""

    run_dir: Path
    copied: tuple[str, ...]


def env_name(field: str) -> str:
    """Return the environment variable name of a :class:`Settings` field."""
    return f"RAG_{field.upper()}"


def staging_root(project_root: Path) -> Path:
    """Return ``<project_root>/data/staging``."""
    return project_root / STAGING_RELATIVE


def live_default_paths(project_root: Path) -> dict[str, Path]:
    """Return the code-default live paths (Settings defaults) resolved under ``project_root``."""
    paths: dict[str, Path] = {}
    for field, _ in PATH_OVERRIDES:
        default = Settings.model_fields[field].default
        path = Path(default)
        paths[field] = (path if path.is_absolute() else project_root / path).resolve()
    return paths


def pdf_files(folder: Path) -> list[Path]:
    """Return the ``*.pdf`` files (case-insensitive, non-recursive) of ``folder``, sorted."""
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.lower(),
    )


def prepare(project_root: Path, live_inbox: Path, now: datetime) -> PrepareResult:
    """Create a new run folder and copy the live inbox PDFs into its ``inbox/``.

    Raises:
        StagingError: the live inbox is missing or the run folder already exists.
    """
    if not live_inbox.is_dir():
        raise StagingError(f"live inbox is not a folder: {live_inbox}")
    sources = pdf_files(live_inbox)
    run_dir = staging_root(project_root) / now.strftime(RUN_NAME_FORMAT)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise StagingError(f"run folder already exists, not overwriting: {run_dir}") from exc
    for name in RUN_SUBFOLDERS:
        (run_dir / name).mkdir(exist_ok=False)
    copied: list[str] = []
    for source in sources:
        target = run_dir / "inbox" / source.name
        if target.exists():  # cannot happen in a new folder; never overwrite anyway
            raise StagingError(f"target already exists, not overwriting: {target}")
        shutil.copy2(source, target)
        copied.append(source.name)
    logger.info("Prepared staging run %s with %d PDF(s)", run_dir, len(copied))
    return PrepareResult(run_dir=run_dir, copied=tuple(copied))


def select_run(project_root: Path, run_name: str | None) -> Path:
    """Return the run folder: ``run_name`` or the newest timestamp-named run.

    Raises:
        StagingError: no matching run folder exists.
    """
    root = staging_root(project_root)
    if run_name is not None:
        run_dir = root / run_name
        if run_dir.resolve().parent != root.resolve() or not run_dir.is_dir():
            raise StagingError(f"no staging run {run_name!r} under {root}")
        return run_dir
    runs = (
        sorted(p.name for p in root.iterdir() if p.is_dir() and RUN_NAME_PATTERN.match(p.name))
        if root.is_dir()
        else []
    )
    if not runs:
        raise StagingError(f"no staging run under {root}: run 'prepare' first")
    return root / runs[-1]


def build_overrides(run_dir: Path) -> dict[str, str]:
    """Return the ``RAG_*`` overrides pointing every live path into ``run_dir``."""
    overrides = {env_name(field): str(run_dir / name) for field, name in PATH_OVERRIDES}
    overrides[env_name("api_host")] = STAGING_HOST
    overrides[env_name("api_port")] = str(STAGING_PORT)
    return overrides


def _is_within(path: Path, parent: Path) -> bool:
    """True when ``path`` equals ``parent`` or lies inside it (both resolved)."""
    return path == parent or path.is_relative_to(parent)


def check_overrides(
    overrides: Mapping[str, str], run_dir: Path, live_paths: Mapping[str, Path]
) -> list[str]:
    """Return safety problems of the path overrides (empty list = safe)."""
    problems: list[str] = []
    run_resolved = run_dir.resolve()
    for field, _ in PATH_OVERRIDES:
        name = env_name(field)
        value = overrides.get(name)
        if value is None:
            problems.append(f"{name} is missing")
            continue
        path = Path(value).resolve()
        if path == run_resolved or not path.is_relative_to(run_resolved):
            problems.append(f"{name}={path} is not inside the run folder {run_resolved}")
        for live_field, live in live_paths.items():
            if _is_within(path, live):
                problems.append(f"{name}={path} is a live path ({env_name(live_field)}={live})")
    return problems


def build_child_env(base: Mapping[str, str], overrides: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``base`` with ``overrides`` applied."""
    env = dict(base)
    env.update(overrides)
    return env


def build_uvicorn_argv(python: str) -> list[str]:
    """Return the uvicorn command line for the app factory on the staging host/port."""
    return [
        python,
        "-m",
        "uvicorn",
        APP_FACTORY,
        "--factory",
        "--host",
        STAGING_HOST,
        "--port",
        str(STAGING_PORT),
    ]


def run_server(argv: Sequence[str], env: Mapping[str, str], cwd: Path) -> int:
    """Run the server in the foreground and return its exit code.

    Ctrl+C reaches the child directly (same console); the parent then waits for
    the child to shut down and returns its exit code.
    """
    process = subprocess.Popen(list(argv), env=dict(env), cwd=cwd)
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            logger.info("Ctrl+C received, waiting for the server to stop")


def _write(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _cmd_prepare(project_root: Path, live_inbox: Path, now: datetime) -> int:
    try:
        result = prepare(project_root, live_inbox, now)
    except (StagingError, OSError) as exc:
        logger.error("prepare failed: %s", exc)
        sys.stderr.write(f"ERROR: {exc}\n")
        return EXIT_ERROR
    _write(f"Run folder: {result.run_dir}")
    for name in result.copied:
        _write(f"  copied: {name}")
    _write(f"Copied {len(result.copied)} PDF(s) from {live_inbox}")
    return EXIT_OK


def _cmd_serve(
    project_root: Path,
    run_name: str | None,
    print_only: bool,
    runner: Runner,
    base_env: Mapping[str, str],
) -> int:
    try:
        run_dir = select_run(project_root, run_name)
    except (StagingError, OSError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return EXIT_ERROR
    overrides = build_overrides(run_dir)
    problems = check_overrides(overrides, run_dir, live_default_paths(project_root))
    if problems:
        for problem in problems:
            logger.error("Safety check failed: %s", problem)
            sys.stderr.write(f"REFUSED: {problem}\n")
        return EXIT_UNSAFE
    argv = build_uvicorn_argv(sys.executable)
    _write("=" * 60)
    _write(f"STAGING RUN: {run_dir.name}")
    _write("=" * 60)
    for name, value in overrides.items():
        _write(f"{name}={value}")
    _write("Command: " + " ".join(argv))
    if print_only:
        _write("--print-only: server not started")
        return EXIT_OK
    return runner(argv, build_child_env(base_env, overrides), project_root)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="copy the live inbox PDFs into a new staging run")
    serve = sub.add_parser("serve", help="start the API against a staging run")
    serve.add_argument("--run", default=None, help="run folder name (default: newest)")
    serve.add_argument(
        "--print-only", action="store_true", help="print the overrides, start nothing"
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
    live_inbox: Path | None = None,
    now: Callable[[], datetime] = datetime.now,
    runner: Runner = run_server,
    base_env: Mapping[str, str] | None = None,
) -> int:
    """Parse ``argv`` and run ``prepare`` or ``serve``; return the exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        inbox = live_inbox if live_inbox is not None else Settings().inbox_dir
        return _cmd_prepare(project_root, inbox, now())
    return _cmd_serve(
        project_root,
        args.run,
        args.print_only,
        runner,
        base_env if base_env is not None else os.environ,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
