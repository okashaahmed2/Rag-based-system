"""Manual page-glossary smoke run against the REAL eval store (and, for ``run``, Ollama).

Not run by pytest. Opens ONLY the existing eval store
``data/eval/stores/all-minilm-l6-v2_600_120`` (status db + Chroma, read path:
the collection is opened with ``create=False`` using the model name and
dimension from the store's ``manifest.json``; nothing is created, no
embedding model is loaded). It is pointed at the same way as
``scripts/rag_smoke.py`` (settings overrides for ``chroma_path`` and
``status_db_path``); the live ``data/chroma`` / ``data/status.db`` paths are
refused (exit 2) before anything is opened.

Subcommands::

    pages   Page menu: for every DONE document its filename, the first 12
            characters of its doc_id, then one line per page with chunks:
            page number, chunk count, total characters and the first 100
            characters of the first chunk. No LLM call, no Ollama needed.

    run --page <doc_id_prefix>:<page> [--page ...] [--repeat N]
            Every target is checked before any LLM call (prefix >= 8
            characters matching exactly one done document, page >= 1 with
            chunks; else exit 2). Then ``extract_page_glossary`` per page with
            ``chat`` over one shared ``httpx.Client``; with ``--repeat 2`` the
            whole list runs twice and each page reports whether the accepted
            items are identical. Results go to
            ``data/eval/glossary_runs/<YYYYMMDD-HHMMSS>.json`` (git-ignored).

Printing page text, terms and definitions to the console is acceptable in
this manual script only; nothing of it is logged.

Usage::

    .venv\\Scripts\\python.exe scripts\\glossary_smoke.py pages
    .venv\\Scripts\\python.exe scripts\\glossary_smoke.py run --page 3f2a9c1b:4 --repeat 2

Exit codes: 0 ok, 1 runtime error (store, Ollama, LLM), 2 usage error or
refused path.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
import os
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError

from rag_pipeline.config import Settings
from rag_pipeline.glossary import GlossaryError, GlossaryResult, GlossaryStatus, extract_page_glossary
from rag_pipeline.llm import LLMError, chat, check_ollama
from rag_pipeline.status_db import DocumentRecord, DocumentStatus, StatusStore, StatusStoreError
from rag_pipeline.structured_output import StructuredChatFn
from rag_pipeline.vector_store import VectorStoreError, get_page_chunks, open_collection

#: Existing eval store (relative to the project root), read only.
EVAL_STORE = Path("data/eval/stores/all-minilm-l6-v2_600_120")
#: Run results (relative to the project root; data/eval/ is git-ignored).
RUNS_DIR = Path("data/eval/glossary_runs")
MANIFEST_NAME = "manifest.json"
MIN_PREFIX_CHARS = 8
DOC_ID_SHOWN_CHARS = 12
PREVIEW_CHARS = 100

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

OllamaCheck = Callable[[Settings], dict[str, Any]]


class UsageError(Exception):
    """Bad arguments or a refused path (exit 2, before any LLM call)."""


class SmokeError(Exception):
    """The store or Ollama cannot be used (exit 1)."""


def out(line: str = "") -> None:
    print(line, flush=True)


# --- store ------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalStore:
    """The opened eval store and the settings pointing at it."""

    store: StatusStore
    collection: Collection
    settings: Settings
    manifest: dict[str, Any]


def eval_settings(live: Settings, store_dir: Path) -> Settings:
    """``live`` with ``chroma_path`` / ``status_db_path`` pointed at ``store_dir``."""
    return live.model_copy(
        update={"chroma_path": store_dir / "chroma", "status_db_path": store_dir / "status.db"}
    )


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def refuse_live_paths(settings: Settings, live: Settings) -> None:
    """Raise :class:`UsageError` if ``settings`` points at the live Chroma or status db."""
    for path in (settings.chroma_path, settings.status_db_path):
        for live_path in (live.chroma_path, live.status_db_path):
            if _same_path(path, live_path):
                raise UsageError(
                    f"Refusing the live store path {live_path}; only the eval store is allowed"
                )


def open_eval_store(store_dir: Path, live: Settings) -> EvalStore:
    """Open the eval store at ``store_dir`` on the read path (nothing is created).

    Raises:
        UsageError: a store path is a live path (checked before any access).
        SmokeError: the status db, Chroma folder, manifest or collection is
            missing or unusable.
    """
    settings = eval_settings(live, store_dir)
    refuse_live_paths(settings, live)
    if not settings.status_db_path.is_file():
        raise SmokeError(f"Status database not found at {settings.status_db_path}")
    if not settings.chroma_path.is_dir():
        raise SmokeError(f"Vector store not found at {settings.chroma_path}")
    manifest_path = store_dir / MANIFEST_NAME
    try:
        manifest = dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        model_name = str(manifest["embedding_model"])
        dimension = int(manifest["embedding_dim"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SmokeError(f"Unusable manifest {manifest_path}: {exc!r}") from exc
    try:
        collection = open_collection(
            settings.chroma_path,
            settings.chroma_collection_name,
            model_name,
            dimension,
            create=False,
        )
    except VectorStoreError as exc:
        raise SmokeError(f"Cannot open the eval collection: {exc}") from exc
    return EvalStore(StatusStore(settings.status_db_path), collection, settings, manifest)


def done_documents(store: StatusStore) -> list[DocumentRecord]:
    """Every ``done`` document, sorted by filename then doc_id."""
    docs = store.list_documents(DocumentStatus.DONE)
    return sorted(docs, key=lambda d: (d.filename.casefold(), d.doc_id))


def document_pages(collection: Collection, doc_id: str) -> list[int]:
    """Sorted page numbers that have at least one stored chunk of ``doc_id``."""
    try:
        result = collection.get(where={"doc_id": {"$eq": doc_id}}, include=["metadatas"])
    except ChromaError as exc:
        raise SmokeError(f"Cannot list pages of {doc_id[:DOC_ID_SHOWN_CHARS]}: {exc}") from exc
    return sorted({int(meta["page_number"]) for meta in result["metadatas"] or [] if meta})


def preview(text: str) -> str:
    """First :data:`PREVIEW_CHARS` characters with line breaks turned into spaces."""
    return " ".join(text.strip().splitlines())[:PREVIEW_CHARS]


# --- pages ------------------------------------------------------------------------------


def page_menu_lines(es: EvalStore) -> list[str]:
    """The page menu of every done document (uses ``get_page_chunks`` per page)."""
    docs = done_documents(es.store)
    lines = [f"Eval store {EVAL_STORE.as_posix()}: {len(docs)} done document(s)"]
    for doc in docs:
        pages = document_pages(es.collection, doc.doc_id)
        lines.append("")
        lines.append(
            f"{doc.filename}  {doc.doc_id[:DOC_ID_SHOWN_CHARS]}  ({len(pages)} page(s) with chunks)"
        )
        for page in pages:
            chunks = get_page_chunks(es.collection, doc.doc_id, page)
            chars = sum(len(chunk.text) for chunk in chunks)
            lines.append(
                f"  p.{page:<4} chunks {len(chunks):<3} chars {chars:<6} | {preview(chunks[0].text)}"
            )
    return lines


# --- run targets ------------------------------------------------------------------------


@dataclass(frozen=True)
class PageTarget:
    """One page to run: a done document and a page number with chunks."""

    doc: DocumentRecord
    page_number: int


def parse_page_arg(value: str) -> tuple[str, int]:
    """``"<doc_id_prefix>:<page>"`` -> ``(prefix, page)``; raises :class:`UsageError`."""
    prefix, sep, page_text = value.strip().rpartition(":")
    if not sep or not prefix:
        raise UsageError(f"--page must look like <doc_id_prefix>:<page>, got {value!r}")
    try:
        page = int(page_text)
    except ValueError:
        raise UsageError(f"Page must be an integer in {value!r}") from None
    if page < 1:
        raise UsageError(f"Page must be >= 1 in {value!r}")
    return prefix, page


def resolve_prefix(prefix: str, docs: Sequence[DocumentRecord]) -> DocumentRecord:
    """The one done document whose doc_id starts with ``prefix``; raises :class:`UsageError`."""
    if len(prefix) < MIN_PREFIX_CHARS:
        raise UsageError(
            f"doc_id prefix {prefix!r} is shorter than {MIN_PREFIX_CHARS} characters"
        )
    wanted = prefix.lower()
    matches = [doc for doc in docs if doc.doc_id.startswith(wanted)]
    if not matches:
        raise UsageError(f"doc_id prefix {prefix!r} matches no done document")
    if len(matches) > 1:
        raise UsageError(f"doc_id prefix {prefix!r} matches {len(matches)} done documents")
    return matches[0]


def resolve_targets(values: Sequence[str], es: EvalStore) -> list[PageTarget]:
    """Every ``--page`` value checked (prefix and page with chunks) before any LLM call."""
    docs = done_documents(es.store)
    targets: list[PageTarget] = []
    for value in values:
        prefix, page = parse_page_arg(value)
        doc = resolve_prefix(prefix, docs)
        if not get_page_chunks(es.collection, doc.doc_id, page):
            raise UsageError(f"{doc.filename} ({prefix}) has no chunks on page {page}")
        targets.append(PageTarget(doc, page))
    return targets


# --- run --------------------------------------------------------------------------------


def result_record(result: GlossaryResult) -> dict[str, Any]:
    """JSON-ready record of one glossary result."""
    return {
        "status": result.status.value,
        "source_count": result.source_count,
        "items": None
        if result.items is None
        else [dataclasses.asdict(item) for item in result.items],
        "rejected_count": result.rejected_count,
        "rejection_reasons": {reason.value: n for reason, n in result.rejection_reasons.items()},
        "invalid_reason": result.invalid_reason.value if result.invalid_reason else None,
        "schema_error_count": result.schema_error_count,
        "model": result.model,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "load_seconds": result.load_seconds,
        "prompt_eval_seconds": result.prompt_eval_seconds,
        "eval_seconds": result.eval_seconds,
        "total_seconds": result.total_seconds,
        "warm_seconds": result.total_seconds - result.load_seconds,
    }


def print_result(run_number: int, repeat: int, target: PageTarget, record: dict[str, Any]) -> None:
    doc = target.doc
    out(
        f"\n=== run {run_number}/{repeat}: {doc.filename} p.{target.page_number} "
        f"({doc.doc_id[:DOC_ID_SHOWN_CHARS]})"
    )
    if "error" in record:
        out(f"ERROR {record['error']}")
        return
    items = record["items"]
    out(
        f"status={record['status']} sources={record['source_count']} "
        f"accepted={'-' if items is None else len(items)} rejected={record['rejected_count']} "
        f"invalid_reason={record['invalid_reason']} schema_errors={record['schema_error_count']}"
    )
    for number, item in enumerate(items or [], start=1):
        out(f"  [{number}] term: {item['term']}")
        out(f"      definition: {item['definition']}")
        out(f"      evidence: {item['evidence']}")
        out(f"      page {item['page_number']} chunk {item['chunk_id']}")
    drops = {reason: n for reason, n in record["rejection_reasons"].items() if n}
    out(f"rejections: {drops or '-'}")
    out(
        f"prompt_tokens={record['prompt_tokens']} completion_tokens={record['completion_tokens']} "
        f"load={record['load_seconds']:.2f}s total={record['total_seconds']:.2f}s "
        f"warm={record['warm_seconds']:.2f}s"
    )


def identical_items(runs: Sequence[dict[str, Any]]) -> bool | None:
    """Whether every run of one page accepted the same items (``None`` for one run or errors)."""
    if len(runs) < 2 or any("error" in run for run in runs):
        return None
    return all(run["items"] == runs[0]["items"] for run in runs[1:])


def run_targets(
    targets: Sequence[PageTarget],
    es: EvalStore,
    chat_fn: StructuredChatFn,
    repeat: int,
) -> list[dict[str, Any]]:
    """Run every target ``repeat`` times (whole list per round); one entry per page."""
    pages: list[dict[str, Any]] = [
        {
            "doc_id": t.doc.doc_id,
            "filename": t.doc.filename,
            "page_number": t.page_number,
            "runs": [],
        }
        for t in targets
    ]
    for run_number in range(1, repeat + 1):
        for target, entry in zip(targets, pages):
            try:
                result = extract_page_glossary(
                    target.doc.doc_id,
                    target.page_number,
                    store=es.store,
                    collection=es.collection,
                    settings=es.settings,
                    chat_fn=chat_fn,
                )
                record = result_record(result)
            except (LLMError, GlossaryError, VectorStoreError, StatusStoreError) as exc:
                record = {"error": f"{type(exc).__name__}: {exc}"}
            entry["runs"].append(record)
            print_result(run_number, repeat, target, record)
    for entry in pages:
        entry["identical"] = identical_items(entry["runs"])
    return pages


def summary(pages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts over every run of every page."""
    runs = [run for entry in pages for run in entry["runs"]]
    ok = [run for run in runs if "error" not in run]
    warm = [run["warm_seconds"] for run in ok]
    return {
        "runs": len(runs),
        "errors": len(runs) - len(ok),
        "status_counts": {
            status.value: sum(run["status"] == status.value for run in ok)
            for status in GlossaryStatus
        },
        "invalid_reasons": dict(Counter(run["invalid_reason"] for run in ok if run["invalid_reason"])),
        "warm_seconds_median": statistics.median(warm) if warm else None,
        "warm_seconds_max": max(warm) if warm else None,
        "max_completion_tokens": max((run["completion_tokens"] for run in ok), default=None),
    }


def write_run_file(runs_dir: Path, payload: dict[str, Any], now: datetime) -> Path:
    """Write ``payload`` to ``runs_dir/<YYYYMMDD-HHMMSS>.json`` (never overwrites)."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{now.strftime('%Y%m%d-%H%M%S')}.json"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise SmokeError(f"Cannot write run file {path}: {exc}") from exc
    return path


def cmd_run(
    targets: Sequence[PageTarget],
    es: EvalStore,
    *,
    repeat: int,
    runs_dir: Path,
    chat_fn: StructuredChatFn | None,
    check: OllamaCheck,
    now: datetime,
) -> int:
    settings = es.settings
    out(
        f"Store {EVAL_STORE.as_posix()} | model {settings.llm_model} | "
        f"temperature {settings.llm_temperature} | num_ctx {settings.llm_num_ctx} | "
        f"num_predict {settings.llm_max_extraction_tokens} | {len(targets)} page(s) x {repeat}"
    )
    try:
        info = check(settings)
    except LLMError as exc:
        raise SmokeError(f"check_ollama: {type(exc).__name__}: {exc}") from exc
    out(f"Ollama {info.get('version')}, model present: {info.get('model_present')}")

    if chat_fn is None:
        with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
            pages = run_targets(targets, es, functools.partial(chat, client=client), repeat)
    else:
        pages = run_targets(targets, es, chat_fn, repeat)

    totals = summary(pages)
    payload = {
        "created_at": now.isoformat(timespec="seconds"),
        "store": EVAL_STORE.as_posix(),
        "config": es.manifest.get("config"),
        "llm_model": settings.llm_model,
        "ollama_version": info.get("version"),
        "temperature": settings.llm_temperature,
        "num_ctx": settings.llm_num_ctx,
        "num_predict": settings.llm_max_extraction_tokens,
        "repeat": repeat,
        "pages": pages,
        "summary": totals,
    }
    path = write_run_file(runs_dir, payload, now)

    out("\n=== SUMMARY")
    out(f"runs {totals['runs']}, errors {totals['errors']}")
    out("status counts: " + ", ".join(f"{k} {v}" for k, v in totals["status_counts"].items()))
    out(f"invalid reasons: {totals['invalid_reasons'] or '-'}")
    if totals["warm_seconds_median"] is not None:
        out(
            f"warm seconds (total - load): median {totals['warm_seconds_median']:.2f}, "
            f"max {totals['warm_seconds_max']:.2f}"
        )
    out(f"max completion_tokens: {totals['max_completion_tokens']}")
    if repeat > 1:
        for entry in pages:
            out(f"identical {entry['filename']} p.{entry['page_number']}: {entry['identical']}")
    out(f"results: {path}")
    return EXIT_ERROR if totals["errors"] else EXIT_OK


# --- main -------------------------------------------------------------------------------


def _repeat_count(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("--repeat must be >= 1")
    return count


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Page-glossary smoke run on the eval store.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pages", help="list the pages of every done document (no LLM call)")
    run = sub.add_parser("run", help="extract the glossary of the given pages")
    run.add_argument(
        "--page",
        action="append",
        required=True,
        metavar="DOC_ID_PREFIX:PAGE",
        help="page to run (repeatable); prefix >= 8 characters",
    )
    run.add_argument("--repeat", type=_repeat_count, default=1, help="rounds over all pages")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    project_root: Path | None = None,
    live: Settings | None = None,
    chat_fn: StructuredChatFn | None = None,
    check: OllamaCheck = check_ollama,
    clock: Callable[[], datetime] = datetime.now,
) -> int:
    """Run the script; the keyword arguments are for tests."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    live_settings = live if live is not None else Settings()
    root = project_root if project_root is not None else live_settings.project_root
    try:
        es = open_eval_store(root / EVAL_STORE, live_settings)
        if args.command == "pages":
            for line in page_menu_lines(es):
                out(line)
            return EXIT_OK
        targets = resolve_targets(args.page, es)
        return cmd_run(
            targets,
            es,
            repeat=args.repeat,
            runs_dir=root / RUNS_DIR,
            chat_fn=chat_fn,
            check=check,
            now=clock(),
        )
    except UsageError as exc:
        out(f"ERROR: {exc}")
        return EXIT_USAGE
    except (SmokeError, StatusStoreError, VectorStoreError) as exc:
        out(f"ERROR: {type(exc).__name__}: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    sys.exit(main())
