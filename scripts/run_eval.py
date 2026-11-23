"""Build (or reuse) an isolated evaluation store and score the golden set.

The evaluation store is a fixed "exam hall", never the live data:

* Corpus: ``data/eval/corpus/`` is a one-time COPY of every PDF in the inbox
  (scanned, invalid, password and duplicate files included, so the done-filter
  is exercised). It is only created when it does not exist; the inbox is read,
  never changed.
* Store: ``data/eval/stores/<config>/`` holds its own Chroma folder and status
  database. ``<config>`` names the model and chunk settings (e.g.
  ``all-minilm-l6-v2_600_120``), so runs with other settings live side by side.
  It is filled with ``build_ingest_context`` + ``ingest_document`` per corpus
  file (no file moves). ``manifest.json`` records the configuration and corpus;
  a store whose manifest differs is refused, never rebuilt in place.

Every golden ``doc_id`` must be ``done`` in the store, otherwise the run stops
(a missing document would silently score 0). Questions go through the real
read path (``build_retrieval_context`` + ``retrieve``, done-filter included).
The result is saved to ``eval/results/<YYYY-MM-DD>_<config>.json`` (ids,
filenames, pages, ranks and similarities only; no question or chunk text) and
summarised on stdout in ASCII.

Usage::

    .venv\\Scripts\\python.exe scripts\\run_eval.py
    .venv\\Scripts\\python.exe scripts\\run_eval.py --chunk-size 800 --chunk-overlap 120
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_pipeline.config import Settings
from rag_pipeline.embeddings import load_embedding_model, model_dimension
from rag_pipeline.evaluation import (
    DEFAULT_KS,
    EvaluationResult,
    GoldenQuestion,
    RetrievedHit,
    evaluate,
    load_golden_set,
)
from rag_pipeline.extraction import compute_file_doc_id
from rag_pipeline.ingestion import (
    IngestResult,
    ModelLoader,
    build_ingest_context,
    ingest_document,
)
from rag_pipeline.retrieval import build_retrieval_context, done_doc_ids, retrieve
from rag_pipeline.status_db import StatusStore

logger = logging.getLogger("run_eval")

#: Locations relative to the project root.
EVAL_DATA_DIR = Path("data/eval")
CORPUS_DIR = EVAL_DATA_DIR / "corpus"
STORES_DIR = EVAL_DATA_DIR / "stores"
GOLDEN_SET = Path("eval/golden_set.jsonl")
RESULTS_DIR = Path("eval/results")
MANIFEST_NAME = "manifest.json"
#: Manifest keys that must match for a store to be reused.
MANIFEST_KEYS = ("config", "embedding_model", "chunk_size", "chunk_overlap", "corpus")


class EvalSetupError(Exception):
    """The evaluation store or golden set is not in a usable state."""


@dataclass(frozen=True)
class EvalPaths:
    """Where one evaluation configuration reads and writes."""

    corpus_dir: Path
    store_dir: Path
    results_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.store_dir / MANIFEST_NAME


def config_name(model_name: str, chunk_size: int, chunk_overlap: int) -> str:
    """``<model slug>_<size>_<overlap>``, e.g. ``all-minilm-l6-v2_600_120``."""
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.rsplit("/", 1)[-1].lower()).strip("-")
    return f"{slug}_{chunk_size}_{chunk_overlap}"


def pdf_files(folder: Path) -> list[Path]:
    """``*.pdf`` directly in ``folder`` (case-insensitive), sorted by name."""
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.lower(),
    )


def ensure_corpus(inbox_dir: Path, corpus_dir: Path) -> bool:
    """Copy the inbox PDFs to ``corpus_dir`` once; return ``True`` if copied now.

    An existing corpus is never touched. The copy goes to a sibling temporary
    folder that is renamed at the end, so a crash never leaves a partial
    corpus that looks complete. The inbox is only read (``copy2`` keeps
    modification times).

    Raises:
        EvalSetupError: the inbox has no PDF, or a leftover temporary folder exists.
    """
    if corpus_dir.exists():
        return False
    pdfs = pdf_files(inbox_dir)
    if not pdfs:
        raise EvalSetupError(f"No PDF files in {inbox_dir} to build the corpus from")
    staging = corpus_dir.with_name(corpus_dir.name + ".partial")
    if staging.exists():
        raise EvalSetupError(f"Leftover {staging} from an interrupted copy; remove it first")
    staging.mkdir(parents=True)
    for pdf in pdfs:
        shutil.copy2(pdf, staging / pdf.name)
    staging.rename(corpus_dir)
    return True


def corpus_entries(corpus_dir: Path) -> list[dict[str, str]]:
    """``{"filename", "doc_id"}`` for every corpus PDF, sorted by name."""
    return [{"filename": p.name, "doc_id": compute_file_doc_id(p)} for p in pdf_files(corpus_dir)]


def eval_settings(
    store_dir: Path, chunk_size: int | None, chunk_overlap: int | None, **base: Any
) -> Settings:
    """Settings for the eval store: its own Chroma folder and status db.

    Everything else (model name, ``local_files_only``, batch size) comes from
    the environment / ``.env`` as usual; ``base`` passes extra overrides (tests).
    ``None`` keeps the configured chunk value.
    """
    overrides: dict[str, Any] = {
        "chroma_path": store_dir / "chroma",
        "status_db_path": store_dir / "status.db",
        **base,
    }
    if chunk_size is not None:
        overrides["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        overrides["chunk_overlap"] = chunk_overlap
    return Settings(**overrides)


def expected_manifest(settings: Settings, config: str, corpus: list[dict[str, str]]) -> dict[str, Any]:
    """Manifest fields known before the model is loaded."""
    return {
        "config": config,
        "embedding_model": settings.embedding_model_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "corpus": corpus,
    }


def read_manifest(path: Path) -> dict[str, Any] | None:
    """The saved manifest, or ``None`` if there is none."""
    if not path.exists():
        return None
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise EvalSetupError(f"Unreadable manifest {path}: {exc}") from exc


def check_manifest(saved: dict[str, Any], expected: dict[str, Any], keys: Sequence[str]) -> None:
    """Refuse a store whose saved manifest differs from ``expected`` on ``keys``."""
    differences = [
        f"{key}: saved {saved.get(key)!r}, expected {expected.get(key)!r}"
        if key != "corpus"
        else "corpus: saved and current corpus files/doc_ids differ"
        for key in keys
        if saved.get(key) != expected.get(key)
    ]
    if differences:
        raise EvalSetupError(
            "Eval store manifest does not match this run (use a new store, never "
            "rebuild in place): " + "; ".join(differences)
        )


def build_eval_store(
    settings: Settings,
    paths: EvalPaths,
    config: str,
    *,
    model_loader: ModelLoader = load_embedding_model,
) -> tuple[dict[str, Any], list[IngestResult]]:
    """Create or reuse the eval store and ingest every corpus file into it.

    A saved manifest is compared BEFORE the store is opened (model name,
    chunk settings, corpus); any difference raises. (A different embedding
    dimension is refused by ``open_collection``.) A new store gets its
    manifest before ingestion starts, so an interrupted build is resumed
    (ingestion is idempotent: done documents are skipped).

    Raises:
        EvalSetupError: manifest mismatch, unreadable manifest or empty corpus.
        IngestionSetupError: the store or model cannot be set up.
    """
    corpus = corpus_entries(paths.corpus_dir)
    if not corpus:
        raise EvalSetupError(f"Corpus {paths.corpus_dir} holds no PDF")
    expected = expected_manifest(settings, config, corpus)
    saved = read_manifest(paths.manifest_path)
    if saved is not None:
        check_manifest(saved, expected, MANIFEST_KEYS)

    # A different model or dimension is also refused here by open_collection.
    context = build_ingest_context(settings, model_loader=model_loader)
    dimension = model_dimension(context.model)
    if saved is not None:
        manifest = saved
    else:
        manifest = {
            **expected,
            "embedding_dim": dimension,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        paths.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    results = [ingest_document(paths.corpus_dir / e["filename"], context) for e in corpus]
    return manifest, results


def check_golden_done(store: StatusStore, questions: Sequence[GoldenQuestion]) -> None:
    """Every expected ``doc_id`` must be ``done`` in ``store``.

    Raises:
        EvalSetupError: naming each missing doc_id and the questions using it.
    """
    done = done_doc_ids(store)
    missing: dict[str, list[str]] = {}
    for question in questions:
        for source in question.expected:
            if source.doc_id not in done:
                missing.setdefault(source.doc_id, []).append(question.id)
    if missing:
        details = "; ".join(
            f"{doc_id} ({', '.join(ids)})" for doc_id, ids in sorted(missing.items())
        )
        raise EvalSetupError(f"Golden doc_id(s) not done in the eval store: {details}")


def run_retrieval(
    settings: Settings,
    questions: Sequence[GoldenQuestion],
    ks: Sequence[int] = DEFAULT_KS,
    *,
    model_loader: ModelLoader = load_embedding_model,
) -> EvaluationResult:
    """Score ``questions`` through the real read path on the eval store."""
    context = build_retrieval_context(settings, model_loader=model_loader)

    def retrieve_hits(question: str, top_k: int) -> list[RetrievedHit]:
        return [
            RetrievedHit(c.doc_id, c.page_number, c.similarity, c.filename)
            for c in retrieve(question, context, top_k)
        ]

    return evaluate(questions, retrieve_hits, ks)


def git_state(project_root: Path) -> dict[str, Any]:
    """Current commit hash and whether the working tree has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, capture_output=True,
            text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("Could not read git state: %s", exc)
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


# --- output ----------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def summary_lines(
    result: EvaluationResult, manifest: dict[str, Any], ingest: Sequence[IngestResult]
) -> list[str]:
    """Readable ASCII summary of one run."""
    # Fallback names only: the first corpus file per doc_id (the one ingested).
    names: dict[str, str] = {}
    for entry in manifest["corpus"]:
        names.setdefault(entry["doc_id"], entry["filename"])
    outcomes = Counter(r.outcome.value for r in ingest)
    lines = [
        f"Config {manifest['config']}: model {manifest['embedding_model']} "
        f"(dim {manifest['embedding_dim']}), chunk {manifest['chunk_size']}/"
        f"{manifest['chunk_overlap']}, corpus {len(manifest['corpus'])} files",
        "Ingest: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())),
        "",
    ]
    header = f"{'group':<14}{'n':>4}{'ans':>5}" + "".join(
        f"{'hit@' + str(k):>8}" for k in result.ks
    ) + f"{'MRR@' + str(result.mrr_k):>9}"
    lines += [header, "-" * len(header)]
    groups = [("OVERALL", result.overall), *result.per_tag.items()]
    for name, summary in groups:
        if summary.hit_at is None:
            cells = "".join(f"{'n/a':>8}" for _ in result.ks) + f"{'n/a':>9}"
        else:
            cells = "".join(f"{summary.hit_at[k]:>8.3f}" for k in result.ks) + f"{summary.mrr:>9.3f}"
        lines.append(f"{name:<14}{summary.questions:>4}{summary.answerable:>5}{cells}")
    lines += ["", f"{'id':<5}{'ans':<5}{'rank':>5}  {'top1 sim':>8}  top-1 result   tags"]
    for q in result.questions:
        top = q.top[0] if q.top else None
        if top is None:
            where = "-"
        else:
            name = top.filename or names.get(top.doc_id, top.doc_id[:12])
            where = f"{name} p{top.page_number}"
        rank = "-" if q.first_relevant_rank is None else str(q.first_relevant_rank)
        lines.append(
            f"{q.id:<5}{'yes' if q.answerable else 'no':<5}{rank:>5}  "
            f"{_fmt(q.top1_similarity):>8}  {where}   {','.join(q.tags)}"
        )
    sim = result.similarity
    lines += [
        "",
        f"Answerable top-1 similarity: min {_fmt(sim.answerable_top1_min)}, median "
        f"{_fmt(sim.answerable_top1_median)}, max {_fmt(sim.answerable_top1_max)} "
        f"({sim.answerable_without_results} without results)",
        "Negative top-1 similarity: "
        + ", ".join(f"{qid} {_fmt(value)}" for qid, value in sim.negative_top1),
    ]
    return [line.encode("ascii", "replace").decode("ascii") for line in lines]


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build/reuse the eval store and score the golden set.")
    parser.add_argument("--chunk-size", type=int, default=None, help="default: configured value")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="default: configured value")
    args = parser.parse_args(argv)

    base = Settings()
    root = base.project_root
    probe = eval_settings(root / STORES_DIR / "_", args.chunk_size, args.chunk_overlap)
    config = config_name(probe.embedding_model_name, probe.chunk_size, probe.chunk_overlap)
    paths = EvalPaths(root / CORPUS_DIR, root / STORES_DIR / config, root / RESULTS_DIR)
    settings = eval_settings(paths.store_dir, args.chunk_size, args.chunk_overlap)

    try:
        questions = load_golden_set(root / GOLDEN_SET)
        if ensure_corpus(base.inbox_dir, paths.corpus_dir):
            sys.stdout.write(f"Corpus copied from {base.inbox_dir} to {paths.corpus_dir}\n")
        manifest, ingest = build_eval_store(settings, paths, config)
        check_golden_done(StatusStore(settings.status_db_path), questions)
        result = run_retrieval(settings, questions)
    except EvalSetupError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1

    run_date = datetime.now(timezone.utc).date().isoformat()
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    result_path = paths.results_dir / f"{run_date}_{config}.json"
    payload = {
        "run": {"date": run_date, "git": git_state(root), "golden_set": GOLDEN_SET.as_posix()},
        "manifest": manifest,
        "ingest": [
            {
                "filename": r.filename,
                "doc_id": r.doc_id,
                "outcome": r.outcome.value,
                "chunk_count": r.chunk_count,
                "skip_reason": r.skip_reason,
            }
            for r in ingest
        ],
        "results": result.to_dict(),
    }
    previous = None
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8")).get("results")
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = summary_lines(result, manifest, ingest)
    lines.append(f"\nResults written to {result_path}")
    if previous is not None:
        lines.append(f"Previous result file for today existed; results identical: {previous == payload['results']}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
