"""Tests for scripts/run_eval.py logic (tmp_path stores, stub model, small PDFs)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pymupdf
import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from rag_pipeline.config import PROJECT_ROOT, Settings
from rag_pipeline.evaluation import ExpectedSource, GoldenQuestion
from rag_pipeline.extraction import compute_file_doc_id
from rag_pipeline.ingestion import IngestOutcome
from rag_pipeline.status_db import DocumentStatus, StatusStore


def _load_script() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_eval", module)
    spec.loader.exec_module(module)
    return module


run_eval = _load_script()

DIM = 8
MODEL_NAME = "stub/model"


class StubTokenizer:
    def __call__(
        self, texts: list[str], *, add_special_tokens: bool, truncation: bool
    ) -> dict[str, list[list[int]]]:
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


class StubModel:
    """Deterministic stand-in for SentenceTransformer (hash-based unit vectors)."""

    def __init__(self, dimension: int = DIM) -> None:
        self.tokenizer = StubTokenizer()
        self.max_seq_length = 256
        self._dimension = dimension
        self.encode_calls = 0

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, sentences: str | list[str], **kwargs: Any) -> np.ndarray:
        self.encode_calls += 1
        texts = [sentences] if isinstance(sentences, str) else sentences
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.frombuffer(digest[: self._dimension], dtype=np.uint8).astype(np.float64) + 1
            rows.append(vector / np.linalg.norm(vector))
        out = np.stack(rows)
        return out[0] if isinstance(sentences, str) else out


def write_pdf(path: Path, page_texts: list[str]) -> Path:
    with pymupdf.open() as pdf:
        for text in page_texts:
            pdf.new_page().insert_text((72, 60), text, fontsize=9)
        pdf.save(path)
    return path


def body(page: int) -> str:
    return "\n".join(f"Page {page} line {i} explains topic {page * 10 + i}." for i in range(8))


def snapshot(folder: Path) -> list[tuple[str, int, int]]:
    return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in folder.iterdir())


@pytest.fixture(autouse=True)
def clear_chroma_system_cache() -> Iterator[None]:
    yield
    SharedSystemClient.clear_system_cache()


@pytest.fixture
def inbox(tmp_path: Path) -> Path:
    folder = tmp_path / "inbox"
    folder.mkdir()
    write_pdf(folder / "lecture.pdf", [body(1), body(2)])
    shutil.copy2(folder / "lecture.pdf", folder / "lecture copy.pdf")
    write_pdf(folder / "paper.pdf", [body(3)])
    (folder / "broken.pdf").write_bytes(b"not a pdf")
    (folder / "notes.txt").write_text("ignored", encoding="utf-8")
    return folder


@pytest.fixture
def paths(tmp_path: Path) -> Any:
    return run_eval.EvalPaths(
        corpus_dir=tmp_path / "eval" / "corpus",
        store_dir=tmp_path / "eval" / "stores" / "stub_200_40",
        results_dir=tmp_path / "results",
    )


def settings_for(paths: Any, chunk_size: int = 200, chunk_overlap: int = 40) -> Settings:
    return run_eval.eval_settings(
        paths.store_dir,
        chunk_size,
        chunk_overlap,
        _env_file=None,
        project_root=paths.store_dir.parent,
        embedding_model_name=MODEL_NAME,
    )


def loader_for(model: StubModel) -> Any:
    return lambda name, local_files_only: model


def build(paths: Any, inbox: Path, model: StubModel, **kwargs: Any) -> tuple[dict, list]:
    run_eval.ensure_corpus(inbox, paths.corpus_dir)
    settings = settings_for(paths, **kwargs)
    paths.store_dir.mkdir(parents=True, exist_ok=True)
    return run_eval.build_eval_store(settings, paths, "stub_200_40", model_loader=loader_for(model))


def golden(qid: str, doc_id: str, page: int = 1) -> GoldenQuestion:
    return GoldenQuestion(
        id=qid,
        question=f"Page {page} line 3 explains topic",
        expected=(ExpectedSource(doc_id, "x.pdf", (page,)),),
        tags=("fact",),
        notes=None,
        line_number=1,
    )


# --- naming and corpus ---------------------------------------------------------------------


def test_config_name() -> None:
    assert run_eval.config_name("sentence-transformers/all-MiniLM-L6-v2", 600, 120) == (
        "all-minilm-l6-v2_600_120"
    )
    assert run_eval.config_name("BAAI/bge-small-en-v1.5", 800, 120) == "bge-small-en-v1-5_800_120"


def test_ensure_corpus_copies_once_and_never_touches_inbox(inbox: Path, paths: Any) -> None:
    before = snapshot(inbox)

    assert run_eval.ensure_corpus(inbox, paths.corpus_dir) is True
    copied = snapshot(paths.corpus_dir)
    write_pdf(inbox / "later.pdf", [body(9)])
    assert run_eval.ensure_corpus(inbox, paths.corpus_dir) is False

    assert [name for name, *_ in copied] == sorted(
        ["broken.pdf", "lecture copy.pdf", "lecture.pdf", "paper.pdf"]
    )
    assert copied == [(n, s, m) for n, s, m in before if n != "notes.txt"]  # copy2 keeps mtime
    assert snapshot(paths.corpus_dir) == copied  # existing corpus left alone
    assert [e for e in snapshot(inbox) if e[0] != "later.pdf"] == before
    assert not paths.corpus_dir.with_name("corpus.partial").exists()


def test_ensure_corpus_without_pdfs_raises(tmp_path: Path, paths: Any) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(run_eval.EvalSetupError, match="No PDF files"):
        run_eval.ensure_corpus(tmp_path / "empty", paths.corpus_dir)
    assert not paths.corpus_dir.exists()


# --- build_eval_store ------------------------------------------------------------------------


def test_build_writes_manifest_and_ingests_every_corpus_file(inbox: Path, paths: Any) -> None:
    model = StubModel()

    manifest, results = build(paths, inbox, model)

    outcomes = {r.filename: r.outcome for r in results}
    assert outcomes == {
        "broken.pdf": IngestOutcome.FAILED,
        "lecture copy.pdf": IngestOutcome.DONE,
        "lecture.pdf": IngestOutcome.SKIPPED,
        "paper.pdf": IngestOutcome.DONE,
    }
    saved = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert saved == manifest
    assert {k: saved[k] for k in ("config", "embedding_model", "embedding_dim")} == {
        "config": "stub_200_40",
        "embedding_model": MODEL_NAME,
        "embedding_dim": DIM,
    }
    assert (saved["chunk_size"], saved["chunk_overlap"]) == (200, 40)
    assert saved["corpus"] == [
        {"filename": p.name, "doc_id": compute_file_doc_id(p)}
        for p in run_eval.pdf_files(paths.corpus_dir)
    ]
    assert "created_at" in saved
    assert (paths.store_dir / "chroma").is_dir() and (paths.store_dir / "status.db").is_file()


def test_rebuild_reuses_store_and_reembeds_nothing(inbox: Path, paths: Any) -> None:
    first_manifest, _ = build(paths, inbox, StubModel())
    SharedSystemClient.clear_system_cache()
    model = StubModel()

    manifest, results = build(paths, inbox, model)

    assert manifest == first_manifest  # created_at unchanged
    assert model.encode_calls == 0
    done_like = {r.filename: r.outcome for r in results if r.filename != "broken.pdf"}
    assert set(done_like.values()) == {IngestOutcome.SKIPPED}


@pytest.mark.parametrize(
    "change, message",
    [({"chunk_size": 300}, "chunk_size: saved 200, expected 300"),
     ({"chunk_overlap": 60}, "chunk_overlap: saved 40, expected 60")],
)
def test_manifest_mismatch_is_refused_before_the_store_is_opened(
    inbox: Path, paths: Any, change: dict[str, int], message: str
) -> None:
    build(paths, inbox, StubModel())
    manifest_before = paths.manifest_path.read_text(encoding="utf-8")
    SharedSystemClient.clear_system_cache()
    calls: list[str] = []

    def spy_loader(name: str, local_files_only: bool) -> StubModel:
        calls.append(name)
        return StubModel()

    settings = settings_for(paths, **change)
    with pytest.raises(run_eval.EvalSetupError, match=message):
        run_eval.build_eval_store(settings, paths, "stub_200_40", model_loader=spy_loader)

    assert calls == []  # refused before any model load / store open
    assert paths.manifest_path.read_text(encoding="utf-8") == manifest_before


def test_manifest_with_other_model_name_is_refused(inbox: Path, paths: Any) -> None:
    build(paths, inbox, StubModel())
    saved = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    saved["embedding_model"] = "other/model"
    paths.manifest_path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(run_eval.EvalSetupError, match="embedding_model: saved 'other/model'"):
        run_eval.build_eval_store(
            settings_for(paths), paths, "stub_200_40", model_loader=loader_for(StubModel())
        )


def test_changed_corpus_is_refused(inbox: Path, paths: Any) -> None:
    build(paths, inbox, StubModel())
    write_pdf(paths.corpus_dir / "extra.pdf", [body(5)])

    with pytest.raises(run_eval.EvalSetupError, match="corpus: saved and current corpus"):
        run_eval.build_eval_store(
            settings_for(paths), paths, "stub_200_40", model_loader=loader_for(StubModel())
        )


def test_unreadable_manifest_is_refused(inbox: Path, paths: Any) -> None:
    build(paths, inbox, StubModel())
    paths.manifest_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(run_eval.EvalSetupError, match="Unreadable manifest"):
        run_eval.build_eval_store(
            settings_for(paths), paths, "stub_200_40", model_loader=loader_for(StubModel())
        )


# --- golden doc_ids must be done ----------------------------------------------------------------


def test_missing_golden_doc_id_raises_naming_ids(inbox: Path, paths: Any) -> None:
    build(paths, inbox, StubModel())
    store = StatusStore(settings_for(paths).status_db_path)
    done_id = compute_file_doc_id(paths.corpus_dir / "paper.pdf")
    failed_id = compute_file_doc_id(paths.corpus_dir / "broken.pdf")
    unknown_id = "c" * 64
    assert store.get_document(failed_id).status is DocumentStatus.FAILED  # type: ignore[union-attr]

    questions = [golden("q1", done_id), golden("q2", failed_id), golden("q3", unknown_id),
                 golden("q4", unknown_id)]
    with pytest.raises(run_eval.EvalSetupError) as info:
        run_eval.check_golden_done(store, questions)

    message = str(info.value)
    assert f"{failed_id} (q2)" in message
    assert f"{unknown_id} (q3, q4)" in message
    assert done_id not in message


def test_all_golden_doc_ids_done_passes(inbox: Path, paths: Any) -> None:
    build(paths, inbox, StubModel())
    store = StatusStore(settings_for(paths).status_db_path)
    negative = GoldenQuestion("n1", "unanswerable", (), ("negative",), None, 1)

    run_eval.check_golden_done(
        store, [golden("q1", compute_file_doc_id(paths.corpus_dir / "paper.pdf")), negative]
    )


# --- retrieval through the real read path ----------------------------------------------------------


def test_run_retrieval_scores_only_done_documents(inbox: Path, paths: Any) -> None:
    model = StubModel()
    manifest, ingest = build(paths, inbox, model)
    paper = compute_file_doc_id(paths.corpus_dir / "paper.pdf")
    lecture = compute_file_doc_id(paths.corpus_dir / "lecture.pdf")

    result = run_eval.run_retrieval(
        settings_for(paths), [golden("q1", paper), golden("q2", lecture, 2)],
        model_loader=loader_for(model),
    )

    assert [q.id for q in result.questions] == ["q1", "q2"]
    seen = {hit.doc_id for q in result.questions for hit in q.top}
    assert seen <= {paper, lecture}
    assert all(hit.filename for q in result.questions for hit in q.top)
    lines = run_eval.summary_lines(result, manifest, ingest)
    assert all(line.isascii() for line in lines)
    assert any(line.startswith("OVERALL") for line in lines)


def test_summary_names_the_ingested_file_not_a_later_duplicate() -> None:
    from rag_pipeline.evaluation import RetrievedHit, evaluate

    doc = "d" * 64
    manifest = {
        "config": "c", "embedding_model": "m", "embedding_dim": 8, "chunk_size": 600,
        "chunk_overlap": 120,
        "corpus": [{"filename": "first.pdf", "doc_id": doc}, {"filename": "z_copy.pdf", "doc_id": doc}],
    }
    question = golden("q1", doc)
    with_name = evaluate([question], lambda q, k: [RetrievedHit(doc, 1, 0.5, "first.pdf")])
    without_name = evaluate([question], lambda q, k: [RetrievedHit(doc, 1, 0.5)])

    for result in (with_name, without_name):
        row = next(l for l in run_eval.summary_lines(result, manifest, []) if l.startswith("q1 "))
        assert "first.pdf p1" in row and "z_copy" not in row
