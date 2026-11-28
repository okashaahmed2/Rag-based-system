"""Tests for scripts/glossary_smoke.py (tmp eval store, fake chat, no Ollama)."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from rag_pipeline.config import PROJECT_ROOT, Settings
from rag_pipeline.embeddings import ChunkEmbeddings
from rag_pipeline.llm import LLMResponse, LLMUnavailableError
from rag_pipeline.preparation import DocumentChunk
from rag_pipeline.status_db import StatusStore
from rag_pipeline.vector_store import open_collection, replace_document_chunks


def _load_script() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "glossary_smoke.py"
    spec = importlib.util.spec_from_file_location("glossary_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("glossary_smoke", module)
    spec.loader.exec_module(module)
    return module


smoke = _load_script()

MODEL = "stub-model"
DIM = 4
DOC_A = "ab12cd34" + "0" * 56
DOC_B = "ab12cd34" + "1" * 56  # same 8-char prefix as DOC_A
DOC_C = "c0ffee00" + "2" * 56
DOC_PENDING = "deadbeef" + "3" * 56

STUMP = "A decision stump is a decision tree with only one split."
LONG_FIRST = "Line one\nline two\r\n" + "x" * 150

#: doc_id -> filename, list of (chunk_index, page, text)
DOCS: dict[str, tuple[str, list[tuple[int, int, str]]]] = {
    DOC_A: ("alpha.pdf", [(0, 1, LONG_FIRST), (1, 1, "Second chunk."), (2, 3, STUMP)]),
    DOC_B: ("beta.pdf", [(0, 2, "Beta page two.")]),
    DOC_C: ("Gamma notes.pdf", [(0, 1, "Gamma.")]),
}
FIXED_NOW = datetime(2026, 11, 28, 21, 30, 5)


@pytest.fixture(autouse=True)
def clear_chroma_system_cache() -> Iterator[None]:
    yield
    SharedSystemClient.clear_system_cache()


def embeddings(chunks: list[DocumentChunk]) -> ChunkEmbeddings:
    return ChunkEmbeddings(
        chunk_ids=tuple(c.chunk_id for c in chunks),
        vectors=np.ones((len(chunks), DIM), dtype=np.float32) / 2.0,
        model_name=MODEL,
        dimension=DIM,
        max_seq_length=256,
        token_counts=tuple(10 for _ in chunks),
        over_limit_chunk_ids=(),
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A project root holding a built eval store at the script's EVAL_STORE path."""
    store_dir = tmp_path / smoke.EVAL_STORE
    store_dir.mkdir(parents=True)
    store = StatusStore(store_dir / "status.db")
    store.initialize()
    collection = open_collection(store_dir / "chroma", "chunks", MODEL, DIM)
    for doc_id, (filename, rows) in DOCS.items():
        store.register_document(doc_id, filename)
        store.mark_processing(doc_id)
        chunks = [
            DocumentChunk(f"{doc_id}:{i}", doc_id, filename, page, i, text, 0, len(text))
            for i, page, text in rows
        ]
        replace_document_chunks(collection, doc_id, chunks, embeddings(chunks))
        store.mark_done(doc_id, len(chunks))
    store.register_document(DOC_PENDING, "pending.pdf")
    (store_dir / "manifest.json").write_text(
        json.dumps({"config": "test", "embedding_model": MODEL, "embedding_dim": DIM}),
        encoding="utf-8",
    )
    SharedSystemClient.clear_system_cache()
    return tmp_path


@pytest.fixture
def live(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        chroma_path=tmp_path / "live" / "chroma",
        status_db_path=tmp_path / "live" / "status.db",
    )


def response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="llama3.2:3b",
        done_reason="stop",
        prompt_tokens=500,
        completion_tokens=90,
        load_seconds=0.5,
        prompt_eval_seconds=0.2,
        eval_seconds=1.0,
        total_seconds=2.0,
        truncated=False,
        thinking_leaked=False,
    )


def items_json(*terms: str) -> str:
    return json.dumps(
        {
            "items": [
                {"term": t, "definition": f"def of {t}", "evidence": STUMP, "source": "S1"}
                for t in terms
            ]
        }
    )


class FakeChat:
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts) or [items_json("decision stump")]
        self.calls = 0

    def __call__(self, messages: Any, settings: Settings, *, format: Any, num_predict: Any) -> LLMResponse:
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return response(text)


def fake_check(settings: Settings) -> dict[str, Any]:
    return {"version": "0.0-test", "model_present": True}


def run_main(root: Path, live: Settings, argv: list[str], chat_fn: Any = None) -> int:
    return smoke.main(
        argv,
        project_root=root,
        live=live,
        chat_fn=chat_fn if chat_fn is not None else FakeChat(),
        check=fake_check,
        clock=lambda: FIXED_NOW,
    )


def run_files(root: Path) -> list[Path]:
    runs_dir = root / smoke.RUNS_DIR
    return sorted(runs_dir.glob("*.json")) if runs_dir.exists() else []


# --------------------------------------------------------------------------- prefix resolution


def test_unique_prefix_resolves(root: Path, live: Settings) -> None:
    es = smoke.open_eval_store(root / smoke.EVAL_STORE, live)

    (target,) = smoke.resolve_targets(["C0FFEE00:1"], es)

    assert (target.doc.doc_id, target.page_number) == (DOC_C, 1)


def test_longer_prefix_disambiguates(root: Path, live: Settings) -> None:
    es = smoke.open_eval_store(root / smoke.EVAL_STORE, live)

    (target,) = smoke.resolve_targets([DOC_B[:9] + ":2"], es)

    assert target.doc.doc_id == DOC_B


@pytest.mark.parametrize(
    ("page_arg", "message"),
    [
        ("ab12cd34:1", "matches 2 done documents"),
        ("ffffffff:1", "matches no done document"),
        ("deadbeef:1", "matches no done document"),  # exists, but not done
        ("ab12cd3:1", "shorter than 8"),
        ("c0ffee00:2", "no chunks on page 2"),
        ("c0ffee00:0", "must be >= 1"),
        ("c0ffee00:x", "must be an integer"),
        ("c0ffee00", "must look like"),
        (":3", "must look like"),
    ],
)
def test_bad_page_arguments_exit_2_before_any_chat_call(
    root: Path, live: Settings, capsys: pytest.CaptureFixture[str], page_arg: str, message: str
) -> None:
    chat_fn = FakeChat()

    code = run_main(root, live, ["run", "--page", "c0ffee00:1", "--page", page_arg], chat_fn)

    assert code == 2
    assert chat_fn.calls == 0
    assert message in capsys.readouterr().out
    assert run_files(root) == []


# --------------------------------------------------------------------------- live paths


def test_live_store_path_is_refused_before_anything_is_opened(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store_dir = tmp_path / smoke.EVAL_STORE
    live = Settings(
        _env_file=None, chroma_path=store_dir / "chroma", status_db_path=store_dir / "status.db"
    )
    chat_fn = FakeChat()

    code = run_main(tmp_path, live, ["pages"], chat_fn)

    assert code == 2
    assert "Refusing the live store path" in capsys.readouterr().out
    assert chat_fn.calls == 0
    assert not store_dir.exists()


def test_default_live_paths_are_refused() -> None:
    live = Settings(_env_file=None)  # data/chroma and data/status.db under the project root
    assert live.chroma_path == PROJECT_ROOT / "data" / "chroma"

    with pytest.raises(smoke.UsageError, match="Refusing the live store path"):
        smoke.open_eval_store(PROJECT_ROOT / "data", live)


def test_refuse_live_paths_accepts_the_eval_store(root: Path, live: Settings) -> None:
    smoke.refuse_live_paths(smoke.eval_settings(live, root / smoke.EVAL_STORE), live)


def test_missing_store_exits_1_and_creates_nothing(
    tmp_path: Path, live: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_main(tmp_path, live, ["pages"])

    assert code == 1
    assert "Status database not found" in capsys.readouterr().out
    assert not (tmp_path / "data").exists()


# --------------------------------------------------------------------------- pages


def test_pages_lists_every_done_document_and_page(
    root: Path, live: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    chat_fn = FakeChat()

    code = run_main(root, live, ["pages"], chat_fn)

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert chat_fn.calls == 0
    assert lines[0] == f"Eval store {smoke.EVAL_STORE.as_posix()}: 3 done document(s)"
    expected_first = ("Line one line two " + "x" * 150)[:100]
    assert lines[1:] == [
        "",
        f"alpha.pdf  {DOC_A[:12]}  (2 page(s) with chunks)",
        f"  p.1    chunks 2   chars {len(LONG_FIRST) + len('Second chunk.'):<6} | {expected_first}",
        f"  p.3    chunks 1   chars {len(STUMP):<6} | {STUMP}",
        "",
        f"beta.pdf  {DOC_B[:12]}  (1 page(s) with chunks)",
        f"  p.2    chunks 1   chars 14     | Beta page two.",
        "",
        f"Gamma notes.pdf  {DOC_C[:12]}  (1 page(s) with chunks)",
        f"  p.1    chunks 1   chars 6      | Gamma.",
    ]
    assert "pending.pdf" not in "\n".join(lines)
    assert run_files(root) == []


def test_preview_replaces_line_breaks_and_truncates() -> None:
    assert smoke.preview("  a\nb\r\nc  ") == "a b c"
    assert len(smoke.preview("y" * 500)) == 100


# --------------------------------------------------------------------------- run


def test_run_writes_the_json_file_with_expected_keys(
    root: Path, live: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    chat_fn = FakeChat(items_json("decision stump", "random forest"))

    code = run_main(root, live, ["run", "--page", "ab12cd340:3"], chat_fn)

    assert code == 0
    assert chat_fn.calls == 1
    (path,) = run_files(root)
    assert path.name == "20261128-213005.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "created_at",
        "store",
        "config",
        "llm_model",
        "ollama_version",
        "temperature",
        "num_ctx",
        "num_predict",
        "repeat",
        "pages",
        "summary",
    }
    assert payload["num_predict"] == 768 and payload["repeat"] == 1
    assert payload["store"] == smoke.EVAL_STORE.as_posix()
    (page,) = payload["pages"]
    assert (page["doc_id"], page["filename"], page["page_number"]) == (DOC_A, "alpha.pdf", 3)
    assert page["identical"] is None
    (run,) = page["runs"]
    assert set(run) == {
        "status",
        "source_count",
        "items",
        "rejected_count",
        "rejection_reasons",
        "source_remapped",
        "invalid_reason",
        "schema_error_count",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "load_seconds",
        "prompt_eval_seconds",
        "eval_seconds",
        "total_seconds",
        "warm_seconds",
    }
    assert run["status"] == "extracted"
    assert run["items"] == [
        {
            "term": "decision stump",
            "definition": "def of decision stump",
            "evidence": STUMP,
            "page_number": 3,
            "chunk_id": f"{DOC_A}:2",
            "filename": "alpha.pdf",
        }
    ]
    assert run["rejection_reasons"]["term_not_in_evidence"] == 1
    assert run["source_remapped"] == 0
    assert run["warm_seconds"] == pytest.approx(1.5)
    assert payload["summary"]["status_counts"]["extracted"] == 1
    assert payload["summary"]["warm_seconds_median"] == pytest.approx(1.5)
    printed = capsys.readouterr().out
    assert "term: decision stump" in printed and f"chunk {DOC_A}:2" in printed
    assert "status=extracted sources=1 accepted=1 rejected=1" in printed


def test_source_remapped_is_printed_and_written(
    root: Path, live: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    # alpha.pdf p.1: S1 = chunk 0 (LONG_FIRST), S2 = chunk 1. Both items cite S2 but
    # their evidence is in S1, so both are remapped to chunk 0.
    reply = json.dumps(
        {
            "items": [
                {"term": "line one", "definition": "d1", "evidence": "Line one line two", "source": "S2"},
                {"term": "line two", "definition": "d2", "evidence": "line two", "source": "S2"},
            ]
        }
    )

    code = run_main(root, live, ["run", "--page", "ab12cd340:1"], FakeChat(reply))

    assert code == 0
    printed = capsys.readouterr().out
    assert "status=extracted sources=2 accepted=2 rejected=0 remapped=2 invalid_reason=None" in printed
    (run,) = json.loads(run_files(root)[0].read_text(encoding="utf-8"))["pages"][0]["runs"]
    assert run["source_remapped"] == 2
    assert [item["chunk_id"] for item in run["items"]] == [f"{DOC_A}:0", f"{DOC_A}:0"]


def test_repeat_reports_identical_per_page(root: Path, live: Settings) -> None:
    # Order of calls: round 1 (p.3, Gamma p.1), round 2 (p.3, Gamma p.1).
    chat_fn = FakeChat(
        items_json("decision stump"),
        items_json(),
        items_json("decision tree"),
        items_json(),
    )

    code = run_main(
        root, live, ["run", "--page", "ab12cd340:3", "--page", "c0ffee00:1", "--repeat", "2"], chat_fn
    )

    assert code == 0
    assert chat_fn.calls == 4
    payload = json.loads(run_files(root)[0].read_text(encoding="utf-8"))
    first, second = payload["pages"]
    assert [len(r["items"]) for r in first["runs"]] == [1, 1]
    assert first["identical"] is False
    assert [r["status"] for r in second["runs"]] == ["none_found", "none_found"]
    assert second["identical"] is True
    assert payload["summary"]["runs"] == 4


def test_identical_items() -> None:
    same = {"items": [{"term": "a"}]}
    assert smoke.identical_items([same]) is None
    assert smoke.identical_items([same, dict(same)]) is True
    assert smoke.identical_items([same, {"items": None}]) is False
    assert smoke.identical_items([same, {"error": "x"}]) is None


def test_llm_error_is_recorded_and_exits_1(root: Path, live: Settings) -> None:
    def failing_chat(*args: Any, **kwargs: Any) -> LLMResponse:
        raise LLMUnavailableError("down")

    code = run_main(root, live, ["run", "--page", "c0ffee00:1"], failing_chat)

    assert code == 1
    payload = json.loads(run_files(root)[0].read_text(encoding="utf-8"))
    assert payload["pages"][0]["runs"] == [{"error": "LLMUnavailableError: down"}]
    assert payload["summary"]["errors"] == 1


def test_ollama_check_failure_exits_1_without_chat(root: Path, live: Settings) -> None:
    def check(settings: Settings) -> dict[str, Any]:
        raise LLMUnavailableError("down")

    chat_fn = FakeChat()
    code = smoke.main(
        ["run", "--page", "c0ffee00:1"], project_root=root, live=live, chat_fn=chat_fn, check=check
    )

    assert code == 1
    assert chat_fn.calls == 0
    assert run_files(root) == []


def test_run_file_is_never_overwritten(tmp_path: Path) -> None:
    smoke.write_run_file(tmp_path, {"a": 1}, FIXED_NOW)

    with pytest.raises(smoke.SmokeError, match="Cannot write run file"):
        smoke.write_run_file(tmp_path, {"a": 2}, FIXED_NOW)

    assert json.loads((tmp_path / "20261128-213005.json").read_text(encoding="utf-8")) == {"a": 1}


def test_repeat_must_be_positive(root: Path, live: Settings) -> None:
    with pytest.raises(SystemExit) as info:
        run_main(root, live, ["run", "--page", "c0ffee00:1", "--repeat", "0"])

    assert info.value.code == 2
