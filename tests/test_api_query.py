"""Tests for POST /query (tmp stores, stub embedding model, Ollama mocked with MockTransport).

Documents are ingested through the real ``POST /ingest`` path (tiny real PDFs,
stub model); the LLM answer comes from an ``httpx.MockTransport`` handler on
the app's shared client.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pymupdf
import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from rag_pipeline import llm as llm_module
from rag_pipeline.answering import NOT_FOUND_MESSAGE
from rag_pipeline.api import app as api_app
from rag_pipeline.api.app import MAX_QUESTION_CHARS, create_app, query
from rag_pipeline.config import MAX_QUERY_TOP_K, MIN_QUERY_TOP_K, Settings
from rag_pipeline.status_db import StatusStore, StatusStoreError
from tests.test_api_app import LLM_MODEL, ClientFactory, SpyLoader, settings_for

#: Distinctive question text: must never appear in a response detail or a log.
SECRET_QUESTION = "What does ZEBRA-QUESTION-7781 say about warranty?"
PAGE_TEXTS = [
    "\n".join(f"Warranty clause {i} covers repairs for two years." for i in range(12)),
    "\n".join(f"Shipping rule {i} says parcels leave within five days." for i in range(12)),
]


# --- mocked Ollama --------------------------------------------------------------------


def chat_body(content: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": LLM_MODEL,
        "message": {"role": "assistant", "content": content},
        "done_reason": "stop",
        "prompt_eval_count": 120,
        "eval_count": 15,
        "load_duration": 100_000_000,
        "prompt_eval_duration": 200_000_000,
        "eval_duration": 300_000_000,
        "total_duration": 700_000_000,
    }
    body.update(extra)
    return body


class FakeOllama:
    """MockTransport handler: version/tags OK; /api/chat answered by ``chat``."""

    def __init__(self, chat: Callable[[httpx.Request], httpx.Response]) -> None:
        self.chat = chat
        self.chat_requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.0"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": LLM_MODEL}]})
        if request.url.path == "/api/chat":
            self.chat_requests.append(json.loads(request.content))
            return self.chat(request)
        return httpx.Response(404, json={"error": "not found"})


def replying(content: str, **extra: Any) -> FakeOllama:
    return FakeOllama(lambda request: httpx.Response(200, json=chat_body(content, **extra)))


# --- app setup ------------------------------------------------------------------------


def write_pdf(path: Path, pages: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as pdf:
        for text in pages:
            pdf.new_page().insert_text((72, 60), text, fontsize=9)
        pdf.save(path)
    return path


def make_app(tmp_path: Path, ollama: FakeOllama) -> tuple[Any, Settings, ClientFactory]:
    settings = settings_for(
        tmp_path,
        inbox_dir=tmp_path / "in",
        processed_dir=tmp_path / "out" / "processed",
        failed_dir=tmp_path / "out" / "failed",
        needs_ocr_dir=tmp_path / "out" / "needs_ocr",
    )
    settings.inbox_dir.mkdir(parents=True, exist_ok=True)
    factory = ClientFactory(ollama)
    app = create_app(
        settings, model_loader=SpyLoader(), http_client_factory=factory, configure_logging=False
    )
    return app, settings, factory


def ingest(client: TestClient, settings: Settings, name: str = "manual.pdf") -> None:
    write_pdf(settings.inbox_dir / name, PAGE_TEXTS)
    response = client.post("/ingest", json={"filename": name})
    assert response.status_code == 202
    job = client.get(f"/status/{response.json()['job_id']}").json()
    assert job["status"] == "succeeded" and job["summary"]["outcomes"]["done"] == 1


def ask(client: TestClient, question: str = SECRET_QUESTION, **extra: Any) -> httpx.Response:
    return client.post("/query", json={"question": question, **extra})


@pytest.fixture(autouse=True)
def clear_chroma_system_cache() -> Iterator[None]:
    yield
    SharedSystemClient.clear_system_cache()


# --- structure ------------------------------------------------------------------------


def test_query_route_is_a_plain_post(tmp_path: Path) -> None:
    app, _, _ = make_app(tmp_path, replying("x"))
    route = {r.path: r for r in app.routes if isinstance(r, APIRoute)}["/query"]

    assert route.endpoint is query
    assert route.methods == {"POST"} and route.status_code in (None, 200)
    assert not inspect.iscoroutinefunction(query)


def test_openapi_shows_query_models_and_error_codes(tmp_path: Path) -> None:
    app, _, _ = make_app(tmp_path, replying("x"))
    spec = app.openapi()
    schemas = spec["components"]["schemas"]
    post = spec["paths"]["/query"]["post"]

    assert set(schemas["QueryRequest"]["properties"]) == {"question", "top_k"}
    assert schemas["QueryRequest"]["required"] == ["question"]
    assert set(schemas["QueryResponse"]["properties"]) == {
        "status", "answer", "citations", "sources", "llm_called", "top_k", "truncated",
        "thinking_leaked", "timings",
    }
    assert set(schemas["QuerySource"]["properties"]) == {
        "source_number", "doc_id", "filename", "page_number", "similarity"
    }
    assert set(schemas["QueryCitation"]["properties"]) == {
        "source_number", "doc_id", "filename", "page_number"
    }
    assert post["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "QueryRequest"
    )
    ok = post["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert ok.endswith("QueryResponse")
    for code in ("502", "503", "504"):
        ref = post["responses"][code]["content"]["application/json"]["schema"]["$ref"]
        assert ref.endswith("ErrorResponse")


# --- answers --------------------------------------------------------------------------


def test_answered_maps_citations_and_sources_without_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ollama = replying("Repairs are covered for two years [S1].")
    app, settings, _ = make_app(tmp_path, ollama)

    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            ingest(client, settings)
            response = ask(client)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["answer"] == "Repairs are covered for two years [S1]."
    assert body["llm_called"] is True and body["top_k"] == settings.query_top_k
    assert body["truncated"] is False and body["thinking_leaked"] is False
    assert body["timings"] == {
        "total_seconds": 0.7, "load_seconds": 0.1, "prompt_eval_seconds": 0.2, "eval_seconds": 0.3
    }
    first = body["sources"][0]
    assert body["citations"] == [
        {k: first[k] for k in ("source_number", "doc_id", "filename", "page_number")}
    ]
    assert first["source_number"] == 1 and first["filename"] == "manual.pdf"
    assert first["page_number"] in (1, 2) and len(first["doc_id"]) == 64
    for source in body["sources"]:
        assert set(source) == {"source_number", "doc_id", "filename", "page_number", "similarity"}
    assert "Warranty clause" not in response.text and "Shipping rule" not in response.text
    assert len(ollama.chat_requests) == 1
    assert "Warranty clause" in json.dumps(ollama.chat_requests[0])  # chunks went to the LLM
    query_lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Query:")]
    assert query_lines == ["Query: status=answered, top_k=5, llm_called=True, total 0.70 s"]
    assert SECRET_QUESTION not in caplog.text


def test_not_found_returns_the_fixed_message(tmp_path: Path) -> None:
    app, settings, _ = make_app(tmp_path, replying("NOT_FOUND"))

    with TestClient(app) as client:
        ingest(client, settings)
        body = ask(client).json()

    assert body["status"] == "not_found"
    assert body["answer"] == NOT_FOUND_MESSAGE
    assert body["citations"] == [] and body["llm_called"] is True
    assert len(body["sources"]) >= 1


def test_no_done_documents_skips_the_llm(tmp_path: Path) -> None:
    ollama = replying("must not be called [S1]")
    app, _, _ = make_app(tmp_path, ollama)

    with TestClient(app) as client:
        response = ask(client)

    assert response.status_code == 200
    body = response.json()
    assert (body["status"], body["answer"], body["llm_called"]) == (
        "not_found", NOT_FOUND_MESSAGE, False
    )
    assert body["citations"] == [] and body["sources"] == []
    assert body["timings"] == {
        "total_seconds": None, "load_seconds": None, "prompt_eval_seconds": None,
        "eval_seconds": None,
    }
    assert ollama.chat_requests == []


@pytest.mark.parametrize(
    ("content", "status"),
    [
        ("Repairs are covered for two years.", "no_citations"),
        ("Repairs are covered [S1] and shipping [S99].", "bad_citations"),
        ("   ", "empty"),
        ("<think>reasoning that never ends", "empty"),
    ],
    ids=["no-citations", "bad-citations", "empty", "unclosed-think"],
)
def test_unusable_answers_fail_closed(tmp_path: Path, content: str, status: str) -> None:
    app, settings, _ = make_app(tmp_path, replying(content))

    with TestClient(app) as client:
        ingest(client, settings)
        response = ask(client)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == status
    assert body["answer"] is None and body["citations"] == []
    assert body["llm_called"] is True and body["sources"]
    if content.strip():
        assert content not in response.text


def test_leaked_thinking_and_truncation_are_reported(tmp_path: Path) -> None:
    ollama = replying("<think>hmm</think>Repairs last two years [S1].", done_reason="length")
    app, settings, _ = make_app(tmp_path, ollama)

    with TestClient(app) as client:
        ingest(client, settings)
        body = ask(client).json()

    assert body["status"] == "answered" and body["answer"] == "Repairs last two years [S1]."
    assert body["thinking_leaked"] is True and body["truncated"] is True


def test_explicit_top_k_is_used(tmp_path: Path) -> None:
    app, settings, _ = make_app(tmp_path, replying("Two years [S1]."))

    with TestClient(app) as client:
        ingest(client, settings)
        body = ask(client, top_k=1).json()

    assert body["top_k"] == 1 and len(body["sources"]) == 1


def test_question_is_stripped_and_1000_chars_are_accepted(tmp_path: Path) -> None:
    ollama = replying("Two years [S1].")
    app, settings, _ = make_app(tmp_path, ollama)
    question = "w" * MAX_QUESTION_CHARS

    with TestClient(app) as client:
        ingest(client, settings)
        response = ask(client, question=f"  \n{question}\t ")

    assert response.status_code == 200
    user_message = ollama.chat_requests[0]["messages"][1]["content"]
    assert user_message.endswith(f"Question: {question}")


# --- shared client ----------------------------------------------------------------------


class _NoNewClients:
    """Stands in for the ``httpx`` module inside llm.py: creating a Client fails."""

    def __getattr__(self, name: str) -> Any:
        return getattr(httpx, name)

    @staticmethod
    def Client(*args: Any, **kwargs: Any) -> Any:  # noqa: N802 - mirrors httpx.Client
        raise AssertionError("chat() created its own httpx.Client instead of the shared one")


def test_query_uses_the_shared_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ollama = replying("Two years [S1].")
    app, settings, factory = make_app(tmp_path, ollama)

    with TestClient(app) as client:
        ingest(client, settings)
        monkeypatch.setattr(llm_module, "httpx", _NoNewClients())
        response = ask(client)
        shared = app.state.pipeline.http_client

    assert response.status_code == 200 and response.json()["status"] == "answered"
    assert len(ollama.chat_requests) == 1  # the MockTransport of the shared client saw it
    assert factory.clients == [shared]  # exactly one client, built at startup


# --- error mapping ----------------------------------------------------------------------


def _raise(error: Exception) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    return handler


LLM_FAILURES = {
    "unavailable": (
        FakeOllama(_raise(httpx.ConnectError("refused"))), 503, "LLM server unavailable"
    ),
    "model-missing": (
        FakeOllama(lambda r: httpx.Response(404, json={"error": "model not found"})),
        503,
        "LLM model not available",
    ),
    "http-500": (
        FakeOllama(lambda r: httpx.Response(500, json={"error": "boom"})),
        502,
        "invalid response from LLM server",
    ),
    "bad-json": (
        FakeOllama(lambda r: httpx.Response(200, content=b"not json")),
        502,
        "invalid response from LLM server",
    ),
    "missing-fields": (
        FakeOllama(lambda r: httpx.Response(200, json={"model": LLM_MODEL})),
        502,
        "invalid response from LLM server",
    ),
    "timeout": (FakeOllama(_raise(httpx.ReadTimeout("slow"))), 504, "LLM timed out"),
}


@pytest.mark.parametrize(
    ("ollama", "code", "detail"), list(LLM_FAILURES.values()), ids=list(LLM_FAILURES)
)
def test_llm_failures_map_to_fixed_errors(
    tmp_path: Path,
    ollama: FakeOllama,
    code: int,
    detail: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app, settings, _ = make_app(tmp_path, ollama)

    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            ingest(client, settings)
            response = ask(client)

    assert response.status_code == code
    assert response.json() == {"detail": detail}
    assert SECRET_QUESTION not in response.text and SECRET_QUESTION not in caplog.text
    assert str(tmp_path) not in response.text and "127.0.0.1" not in response.text
    warnings = [
        r.getMessage() for r in caplog.records
        if r.name == "rag_pipeline.api.app" and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1 and warnings[0].startswith("Query failed: LLM")
    assert f"-> HTTP {code}" in warnings[0]


def test_retrieval_failure_is_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ollama = replying("Two years [S1].")
    app, settings, _ = make_app(tmp_path, ollama)

    def broken(self: StatusStore, status: Any = None) -> Any:
        raise StatusStoreError(f"database at {tmp_path} is locked")

    with caplog.at_level(logging.WARNING, logger="rag_pipeline.api.app"):
        with TestClient(app) as client:
            ingest(client, settings)
            monkeypatch.setattr(StatusStore, "list_documents", broken)
            response = ask(client)

    assert response.status_code == 503
    assert response.json() == {"detail": "retrieval unavailable"}
    assert str(tmp_path) not in response.text
    assert ollama.chat_requests == []
    assert "Query failed: RetrievalError -> HTTP 503" in caplog.text
    assert SECRET_QUESTION not in caplog.text


def test_value_error_is_not_mapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """QueryRequest already validates the input, so a ValueError is a bug: HTTP 500."""

    def bad(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("internal bug")

    app, _, _ = make_app(tmp_path, replying("x"))
    monkeypatch.setattr(api_app, "answer_question", bad)

    with caplog.at_level(logging.WARNING, logger="rag_pipeline.api.app"):
        with TestClient(app) as client:
            with pytest.raises(ValueError, match="internal bug"):
                ask(client)
        with TestClient(app, raise_server_exceptions=False) as client:
            response = ask(client)

    assert response.status_code == 500
    assert "invalid question" not in response.text
    assert "Query failed" not in caplog.text  # not logged as a mapped error


@pytest.mark.parametrize(
    "error", [RuntimeError("unexpected"), llm_module.LLMError("base class only")],
    ids=["runtime", "llm-base"],
)
def test_other_errors_are_not_mapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise error

    app, _, _ = make_app(tmp_path, replying("x"))
    monkeypatch.setattr(api_app, "answer_question", broken)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = ask(client)

    assert response.status_code == 500


# --- request validation (422 from Pydantic) ----------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   \n\t "},
        {"question": "q" * (MAX_QUESTION_CHARS + 1)},
        {"question": "ok?", "top_k": MIN_QUERY_TOP_K - 1},
        {"question": "ok?", "top_k": MAX_QUERY_TOP_K + 1},
        {"question": "ok?", "top_k": "x"},
        {"question": "ok?", "top_k": "5"},
        {"question": "ok?", "top_k": True},
        {"question": "ok?", "top_k": 2.5},
        {"question": 42},
        {},
    ],
    ids=["empty", "whitespace", "1001", "top_k-0", "top_k-21", "top_k-x", "top_k-str",
         "top_k-bool", "top_k-float", "question-int", "missing"],
)
def test_invalid_requests_are_422_and_never_reach_ollama(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    ollama = replying("x [S1]")
    app, _, _ = make_app(tmp_path, ollama)

    with TestClient(app) as client:
        response = client.post("/query", json=payload)

    assert response.status_code == 422
    assert ollama.chat_requests == []


def test_top_k_none_uses_the_setting(tmp_path: Path) -> None:
    app, _, _ = make_app(tmp_path, replying("x"))

    with TestClient(app) as client:
        default = ask(client).json()
        explicit_null = ask(client, top_k=None).json()
        edge = ask(client, top_k=MAX_QUERY_TOP_K).json()

    assert default["top_k"] == explicit_null["top_k"] == 5
    assert edge["top_k"] == MAX_QUERY_TOP_K
