"""Tests for rag_pipeline.api.app (tmp_path stores, stub model, mocked Ollama)."""

from __future__ import annotations

import hashlib
import inspect
import logging
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from rag_pipeline import logging_config
from rag_pipeline.api import app as api_app
from rag_pipeline.api.app import ApiState, create_app, health
from rag_pipeline.config import Settings
from rag_pipeline.embeddings import EmbeddingError
from rag_pipeline.ingestion import IngestionSetupError
from rag_pipeline.status_db import StatusStore

DIM = 8
MODEL_NAME = "stub/model"
LLM_MODEL = "llama3.2:3b"
DOC_A = "a" * 64
DOC_B = "b" * 64


# --- stub embedding model ------------------------------------------------------


class StubTokenizer:
    def __call__(
        self, texts: list[str], *, add_special_tokens: bool, truncation: bool
    ) -> dict[str, list[list[int]]]:
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


class StubModel:
    """Deterministic stand-in for SentenceTransformer."""

    def __init__(self) -> None:
        self.tokenizer = StubTokenizer()
        self.max_seq_length = 256

    def get_embedding_dimension(self) -> int:
        return DIM

    def encode(self, sentences: str | list[str], **kwargs: Any) -> np.ndarray:
        texts = [sentences] if isinstance(sentences, str) else sentences
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.frombuffer(digest[:DIM], dtype=np.uint8).astype(np.float64) + 1
            rows.append(vector / np.linalg.norm(vector))
        out = np.stack(rows)
        return out[0] if isinstance(sentences, str) else out


class SpyLoader:
    """Model loader that counts calls and always returns the same stub."""

    def __init__(self) -> None:
        self.model = StubModel()
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, name: str, local_files_only: bool) -> StubModel:
        self.calls.append((name, local_files_only))
        return self.model


# --- mocked Ollama -------------------------------------------------------------------


OllamaHandler = Callable[[httpx.Request], httpx.Response]


def ollama_ok(installed: tuple[str, ...] = (LLM_MODEL,)) -> OllamaHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.0"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in installed]})
        return httpx.Response(404, json={"error": "not found"})

    return handler


def ollama_down(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def ollama_http_500(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": "boom"})


class ClientFactory:
    """http_client_factory that records the clients it built."""

    def __init__(self, handler: OllamaHandler) -> None:
        self.handler = handler
        self.clients: list[httpx.Client] = []

    def __call__(self, settings: Settings) -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(self.handler))
        self.clients.append(client)
        return client


# --- fixtures ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_chroma_system_cache() -> Iterator[None]:
    yield
    SharedSystemClient.clear_system_cache()


@pytest.fixture(autouse=True)
def remove_app_log_handlers() -> Iterator[None]:
    """Undo setup_logging() calls made by a test (root logger handlers)."""
    yield
    logging_config._remove_own_handlers(logging.getLogger())


def settings_for(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "project_root": tmp_path,
        "chroma_path": tmp_path / "chroma",
        "status_db_path": tmp_path / "state" / "status.db",
        "log_dir": tmp_path / "logs",
        "embedding_model_name": MODEL_NAME,
        "llm_model": LLM_MODEL,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_app(
    tmp_path: Path,
    handler: OllamaHandler | None = None,
    *,
    loader: SpyLoader | None = None,
    factory: ClientFactory | None = None,
) -> tuple[Any, SpyLoader, ClientFactory]:
    spy = loader if loader is not None else SpyLoader()
    clients = factory if factory is not None else ClientFactory(handler or ollama_ok())
    app = create_app(
        settings_for(tmp_path),
        model_loader=spy,
        http_client_factory=clients,
        configure_logging=False,
    )
    return app, spy, clients


# --- import / structure ----------------------------------------------------------------


def test_importing_app_module_loads_no_model_libraries() -> None:
    code = (
        "import sys, rag_pipeline.api.app; "
        "print('sentence_transformers' in sys.modules, 'torch' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )

    assert completed.stdout.strip() == "False False"


def test_module_has_no_app_object() -> None:
    from fastapi import FastAPI

    assert not any(isinstance(value, FastAPI) for value in vars(api_app).values())


def test_health_route_is_a_plain_function() -> None:
    app = create_app(Settings(_env_file=None), configure_logging=False)

    routes = [r for r in app.routes if isinstance(r, APIRoute) and r.path == "/health"]

    assert len(routes) == 1
    assert routes[0].endpoint is health
    assert not inspect.iscoroutinefunction(routes[0].endpoint)
    assert routes[0].methods == {"GET"}


def test_create_app_touches_no_store(tmp_path: Path) -> None:
    app, spy, clients = make_app(tmp_path)

    assert spy.calls == []
    assert clients.clients == []
    assert not (tmp_path / "chroma").exists()
    assert not (tmp_path / "state").exists()
    assert app.title


# --- lifespan ------------------------------------------------------------------------


def test_startup_builds_ingest_context_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds: list[Settings] = []
    recoveries: list[int] = []
    real_build = api_app.build_ingest_context
    real_recover = StatusStore.recover_interrupted

    def counting_build(settings: Settings, **kwargs: Any) -> Any:
        builds.append(settings)
        return real_build(settings, **kwargs)

    def counting_recover(self: StatusStore) -> int:
        recoveries.append(1)
        return real_recover(self)

    monkeypatch.setattr(api_app, "build_ingest_context", counting_build)
    monkeypatch.setattr(StatusStore, "recover_interrupted", counting_recover)
    app, spy, clients = make_app(tmp_path)

    with TestClient(app) as client:
        for _ in range(3):
            assert client.get("/health").status_code == 200

    assert len(builds) == 1
    assert len(recoveries) == 1
    assert spy.calls == [(MODEL_NAME, False)]
    assert len(clients.clients) == 1


def test_retrieval_context_shares_store_collection_and_model(tmp_path: Path) -> None:
    app, spy, _ = make_app(tmp_path)

    with TestClient(app):
        state: ApiState = app.state.pipeline
        ingest = state.ingest_context
        retrieval = state.retrieval_context

        assert retrieval.store is ingest.store
        assert retrieval.collection is ingest.collection
        assert retrieval.model is ingest.model
        assert ingest.model is spy.model


def test_shared_http_client_is_used_and_closed_on_shutdown(tmp_path: Path) -> None:
    app, _, clients = make_app(tmp_path)

    with TestClient(app) as client:
        client.get("/health")
        client.get("/health")
        assert len(clients.clients) == 1
        assert app.state.pipeline.http_client is clients.clients[0]
        assert not clients.clients[0].is_closed

    assert clients.clients[0].is_closed


def test_startup_failure_raises_and_builds_no_http_client(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def failing_loader(name: str, local_files_only: bool) -> StubModel:
        raise EmbeddingError("model not in cache")

    clients = ClientFactory(ollama_ok())
    app = create_app(
        settings_for(tmp_path),
        model_loader=failing_loader,
        http_client_factory=clients,
        configure_logging=False,
    )

    with caplog.at_level(logging.ERROR, logger="rag_pipeline.api.app"):
        with pytest.raises(IngestionSetupError, match="model not in cache"):
            with TestClient(app):
                pass

    assert clients.clients == []
    assert any("API startup failed" in r.getMessage() for r in caplog.records)


def test_startup_recovers_interrupted_documents(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    store = StatusStore(settings.status_db_path)
    store.initialize()
    store.register_document(DOC_A, "a.pdf")
    store.mark_processing(DOC_A)
    store.register_document(DOC_B, "b.pdf")
    app, _, _ = make_app(tmp_path)

    with TestClient(app) as client:
        body = client.get("/health").json()

    assert body["documents"] == {
        "pending": 1,
        "processing": 0,
        "done": 0,
        "failed": 1,
        "needs_ocr": 0,
    }


def test_startup_and_shutdown_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    app, _, _ = make_app(tmp_path)

    with caplog.at_level(logging.INFO, logger="rag_pipeline.api.app"):
        with TestClient(app):
            pass

    messages = [r.getMessage() for r in caplog.records if r.name == "rag_pipeline.api.app"]
    assert any(m.startswith("API ready in") for m in messages)
    assert any(m.startswith("API stopped") for m in messages)


def test_configure_logging_writes_app_log(tmp_path: Path) -> None:
    app = create_app(
        settings_for(tmp_path),
        model_loader=SpyLoader(),
        http_client_factory=ClientFactory(ollama_ok()),
    )

    with TestClient(app):
        pass

    log_text = (tmp_path / "logs" / "app.log").read_text(encoding="utf-8")
    assert "API ready in" in log_text


def test_settings_default_to_get_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL_NAME", MODEL_NAME)
    monkeypatch.setenv("RAG_CHROMA_COLLECTION_NAME", "env_chunks")
    app = create_app(
        model_loader=SpyLoader(),
        http_client_factory=ClientFactory(ollama_ok()),
        configure_logging=False,
    )

    with TestClient(app) as client:
        body = client.get("/health").json()

    assert body["collection"]["name"] == "env_chunks"
    assert (tmp_path / "data" / "status.db").is_file()


# --- /health -----------------------------------------------------------------------


def test_health_ok_when_ollama_has_the_model(tmp_path: Path) -> None:
    app, _, _ = make_app(tmp_path, ollama_ok(installed=("other:1b", LLM_MODEL)))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "embedding_model": MODEL_NAME,
        "collection": {"name": "chunks", "chunk_count": 0},
        "documents": {"pending": 0, "processing": 0, "done": 0, "failed": 0, "needs_ocr": 0},
        "ollama": {"reachable": True, "model": LLM_MODEL, "model_available": True, "error": None},
    }


def test_health_degraded_when_ollama_is_unreachable(tmp_path: Path) -> None:
    app, _, _ = make_app(tmp_path, ollama_down)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["ollama"]["reachable"] is False
    assert body["ollama"]["model_available"] is False
    assert "not running" in body["ollama"]["error"]


def test_health_degraded_when_model_is_missing(tmp_path: Path) -> None:
    app, _, _ = make_app(tmp_path, ollama_ok(installed=("other:1b",)))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["ollama"]["reachable"] is True
    assert body["ollama"]["model_available"] is False
    assert "ollama pull" in body["ollama"]["error"]


def test_health_degraded_when_ollama_returns_http_error(tmp_path: Path) -> None:
    app, _, _ = make_app(tmp_path, ollama_http_500)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["ollama"]["reachable"] is True
    assert body["ollama"]["model_available"] is False
    assert "HTTP 500" in body["ollama"]["error"]


def test_health_degraded_when_ollama_times_out(tmp_path: Path) -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    app, _, _ = make_app(tmp_path, timeout)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["ollama"]["reachable"] is False


def test_health_reports_chunk_count(tmp_path: Path) -> None:
    app, _, _ = make_app(tmp_path)

    with TestClient(app) as client:
        collection = app.state.pipeline.ingest_context.collection
        collection.add(
            ids=["c1", "c2"],
            embeddings=[[1.0] + [0.0] * (DIM - 1), [0.0, 1.0] + [0.0] * (DIM - 2)],
            metadatas=[
                {"doc_id": DOC_A, "filename": "a.pdf", "page_number": 1, "chunk_index": 0},
                {"doc_id": DOC_A, "filename": "a.pdf", "page_number": 1, "chunk_index": 1},
            ],
            documents=["one", "two"],
        )
        body = client.get("/health").json()

    assert body["collection"]["chunk_count"] == 2


def test_default_http_client_uses_llm_timeout() -> None:
    settings = Settings(_env_file=None, llm_timeout_seconds=12.5)

    with api_app.default_http_client(settings) as client:
        assert client.timeout.read == 12.5
        assert client.timeout.connect == 12.5
