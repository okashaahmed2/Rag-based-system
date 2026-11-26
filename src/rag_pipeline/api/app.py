"""FastAPI app factory: lifespan wiring and ``GET /health``.

Run with::

    .venv\\Scripts\\python.exe -m uvicorn rag_pipeline.api.app:create_app --factory --host 127.0.0.1 --port 8000

No app object exists at import time: importing this module loads no model and
opens no store. :func:`create_app` only builds the FastAPI object; the real
work happens in its lifespan, once per server start:

1. :func:`rag_pipeline.ingestion.build_ingest_context` runs EXACTLY ONCE (it
   initializes the status store, recovers interrupted documents, loads the
   embedding model and opens the collection). It must never run per request:
   recovery would mark an in-flight document ``failed``.
2. The :class:`RetrievalContext` reuses that context's store, collection and
   model, so the process holds one Chroma client and one model.
3. One ``httpx.Client`` for Ollama is shared by all requests and closed on
   shutdown.

A failing setup (:class:`IngestionSetupError`) is logged and re-raised, so the
server never starts half-working.

Route functions are plain ``def`` (not ``async def``): the pipeline calls are
blocking, and FastAPI runs ``def`` routes in its worker threadpool.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel

from rag_pipeline.config import Settings, get_settings
from rag_pipeline.embeddings import load_embedding_model
from rag_pipeline.ingestion import (
    IngestContext,
    IngestionSetupError,
    ModelLoader,
    build_ingest_context,
)
from rag_pipeline.llm import (
    LLMError,
    LLMModelNotFoundError,
    LLMResponseError,
    check_ollama,
)
from rag_pipeline.logging_config import setup_logging
from rag_pipeline.retrieval import RetrievalContext
from rag_pipeline.status_db import DocumentStatus

logger = logging.getLogger(__name__)

API_TITLE = "RAG Document Pipeline API"
API_VERSION = "0.1.0"

#: ``http_client_factory(settings)`` builds the shared Ollama client (tests inject one).
HttpClientFactory = Callable[[Settings], httpx.Client]


@dataclass(frozen=True)
class ApiState:
    """Everything the routes need, built once by the lifespan (``app.state.pipeline``)."""

    settings: Settings
    ingest_context: IngestContext
    retrieval_context: RetrievalContext
    http_client: httpx.Client


# --- response models -------------------------------------------------------------


class CollectionHealth(BaseModel):
    """The Chroma collection holding the chunks."""

    name: str
    chunk_count: int


class OllamaHealth(BaseModel):
    """Whether answers can be generated right now."""

    reachable: bool
    model: str
    model_available: bool
    error: str | None


class HealthResponse(BaseModel):
    """``GET /health``: ``degraded`` when Ollama is down or the model is missing."""

    status: Literal["ok", "degraded"]
    embedding_model: str
    collection: CollectionHealth
    documents: dict[str, int]
    ollama: OllamaHealth


# --- helpers -----------------------------------------------------------------------


def default_http_client(settings: Settings) -> httpx.Client:
    """The shared Ollama client with the configured timeout."""
    return httpx.Client(timeout=settings.llm_timeout_seconds)


def retrieval_context_from(ingest_context: IngestContext) -> RetrievalContext:
    """A retrieval context sharing the ingest context's store, collection and model."""
    return RetrievalContext(
        store=ingest_context.store,
        collection=ingest_context.collection,
        model=ingest_context.model,
    )


def get_api_state(request: Request) -> ApiState:
    """The :class:`ApiState` stored by the lifespan."""
    state: ApiState = request.app.state.pipeline
    return state


def ollama_health(settings: Settings, client: httpx.Client) -> OllamaHealth:
    """Check Ollama; never raises for an Ollama problem (it is reported instead)."""
    try:
        check_ollama(settings, client=client)
    except LLMModelNotFoundError as exc:
        return OllamaHealth(
            reachable=True, model=settings.llm_model, model_available=False, error=str(exc)
        )
    except LLMResponseError as exc:
        # The server answered, but not usefully: reachable, model state unknown.
        return OllamaHealth(
            reachable=True, model=settings.llm_model, model_available=False, error=str(exc)
        )
    except LLMError as exc:  # unavailable or timeout
        return OllamaHealth(
            reachable=False, model=settings.llm_model, model_available=False, error=str(exc)
        )
    return OllamaHealth(
        reachable=True, model=settings.llm_model, model_available=True, error=None
    )


def document_counts(state: ApiState) -> dict[str, int]:
    """Number of documents per status (every status present, 0 if none)."""
    counts = Counter(r.status.value for r in state.ingest_context.store.list_documents())
    return {status.value: counts.get(status.value, 0) for status in DocumentStatus}


# --- routes ------------------------------------------------------------------------

def health(request: Request) -> HealthResponse:
    """Pipeline and Ollama status; HTTP 200 even when Ollama is down (``degraded``)."""
    state = get_api_state(request)
    collection = state.ingest_context.collection
    ollama = ollama_health(state.settings, state.http_client)
    return HealthResponse(
        status="ok" if ollama.reachable and ollama.model_available else "degraded",
        embedding_model=state.ingest_context.embedding_model_name,
        collection=CollectionHealth(name=collection.name, chunk_count=collection.count()),
        documents=document_counts(state),
        ollama=ollama,
    )


# --- app factory -------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    *,
    model_loader: ModelLoader = load_embedding_model,
    http_client_factory: HttpClientFactory | None = None,
    configure_logging: bool = True,
) -> FastAPI:
    """Build the FastAPI app; the pipeline is wired in its lifespan (see module docs).

    Args:
        settings: Settings to use; ``None`` means :func:`get_settings`.
        model_loader: Embedding model loader passed to the ingest context.
        http_client_factory: Builds the shared Ollama client; ``None`` means
            :func:`default_http_client`.
        configure_logging: Call :func:`setup_logging` at startup (tests pass
            ``False`` to keep pytest's logging untouched).
    """
    app_settings = settings if settings is not None else get_settings()
    make_client = http_client_factory if http_client_factory is not None else default_http_client

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if configure_logging:
            setup_logging(app_settings)
        started = time.perf_counter()
        logger.info("API starting: building the pipeline contexts")
        try:
            ingest_context = build_ingest_context(app_settings, model_loader=model_loader)
        except IngestionSetupError as exc:
            logger.error("API startup failed, server not started: %s", exc)
            raise
        http_client = make_client(app_settings)
        app.state.pipeline = ApiState(
            settings=app_settings,
            ingest_context=ingest_context,
            retrieval_context=retrieval_context_from(ingest_context),
            http_client=http_client,
        )
        logger.info(
            "API ready in %.2f s (embedding model %s, collection %s)",
            time.perf_counter() - started,
            ingest_context.embedding_model_name,
            ingest_context.collection.name,
        )
        try:
            yield
        finally:
            http_client.close()
            logger.info("API stopped: HTTP client closed")

    app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)
    app.add_api_route("/health", health, methods=["GET"], response_model=HealthResponse)
    return app
