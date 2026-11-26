"""FastAPI app factory: lifespan wiring, ``GET /health``, scan jobs.

Routes: ``GET /health``, ``POST /ingest/scan`` (202, starts a background scan
job or returns the active one) and ``GET /status/{job_id}``.

Run with::

    .venv\\Scripts\\python.exe -m uvicorn rag_pipeline.api.app:create_app --factory --host 127.0.0.1 --port 8000

No app object exists at import time: importing this module loads no model and
opens no store. :func:`create_app` only builds the FastAPI object; the real
work happens in its lifespan, once per server start:

1. :func:`rag_pipeline.ingestion.build_ingest_context` runs EXACTLY ONCE (it
   initializes the status store, recovers interrupted documents, loads the
   embedding model and opens the collection). It must never run per request:
   recovery would mark an in-flight document ``failed``.
   Then jobs left ``queued``/``running`` by an earlier process are marked
   ``failed`` (:meth:`StatusStore.recover_interrupted_jobs`), also exactly
   once and before any request is accepted.
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
import re
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel

from rag_pipeline.api.jobs import InboxRunner, run_job
from rag_pipeline.config import Settings, get_settings
from rag_pipeline.embeddings import load_embedding_model
from rag_pipeline.inbox_ingestion import ingest_inbox
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
from rag_pipeline.status_db import (
    DocumentStatus,
    JobKind,
    JobRecord,
    JobStatus,
    StatusStoreError,
)

logger = logging.getLogger(__name__)

API_TITLE = "RAG Document Pipeline API"
API_VERSION = "0.1.0"

#: ``http_client_factory(settings)`` builds the shared Ollama client (tests inject one).
HttpClientFactory = Callable[[Settings], httpx.Client]

#: Job ids made by the store (``uuid4().hex``); anything else is 404 without a lookup.
JOB_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
JOB_NOT_FOUND = "job not found"


@dataclass(frozen=True)
class ApiState:
    """Everything the routes need, built once by the lifespan (``app.state.pipeline``)."""

    settings: Settings
    ingest_context: IngestContext
    retrieval_context: RetrievalContext
    http_client: httpx.Client
    inbox_runner: InboxRunner


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


class DocumentCounts(BaseModel):
    """Number of documents per status (every status present, 0 if none)."""

    pending: int
    processing: int
    done: int
    failed: int
    needs_ocr: int


class ActiveJob(BaseModel):
    """The job that is ``queued`` or ``running`` right now."""

    job_id: str
    kind: JobKind
    status: JobStatus


class HealthResponse(BaseModel):
    """``GET /health``: ``degraded`` when Ollama is down or the model is missing."""

    status: Literal["ok", "degraded"]
    embedding_model: str
    collection: CollectionHealth
    documents: DocumentCounts
    active_job: ActiveJob | None
    ollama: OllamaHealth


class ScanResponse(BaseModel):
    """``POST /ingest/scan`` (HTTP 202): the new job, or the job already active."""

    job_id: str
    kind: JobKind
    status: JobStatus
    already_running: bool


class JobResponse(BaseModel):
    """``GET /status/{job_id}``: one job; ``summary`` is set once it finished."""

    job_id: str
    kind: JobKind
    target: str | None
    status: JobStatus
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    summary: dict[str, Any] | None

    @classmethod
    def from_record(cls, job: JobRecord) -> JobResponse:
        return cls(
            job_id=job.job_id,
            kind=job.kind,
            target=job.target,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error_message=job.error_message,
            summary=job.summary,
        )


class ErrorResponse(BaseModel):
    """Body of an error response."""

    detail: str


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


def document_counts(state: ApiState) -> DocumentCounts:
    """Number of documents per status (every status present, 0 if none)."""
    counts = Counter(r.status.value for r in state.ingest_context.store.list_documents())
    values = {status.value: counts.get(status.value, 0) for status in DocumentStatus}
    return DocumentCounts(**values)


def active_job(state: ApiState) -> ActiveJob | None:
    """The active job, or ``None``."""
    job = state.ingest_context.store.get_active_job()
    if job is None:
        return None
    return ActiveJob(job_id=job.job_id, kind=job.kind, status=job.status)


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
        active_job=active_job(state),
        ollama=ollama,
    )


def ingest_scan(request: Request, background_tasks: BackgroundTasks) -> ScanResponse:
    """Start a scan of the inbox in the background, unless a job is already active.

    Always HTTP 202. If any job (scan or ingest) is ``queued`` or ``running``,
    that job is returned with ``already_running=true`` and nothing is started.
    """
    state = get_api_state(request)
    job, created = state.ingest_context.store.create_job_or_get_active(JobKind.SCAN)
    if created:
        background_tasks.add_task(
            run_job, job.job_id, state.ingest_context, state.settings, state.inbox_runner
        )
    return ScanResponse(
        job_id=job.job_id, kind=job.kind, status=job.status, already_running=not created
    )


def job_status(job_id: str, request: Request) -> JobResponse:
    """One job by id; 404 for an unknown or malformed id (not 32 lowercase hex)."""
    if JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise HTTPException(status_code=404, detail=JOB_NOT_FOUND)
    job = get_api_state(request).ingest_context.store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=JOB_NOT_FOUND)
    return JobResponse.from_record(job)


# --- app factory -------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    *,
    model_loader: ModelLoader = load_embedding_model,
    http_client_factory: HttpClientFactory | None = None,
    inbox_runner: InboxRunner = ingest_inbox,
    configure_logging: bool = True,
) -> FastAPI:
    """Build the FastAPI app; the pipeline is wired in its lifespan (see module docs).

    Args:
        settings: Settings to use; ``None`` means :func:`get_settings`.
        model_loader: Embedding model loader passed to the ingest context.
        http_client_factory: Builds the shared Ollama client; ``None`` means
            :func:`default_http_client`.
        inbox_runner: What a scan job runs (tests inject a fake).
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
        try:
            recovered_jobs = ingest_context.store.recover_interrupted_jobs()
        except StatusStoreError as exc:
            logger.error("API startup failed, server not started: job recovery failed: %s", exc)
            raise
        logger.info("Startup recovery: %d interrupted job(s) marked failed", recovered_jobs)
        http_client = make_client(app_settings)
        app.state.pipeline = ApiState(
            settings=app_settings,
            ingest_context=ingest_context,
            retrieval_context=retrieval_context_from(ingest_context),
            http_client=http_client,
            inbox_runner=inbox_runner,
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
    app.add_api_route(
        "/ingest/scan",
        ingest_scan,
        methods=["POST"],
        status_code=202,
        response_model=ScanResponse,
    )
    app.add_api_route(
        "/status/{job_id}",
        job_status,
        methods=["GET"],
        response_model=JobResponse,
        responses={404: {"model": ErrorResponse, "description": "Unknown or malformed job id"}},
    )
    return app
