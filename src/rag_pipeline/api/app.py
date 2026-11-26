"""FastAPI app factory: lifespan wiring, ``GET /health``, scan and ingest jobs.

Routes: ``GET /health``, ``POST /ingest/scan`` (202, starts a background scan
job or returns the active one), ``POST /ingest`` (202, ingests one file that
is already in the inbox; 409 while any job is active),
``GET /status/{job_id}`` and ``POST /query`` (200, answer with citations;
502/503/504 when retrieval or the LLM fails).

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

import functools
import logging
import re
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints

from rag_pipeline.answering import AnswerStatus, RagAnswer, answer_question

from rag_pipeline.api.jobs import InboxRunner, SingleFileRunner, run_ingest_job, run_job
from rag_pipeline.api.validation import (
    InboxFilenameError,
    InboxFileNotFoundError,
    validate_inbox_filename,
)
from rag_pipeline.config import MAX_QUERY_TOP_K, MIN_QUERY_TOP_K, Settings, get_settings
from rag_pipeline.embeddings import load_embedding_model
from rag_pipeline.inbox_ingestion import ingest_inbox, ingest_one_file
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
    LLMTimeoutError,
    LLMUnavailableError,
    chat,
    check_ollama,
)
from rag_pipeline.logging_config import setup_logging
from rag_pipeline.retrieval import RetrievalContext, RetrievalError
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
JOB_CONFLICT = "another job is active; retry when it has finished"

#: Longest accepted question, after stripping.
MAX_QUESTION_CHARS = 1000

#: ``POST /query`` error mapping: exception type -> (HTTP status, fixed detail).
#: Checked in order; anything else is not caught (HTTP 500). The detail never
#: contains the question, a URL or a local path.
QUERY_ERRORS: tuple[tuple[type[Exception], int, str], ...] = (
    (RetrievalError, 503, "retrieval unavailable"),
    (LLMUnavailableError, 503, "LLM server unavailable"),
    (LLMModelNotFoundError, 503, "LLM model not available"),
    (LLMResponseError, 502, "invalid response from LLM server"),
    (LLMTimeoutError, 504, "LLM timed out"),
)


@dataclass(frozen=True)
class ApiState:
    """Everything the routes need, built once by the lifespan (``app.state.pipeline``)."""

    settings: Settings
    ingest_context: IngestContext
    retrieval_context: RetrievalContext
    http_client: httpx.Client
    inbox_runner: InboxRunner
    single_file_runner: SingleFileRunner


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


class IngestRequest(BaseModel):
    """``POST /ingest`` body: the name of a PDF already in the inbox (no folders)."""

    filename: str


class IngestResponse(BaseModel):
    """``POST /ingest`` (HTTP 202): the new single-file ingest job."""

    job_id: str
    kind: JobKind
    target: str
    status: JobStatus
    already_running: bool


class ConflictingJob(BaseModel):
    """The active job that blocks a new ``POST /ingest``."""

    job_id: str
    kind: JobKind
    status: JobStatus
    target: str | None


class IngestConflictResponse(BaseModel):
    """``POST /ingest`` (HTTP 409): a job is active, nothing was started."""

    detail: str
    active_job: ConflictingJob


class QueryRequest(BaseModel):
    """``POST /query`` body. ``question`` is stripped, then must be 1..1000 characters."""

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS),
    ]
    #: ``None`` uses ``RAG_QUERY_TOP_K``; a strict int (no bool, no string).
    top_k: Annotated[int, Field(strict=True, ge=MIN_QUERY_TOP_K, le=MAX_QUERY_TOP_K)] | None = (
        None
    )


class QueryCitation(BaseModel):
    """A source the answer cites (``[S<n>]``)."""

    source_number: int
    doc_id: str
    filename: str
    page_number: int


class QuerySource(BaseModel):
    """A retrieved source sent to the LLM (no chunk text)."""

    source_number: int
    doc_id: str
    filename: str
    page_number: int
    similarity: float


class QueryTimings(BaseModel):
    """LLM timings in seconds; ``None`` when the LLM was not called."""

    total_seconds: float | None
    load_seconds: float | None
    prompt_eval_seconds: float | None
    eval_seconds: float | None


class QueryResponse(BaseModel):
    """``POST /query`` (HTTP 200).

    ``answer`` fails closed: the model text only when ``status`` is
    ``answered``, the fixed not-found message for ``not_found``, otherwise
    ``null``. ``citations`` is empty unless ``status`` is ``answered``.
    """

    status: AnswerStatus
    answer: str | None
    citations: list[QueryCitation]
    sources: list[QuerySource]
    llm_called: bool
    top_k: int
    truncated: bool
    thinking_leaked: bool
    timings: QueryTimings

    @classmethod
    def from_answer(cls, result: RagAnswer) -> QueryResponse:
        answered = result.status is AnswerStatus.ANSWERED
        if answered or result.status is AnswerStatus.NOT_FOUND:
            answer: str | None = result.answer_text
        else:
            answer = None
        citations = [
            QueryCitation(
                source_number=c.source_number,
                doc_id=c.doc_id,
                filename=c.filename,
                page_number=c.page_number,
            )
            for c in result.citations
        ] if answered else []
        sources = [
            QuerySource(
                source_number=s.source_number,
                doc_id=s.doc_id,
                filename=s.filename,
                page_number=s.page_number,
                similarity=s.similarity,
            )
            for s in result.sources
        ]
        called = result.llm_called
        timings = QueryTimings(
            total_seconds=result.total_seconds if called else None,
            load_seconds=result.load_seconds if called else None,
            prompt_eval_seconds=result.prompt_eval_seconds if called else None,
            eval_seconds=result.eval_seconds if called else None,
        )
        return cls(
            status=result.status,
            answer=answer,
            citations=citations,
            sources=sources,
            llm_called=called,
            top_k=result.top_k,
            truncated=result.truncated,
            thinking_leaked=result.thinking_leaked,
            timings=timings,
        )


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


def ingest_file(
    body: IngestRequest, request: Request, background_tasks: BackgroundTasks
) -> IngestResponse | JSONResponse:
    """Ingest one PDF that is already in the inbox, in the background.

    The name is validated before any job is created: 422 for a bad name, 404
    if no such regular file is directly in the inbox. If any job is active,
    409 with that job and nothing is started; otherwise 202 with the new job.
    """
    state = get_api_state(request)
    try:
        path = validate_inbox_filename(body.filename, state.settings.inbox_dir)
    except InboxFilenameError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InboxFileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    job, created = state.ingest_context.store.create_job_or_get_active(
        JobKind.INGEST, target=path.name
    )
    if not created:
        conflict = IngestConflictResponse(
            detail=JOB_CONFLICT,
            active_job=ConflictingJob(
                job_id=job.job_id, kind=job.kind, status=job.status, target=job.target
            ),
        )
        return JSONResponse(status_code=409, content=conflict.model_dump(mode="json"))
    background_tasks.add_task(
        run_ingest_job,
        job.job_id,
        state.ingest_context,
        state.settings,
        state.single_file_runner,
        path.name,
    )
    return IngestResponse(
        job_id=job.job_id,
        kind=job.kind,
        target=path.name,
        status=job.status,
        already_running=False,
    )


def query(body: QueryRequest, request: Request) -> QueryResponse:
    """Answer a question from the ``done`` documents, with citations.

    Uses the shared Ollama client. Known failures map to fixed messages
    (:data:`QUERY_ERRORS`); the question is never logged or echoed.
    """
    state = get_api_state(request)
    try:
        result = answer_question(
            body.question,
            state.retrieval_context,
            state.settings,
            top_k=body.top_k,
            chat_fn=functools.partial(chat, client=state.http_client),
        )
    except Exception as exc:
        for error_type, status_code, detail in QUERY_ERRORS:
            if isinstance(exc, error_type):
                logger.warning("Query failed: %s -> HTTP %d", type(exc).__name__, status_code)
                raise HTTPException(status_code=status_code, detail=detail) from exc
        raise
    logger.info(
        "Query: status=%s, top_k=%d, llm_called=%s, total %.2f s",
        result.status.value,
        result.top_k,
        result.llm_called,
        result.total_seconds,
    )
    return QueryResponse.from_answer(result)


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
    single_file_runner: SingleFileRunner = ingest_one_file,
    configure_logging: bool = True,
) -> FastAPI:
    """Build the FastAPI app; the pipeline is wired in its lifespan (see module docs).

    Args:
        settings: Settings to use; ``None`` means :func:`get_settings`.
        model_loader: Embedding model loader passed to the ingest context.
        http_client_factory: Builds the shared Ollama client; ``None`` means
            :func:`default_http_client`.
        inbox_runner: What a scan job runs (tests inject a fake).
        single_file_runner: What a ``POST /ingest`` job runs (tests inject a fake).
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
            single_file_runner=single_file_runner,
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
        "/ingest",
        ingest_file,
        methods=["POST"],
        status_code=202,
        response_model=IngestResponse,
        responses={
            404: {"model": ErrorResponse, "description": "No such file in the inbox"},
            409: {"model": IngestConflictResponse, "description": "Another job is active"},
        },
    )
    app.add_api_route(
        "/status/{job_id}",
        job_status,
        methods=["GET"],
        response_model=JobResponse,
        responses={404: {"model": ErrorResponse, "description": "Unknown or malformed job id"}},
    )
    app.add_api_route(
        "/query",
        query,
        methods=["POST"],
        response_model=QueryResponse,
        responses={
            502: {"model": ErrorResponse, "description": "Invalid response from the LLM server"},
            503: {"model": ErrorResponse, "description": "Retrieval or LLM unavailable"},
            504: {"model": ErrorResponse, "description": "The LLM timed out"},
        },
    )
    return app
