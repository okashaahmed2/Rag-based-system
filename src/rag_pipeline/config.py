"""Application configuration loaded from environment variables and ``.env``.

Settings are read from ``RAG_``-prefixed environment variables and from an
optional ``.env`` file in the project root. Environment variables take
precedence over ``.env`` values, which take precedence over the defaults.

Importing this module has no side effects: no directories are created and no
files are read. The ``.env`` file is only read when :class:`Settings` is
instantiated (for example through :func:`get_settings`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/rag_pipeline/config.py -> parents[0]=rag_pipeline, [1]=src, [2]=project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_PATH_FIELDS: tuple[str, ...] = (
    "inbox_dir",
    "processed_dir",
    "failed_dir",
    "needs_ocr_dir",
    "log_dir",
    "chroma_path",
    "status_db_path",
)

#: Smallest allowed ``chunk_size`` setting, in characters.
MIN_CHUNK_SIZE = 100

#: Smallest allowed ``llm_num_ctx`` setting, in tokens.
MIN_LLM_NUM_CTX = 512
#: Largest allowed ``llm_temperature`` setting.
MAX_LLM_TEMPERATURE = 2.0
#: Allowed range of the ``llm_max_answer_tokens`` setting.
MIN_LLM_MAX_ANSWER_TOKENS = 16
MAX_LLM_MAX_ANSWER_TOKENS = 2048
#: Allowed range of the ``query_top_k`` setting.
MIN_QUERY_TOP_K = 1
MAX_QUERY_TOP_K = 20
#: Allowed range of the ``api_port`` setting.
MIN_API_PORT = 1
MAX_API_PORT = 65535


class Settings(BaseSettings):
    """Typed application settings.

    Relative directory paths are resolved against ``project_root``; absolute
    paths are kept exactly as given.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT
    inbox_dir: Path = Path("data/inbox")
    processed_dir: Path = Path("data/processed")
    failed_dir: Path = Path("data/failed")
    #: PDFs without extractable text, waiting for OCR (Phase 13).
    needs_ocr_dir: Path = Path("data/needs_ocr")
    log_dir: Path = Path("logs")
    log_level: LogLevel = "INFO"
    #: Maximum chunk length in characters (not tokens).
    chunk_size: int = 600
    #: Maximum overlap between consecutive chunks of a page, in characters.
    chunk_overlap: int = 120
    #: Sentence Transformers model used for both ingestion and query.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    #: Number of chunks encoded per batch.
    embedding_batch_size: int = 32
    #: True = load the model from the local Hugging Face cache only (no Hub calls).
    embedding_local_files_only: bool = False
    #: Folder of the persistent ChromaDB vector store.
    chroma_path: Path = Path("data/chroma")
    #: Name of the ChromaDB collection holding the document chunks.
    chroma_collection_name: str = "chunks"
    #: SQLite file holding the document status register.
    status_db_path: Path = Path("data/status.db")
    #: Failed documents are retried until they have this many attempts.
    max_ingest_attempts: int = 3
    #: Base URL of the local Ollama server (stored without a trailing slash).
    #: 127.0.0.1, not localhost: on Windows localhost resolves to ::1 first while
    #: Ollama listens on IPv4 only, measured at ~2.2 s extra per new connection.
    ollama_base_url: str = "http://127.0.0.1:11434"
    #: Ollama model used to generate answers. granite4.2:3b was rejected: its template
    #: enables thinking, think:false leaks reasoning, and answers took 60-150 s on CPU.
    llm_model: str = "llama3.2:3b"
    #: Sampling temperature (0 = deterministic).
    llm_temperature: float = 0.0
    #: Context window requested from Ollama, in tokens (prompt + answer).
    llm_num_ctx: int = 4096
    #: Timeout of one HTTP request to Ollama, in seconds.
    llm_timeout_seconds: float = 120.0
    #: Most tokens the LLM may generate per answer (Ollama ``num_predict``).
    llm_max_answer_tokens: int = 256
    #: Most tokens the LLM may generate per structured extraction (Ollama ``num_predict``);
    #: about 70 tokens per glossary item were measured, so 768 leaves room for ~10 items.
    llm_max_extraction_tokens: int = 768
    #: Number of chunks retrieved and sent to the LLM per question.
    query_top_k: int = 5
    #: Interface the FastAPI server binds to (127.0.0.1 = this machine only).
    api_host: str = "127.0.0.1"
    #: TCP port of the FastAPI server.
    api_port: int = 8000

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        """Accept log levels case-insensitively and store them upper-case."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("embedding_batch_size")
    @classmethod
    def _check_embedding_batch_size(cls, value: int) -> int:
        """Require ``embedding_batch_size >= 1``."""
        if value < 1:
            raise ValueError(f"RAG_EMBEDDING_BATCH_SIZE must be >= 1, got {value}")
        return value

    @field_validator("chroma_collection_name")
    @classmethod
    def _check_chroma_collection_name(cls, value: str) -> str:
        """Require a non-empty collection name (surrounding whitespace removed)."""
        name = value.strip()
        if not name:
            raise ValueError("RAG_CHROMA_COLLECTION_NAME must not be empty")
        return name

    @field_validator("max_ingest_attempts")
    @classmethod
    def _check_max_ingest_attempts(cls, value: int) -> int:
        """Require ``max_ingest_attempts >= 1``."""
        if value < 1:
            raise ValueError(f"RAG_MAX_INGEST_ATTEMPTS must be >= 1, got {value}")
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def _check_ollama_base_url(cls, value: str) -> str:
        """Require an ``http(s)://`` URL; strip whitespace and trailing slashes."""
        url = value.strip().rstrip("/")
        scheme, separator, host = url.partition("://")
        if not separator or scheme.lower() not in ("http", "https") or not host:
            raise ValueError(
                f"RAG_OLLAMA_BASE_URL must be an http:// or https:// URL, got {value!r}"
            )
        return url

    @field_validator("llm_model")
    @classmethod
    def _check_llm_model(cls, value: str) -> str:
        """Require a non-empty model name (surrounding whitespace removed)."""
        name = value.strip()
        if not name:
            raise ValueError("RAG_LLM_MODEL must not be empty")
        return name

    @field_validator("llm_temperature")
    @classmethod
    def _check_llm_temperature(cls, value: float) -> float:
        """Require ``0 <= llm_temperature <= MAX_LLM_TEMPERATURE`` (NaN rejected)."""
        if not 0.0 <= value <= MAX_LLM_TEMPERATURE:
            raise ValueError(
                f"RAG_LLM_TEMPERATURE must be between 0 and {MAX_LLM_TEMPERATURE}, "
                f"got {value}"
            )
        return value

    @field_validator("llm_num_ctx")
    @classmethod
    def _check_llm_num_ctx(cls, value: int) -> int:
        """Require ``llm_num_ctx >= MIN_LLM_NUM_CTX``."""
        if value < MIN_LLM_NUM_CTX:
            raise ValueError(f"RAG_LLM_NUM_CTX must be >= {MIN_LLM_NUM_CTX}, got {value}")
        return value

    @field_validator("llm_timeout_seconds")
    @classmethod
    def _check_llm_timeout_seconds(cls, value: float) -> float:
        """Require ``llm_timeout_seconds > 0`` (NaN rejected)."""
        if not value > 0:
            raise ValueError(f"RAG_LLM_TIMEOUT_SECONDS must be > 0, got {value}")
        return value

    @field_validator("llm_max_answer_tokens")
    @classmethod
    def _check_llm_max_answer_tokens(cls, value: int) -> int:
        """Require ``MIN_LLM_MAX_ANSWER_TOKENS <= value <= MAX_LLM_MAX_ANSWER_TOKENS``."""
        if not MIN_LLM_MAX_ANSWER_TOKENS <= value <= MAX_LLM_MAX_ANSWER_TOKENS:
            raise ValueError(
                f"RAG_LLM_MAX_ANSWER_TOKENS must be between {MIN_LLM_MAX_ANSWER_TOKENS} "
                f"and {MAX_LLM_MAX_ANSWER_TOKENS}, got {value}"
            )
        return value

    @field_validator("llm_max_extraction_tokens")
    @classmethod
    def _check_llm_max_extraction_tokens(cls, value: int) -> int:
        """Require ``MIN_LLM_MAX_ANSWER_TOKENS <= value <= MAX_LLM_MAX_ANSWER_TOKENS``."""
        if not MIN_LLM_MAX_ANSWER_TOKENS <= value <= MAX_LLM_MAX_ANSWER_TOKENS:
            raise ValueError(
                f"RAG_LLM_MAX_EXTRACTION_TOKENS must be between {MIN_LLM_MAX_ANSWER_TOKENS} "
                f"and {MAX_LLM_MAX_ANSWER_TOKENS}, got {value}"
            )
        return value

    @field_validator("query_top_k")
    @classmethod
    def _check_query_top_k(cls, value: int) -> int:
        """Require ``MIN_QUERY_TOP_K <= query_top_k <= MAX_QUERY_TOP_K``."""
        if not MIN_QUERY_TOP_K <= value <= MAX_QUERY_TOP_K:
            raise ValueError(
                f"RAG_QUERY_TOP_K must be between {MIN_QUERY_TOP_K} and "
                f"{MAX_QUERY_TOP_K}, got {value}"
            )
        return value

    @field_validator("api_host")
    @classmethod
    def _check_api_host(cls, value: str) -> str:
        """Require a non-empty host (surrounding whitespace removed)."""
        host = value.strip()
        if not host:
            raise ValueError("RAG_API_HOST must not be empty")
        return host

    @field_validator("api_port")
    @classmethod
    def _check_api_port(cls, value: int) -> int:
        """Require ``MIN_API_PORT <= api_port <= MAX_API_PORT``."""
        if not MIN_API_PORT <= value <= MAX_API_PORT:
            raise ValueError(
                f"RAG_API_PORT must be between {MIN_API_PORT} and {MAX_API_PORT}, got {value}"
            )
        return value

    @model_validator(mode="after")
    def _check_chunk_sizes(self) -> Settings:
        """Require ``chunk_size >= MIN_CHUNK_SIZE`` and ``0 <= overlap < size``."""
        if self.chunk_size < MIN_CHUNK_SIZE:
            raise ValueError(
                f"RAG_CHUNK_SIZE must be >= {MIN_CHUNK_SIZE} characters, "
                f"got {self.chunk_size}"
            )
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "RAG_CHUNK_OVERLAP must be >= 0 and smaller than RAG_CHUNK_SIZE "
                f"({self.chunk_size}), got {self.chunk_overlap}"
            )
        return self

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> Settings:
        """Resolve relative directory paths against ``project_root``."""
        for name in _PATH_FIELDS:
            path: Path = getattr(self, name)
            if not path.is_absolute():
                setattr(self, name, self.project_root / path)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (created once, then cached).

    Call ``get_settings.cache_clear()`` to force a reload, e.g. in tests.
    """
    return Settings()
