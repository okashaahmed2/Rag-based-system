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

_PATH_FIELDS: tuple[str, ...] = ("inbox_dir", "processed_dir", "failed_dir", "log_dir")

#: Smallest allowed ``chunk_size`` setting, in characters.
MIN_CHUNK_SIZE = 100


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
    log_dir: Path = Path("logs")
    log_level: LogLevel = "INFO"
    #: Maximum chunk length in characters (not tokens).
    chunk_size: int = 600
    #: Maximum overlap between consecutive chunks of a page, in characters.
    chunk_overlap: int = 120

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        """Accept log levels case-insensitively and store them upper-case."""
        if isinstance(value, str):
            return value.strip().upper()
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
