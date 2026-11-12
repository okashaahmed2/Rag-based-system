"""Tests for rag_pipeline.logging_config."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from rag_pipeline.config import Settings
from rag_pipeline.logging_config import (
    CONSOLE_HANDLER_NAME,
    FILE_HANDLER_NAME,
    LOG_FILE_BACKUP_COUNT,
    LOG_FILE_MAX_BYTES,
    LOG_FILE_NAME,
    setup_logging,
)


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    """Restore root logger handlers and level after each test.

    Handlers added by the test are closed so Windows releases the log file
    before pytest deletes tmp_path.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for handler in list(root.handlers):
        if handler not in saved_handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved_level)


def _settings_for(tmp_path: Path, log_level: str = "INFO") -> Settings:
    return Settings(
        _env_file=None,
        project_root=tmp_path,
        log_dir=tmp_path / "logs",
        log_level=log_level,
    )


def _own_handlers() -> list[logging.Handler]:
    return [
        h
        for h in logging.getLogger().handlers
        if h.get_name() in {CONSOLE_HANDLER_NAME, FILE_HANDLER_NAME}
    ]


def test_message_is_written_to_app_log(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path)
    assert not settings.log_dir.exists()

    setup_logging(settings)
    logging.getLogger("rag_pipeline.test").info("hello from the test")
    for handler in _own_handlers():
        handler.flush()

    log_file = tmp_path / "logs" / LOG_FILE_NAME
    assert log_file.is_file()
    content = log_file.read_text(encoding="utf-8")
    assert "hello from the test" in content
    assert "INFO" in content
    assert "rag_pipeline.test" in content


def test_level_from_settings_is_applied(tmp_path: Path) -> None:
    setup_logging(_settings_for(tmp_path, log_level="warning"))
    logger = logging.getLogger("rag_pipeline.test")
    logger.info("should be filtered out")
    logger.warning("should appear")
    for handler in _own_handlers():
        handler.flush()

    content = (tmp_path / "logs" / LOG_FILE_NAME).read_text(encoding="utf-8")
    assert logging.getLogger().level == logging.WARNING
    assert "should appear" in content
    assert "should be filtered out" not in content


def test_file_handler_rotation_settings(tmp_path: Path) -> None:
    setup_logging(_settings_for(tmp_path))

    file_handlers = [h for h in _own_handlers() if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0].maxBytes == LOG_FILE_MAX_BYTES
    assert file_handlers[0].backupCount == LOG_FILE_BACKUP_COUNT
    assert file_handlers[0].encoding == "utf-8"


def test_setup_logging_twice_does_not_duplicate_handlers(tmp_path: Path) -> None:
    root = logging.getLogger()
    handlers_before = len(root.handlers)
    settings = _settings_for(tmp_path)

    setup_logging(settings)
    setup_logging(settings)

    assert len(root.handlers) == handlers_before + 2
    assert sorted(h.get_name() for h in _own_handlers()) == sorted(
        [CONSOLE_HANDLER_NAME, FILE_HANDLER_NAME]
    )

    logging.getLogger("rag_pipeline.test").info("logged once")
    for handler in _own_handlers():
        handler.flush()
    content = (tmp_path / "logs" / LOG_FILE_NAME).read_text(encoding="utf-8")
    assert content.count("logged once") == 1
