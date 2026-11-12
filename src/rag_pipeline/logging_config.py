"""Central logging setup for the application.

Call :func:`setup_logging` once at application start-up (e.g. in the API or
CLI entry point). Every other module obtains its own logger with::

    import logging

    logger = logging.getLogger(__name__)

and never configures handlers itself. Records propagate to the root logger,
which writes them to the console and to a rotating ``<log_dir>/app.log`` file.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from rag_pipeline.config import Settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE_NAME = "app.log"
LOG_FILE_MAX_BYTES = 1_000_000  # about 1 MB per file
LOG_FILE_BACKUP_COUNT = 3

# Names used to recognise the handlers installed by setup_logging().
CONSOLE_HANDLER_NAME = "rag_pipeline.console"
FILE_HANDLER_NAME = "rag_pipeline.file"
_OWN_HANDLER_NAMES = frozenset({CONSOLE_HANDLER_NAME, FILE_HANDLER_NAME})


def setup_logging(settings: Settings) -> None:
    """Configure the root logger with a console and a rotating file handler.

    Creates ``settings.log_dir`` if it does not exist. The function is
    idempotent: handlers installed by a previous call are closed and replaced,
    so calling it repeatedly never duplicates output. Handlers added by other
    code (e.g. pytest) are left untouched.

    Raises:
        OSError: if the log directory or log file cannot be created.
    """
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = settings.log_dir / LOG_FILE_NAME

    root = logging.getLogger()
    _remove_own_handlers(root)

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.set_name(CONSOLE_HANDLER_NAME)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.set_name(FILE_HANDLER_NAME)
    file_handler.setFormatter(formatter)

    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.setLevel(settings.log_level)

    logging.getLogger(__name__).debug(
        "Logging configured: level=%s, file=%s", settings.log_level, log_file
    )


def _remove_own_handlers(logger: logging.Logger) -> None:
    """Detach and close handlers previously installed by :func:`setup_logging`."""
    for handler in list(logger.handlers):
        if handler.get_name() in _OWN_HANDLER_NAMES:
            logger.removeHandler(handler)
            handler.close()
