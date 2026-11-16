"""Application-wide logging configuration."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


DEFAULT_LOG_PATH = Path("log/OSIRIS.log")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(
    log_path: str | Path = DEFAULT_LOG_PATH,
    level: str = "INFO",
    console: bool = True,
) -> None:
    """Configure a rotating UTF-8 file log and optional console output."""

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Niveau de log inconnu : {level}")

    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid adding duplicate handlers when a command configures logging more
    # than once in an interactive process.
    target = destination.resolve()
    existing_files = {
        getattr(handler, "baseFilename", None)
        for handler in root.handlers
        if isinstance(handler, RotatingFileHandler)
    }
    if str(target) not in existing_files:
        file_handler = RotatingFileHandler(
            destination,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        file_handler.setLevel(numeric_level)
        root.addHandler(file_handler)

    if console and not any(getattr(handler, "_osiris_console", False) for handler in root.handlers):
        console_handler = logging.StreamHandler()
        console_handler._osiris_console = True  # type: ignore[attr-defined]
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        console_handler.setLevel(numeric_level)
        root.addHandler(console_handler)


def log_stage(logger: logging.Logger, stage: str, **details: object) -> None:
    suffix = " | " + " | ".join(f"{key}={value}" for key, value in details.items()) if details else ""
    logger.info("[%s]%s", stage, suffix)
