"""
logging_utils.py
----------------
Centralised logging configuration for CornScout.

All modules obtain their logger via ``get_logger(__name__)`` so that the
log level and output format are controlled from a single place.

The root ``agriindex`` logger is configured once when this module is first
imported.  Subsequent calls to ``get_logger`` simply return child loggers
that inherit the root configuration.

Phase 2+ could extend this module to:
- Route logs to a structured JSON format for log aggregation services.
- Add a rotating file handler with size limits.
- Integrate with Sentry or another error tracking service.
"""

import logging
import sys

from agriindex.config import settings

_CONFIGURED = False


def _configure_root_logger() -> None:
    """Set up the root ``agriindex`` logger the first time this is called."""
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    root = logging.getLogger("agriindex")
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # File handler (optional — only added when LOG_FILE is non-empty)
    if settings.LOG_FILE:
        try:
            file_handler = logging.FileHandler(settings.LOG_FILE, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:
            root.warning("Could not open log file %s: %s", settings.LOG_FILE, exc)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for *name*.

    Configures the root ``agriindex`` logger on the first call.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    _configure_root_logger()
    # Ensure child loggers are nested under the agriindex namespace
    if not name.startswith("agriindex"):
        name = f"agriindex.{name}"
    return logging.getLogger(name)
