"""
settings.py
-----------
Central configuration for CornScout.

All tuneable parameters are encapsulated in :class:`Settings`, which reads
values from environment variables (with sensible defaults) and optionally
from a ``.env`` file in the project root.

Usage::

    from agriindex.config.settings import load_settings

    cfg = load_settings()
    print(cfg.DATABASE_PATH)

Module-level aliases are provided for backward compatibility with code that
imports individual names directly from this module (e.g.
``from agriindex.config import settings; settings.DB_PATH``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional .env loader – graceful fallback when python-dotenv is absent
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore[import]

    def _try_load_dotenv(dotenv_path: str | None = None) -> None:
        """Load a ``.env`` file into the process environment if it exists.

        Parameters
        ----------
        dotenv_path:
            Explicit path to the ``.env`` file.  When *None* the function
            walks up from this file's directory until it finds a ``.env`` or
            reaches the filesystem root.
        """
        _load_dotenv(dotenv_path=dotenv_path, override=False)

except ImportError:  # pragma: no cover – python-dotenv is optional

    def _try_load_dotenv(dotenv_path: str | None = None) -> None:  # type: ignore[misc]
        """No-op fallback used when python-dotenv is not installed."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HERE: Path = Path(__file__).resolve().parent

_FALSY = frozenset({"false", "0", "no", "off"})


def _bool_env(key: str, default: bool) -> bool:
    """Return a boolean parsed from an environment variable.

    Parameters
    ----------
    key:
        Name of the environment variable.
    default:
        Value to return when the variable is not set.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.lower() not in _FALSY


# ---------------------------------------------------------------------------
# Settings class
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """Centralised, type-annotated configuration for CornScout.

    All attributes are populated from environment variables; the defaults
    listed here are used when a variable is absent from both the environment
    and any loaded ``.env`` file.

    Attributes
    ----------
    DATABASE_PATH:
        Filesystem path to the SQLite database file.
    DEFAULT_QUERY_LIMIT:
        Default number of results returned per search query.
    HTTP_TIMEOUT:
        Seconds to wait before timing out an HTTP request.
    USER_AGENT:
        ``User-Agent`` header sent with every HTTP request.
    CRAWL_DELAY_SECONDS:
        Courtesy pause (seconds) between successive crawl/search requests.
    LLM_ENABLED:
        Master switch for all LLM processing.  Set the
        ``AGRIINDEX_LLAMA_ENABLED`` environment variable to ``false`` /
        ``0`` / ``no`` / ``off`` to disable without removing config.
    LLAMA_MODEL_PATH:
        Path to the GGUF model file used for LLM summarisation.
    LLAMA_MAX_TOKENS:
        Maximum number of tokens to generate per LLM response.
    LLAMA_CTX:
        Context-window size in tokens for the LLM.
    LLAMA_THREADS:
        Number of CPU threads used for LLM inference.
    KEYWORD_SET_PATH:
        Path to the YAML file that defines keyword sets for classification.
    BLOCKED_DOMAIN_PATH:
        Path to the YAML file listing domains to skip during crawling.
    MAX_WORKERS:
        Maximum number of concurrent fetch workers (Phase 2+).
    MIN_KEYWORD_HITS:
        Minimum keyword-hit count required to consider a page relevant.
    LLAMA_BIN:
        Absolute path to a compiled llama.cpp CLI executable.
    LLAMA_GPU_LAYERS:
        Number of model layers to offload to GPU (-1 = CPU only).
    LLAMA_TEMPERATURE:
        Sampling temperature for LLM inference (lower = more deterministic).
    LOG_LEVEL:
        Python logging level string (e.g. ``"INFO"``, ``"DEBUG"``).
    LOG_FILE:
        Path to the log file written by :mod:`agriindex.utils.logging_utils`.
    """

    # -- Database ------------------------------------------------------------
    DATABASE_PATH: str = field(
        default_factory=lambda: os.environ.get("AGRIINDEX_DB_PATH", "agriindex.db")
    )

    # -- HTTP fetcher --------------------------------------------------------
    HTTP_TIMEOUT: int = field(
        default_factory=lambda: int(os.environ.get("AGRIINDEX_REQUEST_TIMEOUT", "15"))
    )
    USER_AGENT: str = field(
        default_factory=lambda: os.environ.get(
            "AGRIINDEX_USER_AGENT",
            "CornScout/0.1 (agricultural research crawler; contact: info@example.com)",
        )
    )
    MAX_WORKERS: int = field(
        default_factory=lambda: int(os.environ.get("AGRIINDEX_MAX_WORKERS", "4"))
    )

    # -- Search / crawl ------------------------------------------------------
    DEFAULT_QUERY_LIMIT: int = field(
        default_factory=lambda: int(os.environ.get("AGRIINDEX_SEARCH_LIMIT", "20"))
    )
    CRAWL_DELAY_SECONDS: float = field(
        default_factory=lambda: float(os.environ.get("AGRIINDEX_DDG_SLEEP", "1.5"))
    )

    # -- Classification ------------------------------------------------------
    MIN_KEYWORD_HITS: int = field(
        default_factory=lambda: int(os.environ.get("AGRIINDEX_MIN_KW_HITS", "2"))
    )

    # -- LLM (llama.cpp) -----------------------------------------------------
    LLM_ENABLED: bool = field(
        default_factory=lambda: _bool_env("AGRIINDEX_LLAMA_ENABLED", default=True)
    )
    LLAMA_BIN: str = field(
        default_factory=lambda: os.environ.get(
            "AGRIINDEX_LLAMA_BIN", "/usr/local/bin/llama"
        )
    )
    LLAMA_MODEL_PATH: str = field(
        default_factory=lambda: os.environ.get(
            "AGRIINDEX_LLAMA_MODEL", "models/llama-ag-7b.gguf"
        )
    )
    LLAMA_MAX_TOKENS: int = field(
        default_factory=lambda: int(os.environ.get("AGRIINDEX_LLAMA_MAX_TOKENS", "256"))
    )
    LLAMA_GPU_LAYERS: int = field(
        default_factory=lambda: int(os.environ.get("AGRIINDEX_LLAMA_GPU_LAYERS", "-1"))
    )
    LLAMA_CTX: int = field(
        default_factory=lambda: int(os.environ.get("AGRIINDEX_LLAMA_N_CTX", "2048"))
    )
    LLAMA_THREADS: int = field(
        default_factory=lambda: int(os.environ.get("AGRIINDEX_LLAMA_N_THREADS", "4"))
    )
    LLAMA_TEMPERATURE: float = field(
        default_factory=lambda: float(
            os.environ.get("AGRIINDEX_LLAMA_TEMPERATURE", "0.1")
        )
    )

    # -- Config data files ---------------------------------------------------
    KEYWORD_SET_PATH: str = field(
        default_factory=lambda: os.environ.get(
            "AGRIINDEX_KEYWORD_SET_PATH",
            str(_HERE / "keyword_sets.yaml"),
        )
    )
    BLOCKED_DOMAIN_PATH: str = field(
        default_factory=lambda: os.environ.get(
            "AGRIINDEX_BLOCKED_DOMAIN_PATH",
            str(_HERE / "blocked_domains.yaml"),
        )
    )

    # -- Logging -------------------------------------------------------------
    LOG_LEVEL: str = field(
        default_factory=lambda: os.environ.get("AGRIINDEX_LOG_LEVEL", "INFO")
    )
    LOG_FILE: str = field(
        default_factory=lambda: os.environ.get("AGRIINDEX_LOG_FILE", "agriindex.log")
    )

    # -- Backward-compatible read-only properties ----------------------------

    @property
    def DB_PATH(self) -> str:  # noqa: N802
        """Alias for :attr:`DATABASE_PATH` (backward compatibility)."""
        return self.DATABASE_PATH

    @property
    def REQUEST_TIMEOUT(self) -> int:  # noqa: N802
        """Alias for :attr:`HTTP_TIMEOUT` (backward compatibility)."""
        return self.HTTP_TIMEOUT

    @property
    def DEFAULT_SEARCH_LIMIT(self) -> int:  # noqa: N802
        """Alias for :attr:`DEFAULT_QUERY_LIMIT` (backward compatibility)."""
        return self.DEFAULT_QUERY_LIMIT

    @property
    def DDG_SLEEP_SECONDS(self) -> float:  # noqa: N802
        """Alias for :attr:`CRAWL_DELAY_SECONDS` (backward compatibility)."""
        return self.CRAWL_DELAY_SECONDS

    @property
    def LLAMA_ENABLED(self) -> bool:  # noqa: N802
        """Alias for :attr:`LLM_ENABLED` (backward compatibility)."""
        return self.LLM_ENABLED

    @property
    def LLAMA_N_CTX(self) -> int:  # noqa: N802
        """Alias for :attr:`LLAMA_CTX` (backward compatibility)."""
        return self.LLAMA_CTX

    @property
    def LLAMA_N_THREADS(self) -> int:  # noqa: N802
        """Alias for :attr:`LLAMA_THREADS` (backward compatibility)."""
        return self.LLAMA_THREADS

    @property
    def BLOCKED_DOMAINS_PATH(self) -> str:  # noqa: N802
        """Alias for :attr:`BLOCKED_DOMAIN_PATH` (backward compatibility)."""
        return self.BLOCKED_DOMAIN_PATH

    @property
    def KEYWORD_SETS_PATH(self) -> str:  # noqa: N802
        """Alias for :attr:`KEYWORD_SET_PATH` (backward compatibility)."""
        return self.KEYWORD_SET_PATH


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def load_settings(dotenv_path: str | None = None) -> Settings:
    """Create and return a :class:`Settings` instance.

    This function first attempts to load a ``.env`` file (via
    *python-dotenv* when available), then constructs a :class:`Settings`
    object whose attributes are resolved from the current environment.

    Parameters
    ----------
    dotenv_path:
        Explicit path to a ``.env`` file.  Passed through to
        :func:`_try_load_dotenv`.  When *None* the library searches
        upward from this module's directory for a ``.env`` file.

    Returns
    -------
    Settings
        A fully populated :class:`Settings` instance.

    Examples
    --------
    >>> cfg = load_settings()
    >>> cfg.DATABASE_PATH
    'agriindex.db'
    """
    _try_load_dotenv(dotenv_path=dotenv_path)
    return Settings()


# ---------------------------------------------------------------------------
# Module-level singleton – load once at import time so that existing code
# which does ``from agriindex.config import settings; settings.DB_PATH``
# continues to work unchanged.
# ---------------------------------------------------------------------------

_settings: Settings = load_settings()

# Expose every attribute of the singleton at module level so that
# ``import agriindex.config.settings as settings; settings.DB_PATH`` works.
DB_PATH: str = _settings.DATABASE_PATH
DATABASE_PATH: str = _settings.DATABASE_PATH
REQUEST_TIMEOUT: int = _settings.HTTP_TIMEOUT
HTTP_TIMEOUT: int = _settings.HTTP_TIMEOUT
USER_AGENT: str = _settings.USER_AGENT
MAX_WORKERS: int = _settings.MAX_WORKERS
DEFAULT_SEARCH_LIMIT: int = _settings.DEFAULT_QUERY_LIMIT
DEFAULT_QUERY_LIMIT: int = _settings.DEFAULT_QUERY_LIMIT
DDG_SLEEP_SECONDS: float = _settings.CRAWL_DELAY_SECONDS
CRAWL_DELAY_SECONDS: float = _settings.CRAWL_DELAY_SECONDS
MIN_KEYWORD_HITS: int = _settings.MIN_KEYWORD_HITS
LLAMA_ENABLED: bool = _settings.LLM_ENABLED
LLM_ENABLED: bool = _settings.LLM_ENABLED
LLAMA_BIN: str = _settings.LLAMA_BIN
LLAMA_MODEL_PATH: str = _settings.LLAMA_MODEL_PATH
LLAMA_MAX_TOKENS: int = _settings.LLAMA_MAX_TOKENS
LLAMA_GPU_LAYERS: int = _settings.LLAMA_GPU_LAYERS
LLAMA_N_CTX: int = _settings.LLAMA_CTX
LLAMA_CTX: int = _settings.LLAMA_CTX
LLAMA_N_THREADS: int = _settings.LLAMA_THREADS
LLAMA_THREADS: int = _settings.LLAMA_THREADS
LLAMA_TEMPERATURE: float = _settings.LLAMA_TEMPERATURE
KEYWORD_SETS_PATH: str = _settings.KEYWORD_SET_PATH
KEYWORD_SET_PATH: str = _settings.KEYWORD_SET_PATH
BLOCKED_DOMAINS_PATH: str = _settings.BLOCKED_DOMAIN_PATH
BLOCKED_DOMAIN_PATH: str = _settings.BLOCKED_DOMAIN_PATH
LOG_LEVEL: str = _settings.LOG_LEVEL
LOG_FILE: str = _settings.LOG_FILE
