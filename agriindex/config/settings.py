"""
settings.py
-----------
Central configuration for AgriIndex.

All tuneable parameters live here so that individual modules never hard-code
magic values.  Later phases can extend this file (or load from an external
.env / config file) without touching module logic.
"""

import os

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH: str = os.environ.get("AGRIINDEX_DB_PATH", "agriindex.db")

# ---------------------------------------------------------------------------
# HTTP fetcher
# ---------------------------------------------------------------------------

# Seconds to wait before timing out a page request
REQUEST_TIMEOUT: int = int(os.environ.get("AGRIINDEX_REQUEST_TIMEOUT", "15"))

# User-agent string sent with every HTTP request
USER_AGENT: str = os.environ.get(
    "AGRIINDEX_USER_AGENT",
    "AgriIndex/0.1 (agricultural research crawler; contact: info@example.com)",
)

# Maximum number of concurrent fetch workers (Phase 2+)
MAX_WORKERS: int = int(os.environ.get("AGRIINDEX_MAX_WORKERS", "4"))

# ---------------------------------------------------------------------------
# DuckDuckGo search
# ---------------------------------------------------------------------------

# Default number of search results to request per query
DEFAULT_SEARCH_LIMIT: int = int(os.environ.get("AGRIINDEX_SEARCH_LIMIT", "20"))

# Seconds to pause between successive DDG search requests (rate-limit courtesy)
DDG_SLEEP_SECONDS: float = float(os.environ.get("AGRIINDEX_DDG_SLEEP", "1.5"))

# ---------------------------------------------------------------------------
# Relevance / classification
# ---------------------------------------------------------------------------

# Minimum keyword hit count required to consider a page relevant
MIN_KEYWORD_HITS: int = int(os.environ.get("AGRIINDEX_MIN_KW_HITS", "2"))

# ---------------------------------------------------------------------------
# LLM (llama.cpp)
# ---------------------------------------------------------------------------

# Set to "false" or "0" to disable all LLM processing without removing config
LLAMA_ENABLED: bool = os.environ.get("AGRIINDEX_LLAMA_ENABLED", "true").lower() not in (
    "false", "0", "no", "off"
)

# Absolute path to a compiled llama.cpp executable
LLAMA_BIN: str = os.environ.get("AGRIINDEX_LLAMA_BIN", "/usr/local/bin/llama")

# Path to the GGUF model file used for summarisation
LLAMA_MODEL_PATH: str = os.environ.get(
    "AGRIINDEX_LLAMA_MODEL", "models/llama-ag-7b.gguf"
)

# Maximum tokens to generate for a summary
LLAMA_MAX_TOKENS: int = int(os.environ.get("AGRIINDEX_LLAMA_MAX_TOKENS", "256"))

# Number of GPU layers to offload (-1 = CPU only)
LLAMA_GPU_LAYERS: int = int(os.environ.get("AGRIINDEX_LLAMA_GPU_LAYERS", "-1"))

# Context window size in tokens
LLAMA_N_CTX: int = int(os.environ.get("AGRIINDEX_LLAMA_N_CTX", "2048"))

# Number of CPU threads to use for inference
LLAMA_N_THREADS: int = int(os.environ.get("AGRIINDEX_LLAMA_N_THREADS", "4"))

# Sampling temperature (lower = more deterministic)
LLAMA_TEMPERATURE: float = float(os.environ.get("AGRIINDEX_LLAMA_TEMPERATURE", "0.1"))

# ---------------------------------------------------------------------------
# Paths to configuration data files
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

BLOCKED_DOMAINS_PATH: str = os.path.join(_HERE, "blocked_domains.yaml")
KEYWORD_SETS_PATH: str = os.path.join(_HERE, "keyword_sets.yaml")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.environ.get("AGRIINDEX_LOG_LEVEL", "INFO")
LOG_FILE: str = os.environ.get("AGRIINDEX_LOG_FILE", "agriindex.log")
