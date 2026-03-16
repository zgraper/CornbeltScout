"""
llama_processor.py
------------------
Generate summaries and perform structured extraction using a local
llama.cpp model.

**Placeholder status:** This module defines the interface and configuration
wiring for llama.cpp integration.  The actual subprocess call is stubbed out
in Phase 1 so the pipeline runs without a model file.  Replace the
``_call_llama`` helper with real invocation once a GGUF model is available.

Responsibilities
----------------
- Accept cleaned page text and produce a short summary (≤ 3 sentences).
- Optionally extract structured data (topics, page type) from the text.
- Fall back to an empty summary when the model is unavailable.

Phase 2+ could extend this module to:
- Run structured extraction prompts (JSON output mode).
- Batch-process multiple pages to amortise model load time.
- Support the llama.cpp HTTP server API for better throughput.
- Swap in other LLM backends (Ollama, llama-cpp-python, etc.).
"""

import subprocess
import textwrap
from typing import Any, Dict

from agriindex.config import settings
from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an agricultural research assistant.
    Summarise the following web page content in 2-3 sentences.
    Focus on agricultural relevance, key findings, or actionable insights.
    Be concise and factual.

    ---
    {text}
    ---

    Summary:
""")

# Maximum characters of page text sent to the model (avoid token overflow)
_MAX_TEXT_CHARS = 4_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_prompt(text: str) -> str:
    """Build the summary prompt for the given page text."""
    truncated = text[:_MAX_TEXT_CHARS].strip()
    return _SUMMARY_PROMPT_TEMPLATE.format(text=truncated)


def _call_llama(prompt: str) -> str:
    """
    Invoke the llama.cpp CLI and return the generated text.

    **Phase 1 stub:** Returns an empty string without making a subprocess
    call.  Implement this function when a GGUF model is available.

    Expected Phase 2 implementation (uncomment and adapt)::

        cmd = [
            settings.LLAMA_BIN,
            "--model", settings.LLAMA_MODEL_PATH,
            "--ctx-size", "2048",
            "--n-predict", str(settings.LLAMA_MAX_TOKENS),
            "--n-gpu-layers", str(settings.LLAMA_GPU_LAYERS),
            "--prompt", prompt,
            "--log-disable",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.stdout.strip()
    """
    # --- Placeholder ---
    logger.debug("llama_processor: stub called (model not configured)")
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_llama_summary(text: str) -> Dict[str, Any]:
    """
    Generate a short summary of *text* using the local llama.cpp model.

    Parameters
    ----------
    text : str
        Cleaned page text.

    Returns
    -------
    dict with keys:
        ``summary`` – str  (empty string when model is unavailable)
    """
    if not text:
        return {"summary": ""}

    import os

    model_available = (
        os.path.isfile(settings.LLAMA_MODEL_PATH)
        and os.path.isfile(settings.LLAMA_BIN)
    )

    if not model_available:
        logger.debug(
            "llama_processor: model or binary not found; skipping summarisation. "
            "Set AGRIINDEX_LLAMA_BIN and AGRIINDEX_LLAMA_MODEL env vars."
        )
        return {"summary": ""}

    prompt = _build_prompt(text)
    summary = _call_llama(prompt)
    logger.debug("llama_processor: summary length=%d", len(summary))
    return {"summary": summary}
