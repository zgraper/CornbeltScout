"""
llama_processor.py
------------------
Generate compact structured summaries using a local llama.cpp model.

AgriIndex Phase 1 uses cheap rule-based extraction first, then optionally
runs a small local model through llama.cpp to produce a short summary and
lightweight structured metadata.  This module is designed to be reliable,
conservative, and easy to disable if no local model is available.

Two invocation strategies are tried in order:

1. **Python bindings** – ``llama-cpp-python`` (``import llama_cpp``), if
   installed.  Provides the best performance and control.
2. **Subprocess CLI** – calls the compiled ``llama`` / ``llama-cli`` binary
   via :mod:`subprocess`.  Used as a fallback when the Python package is
   absent.

If neither strategy is available, every call returns a safe fallback
structure so the rest of the pipeline is unaffected.

Phase 2+ could extend this module to:
- TODO: Add batched inference to amortise model load time across multiple
  pages in a single pipeline run.
- TODO: Implement prompt caching so identical page-data fingerprints skip
  the model entirely.
- TODO: Retry with stricter JSON repair (e.g. ``json-repair`` library) when
  the model produces malformed output.
- TODO: Add model-specific prompt variants (Gemma 2 / Phi-3-mini chat
  templates differ from plain-text instruction prompts).
- TODO: Support the llama.cpp HTTP server API for higher throughput.
"""

import json
import logging
import os
import re
import subprocess
import textwrap
from typing import Any, Dict, List, Optional

from agriindex.config import settings
from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters of cleaned_text forwarded to the model (keeps prompts compact)
_MAX_BODY_CHARS: int = 1_500

# Characters allowed for the title and meta-description fields in the prompt
_MAX_TITLE_CHARS: int = 120
_MAX_META_CHARS: int = 200

# JSON schema description sent inside the prompt so the model knows what to
# produce.  Keep it short – small models (Phi-3-mini, Gemma 2) have limited
# context windows.
_JSON_SCHEMA_HINT: str = textwrap.dedent("""\
    {"summary":"1-3 sentence string","topics":["str"],"keywords":["str"],\
"entities":[{"name":"str","type":"str"}],"page_type_suggestion":"str",\
"why_relevant":"short str"}""")

# Compact system-level instruction for the model
_SYSTEM_INSTRUCTION: str = (
    "You are an agricultural research assistant. "
    "Return ONLY compact JSON matching the schema below. "
    "No extra text, no markdown fences."
)

# ---------------------------------------------------------------------------
# Fallback result returned when the model is unavailable or fails
# ---------------------------------------------------------------------------

def _make_fallback(error_note: str = "") -> Dict[str, Any]:
    """Return a safe empty result dict with ``llm_success=False``."""
    return {
        "summary": "",
        "topics": [],
        "keywords": [],
        "entities": [],
        "page_type_suggestion": "",
        "why_relevant": "",
        "llm_success": False,
        "raw_response": error_note,
    }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_truncate(text: str, max_chars: int) -> str:
    """
    Truncate *text* to at most *max_chars* characters without splitting words.

    Cuts at the last whitespace boundary within the limit so the model does
    not receive a partially broken word at the edge of the context window.

    Parameters
    ----------
    text : str
        Input string to truncate.
    max_chars : int
        Maximum number of characters to keep.

    Returns
    -------
    str
        Truncated string (may be shorter than *max_chars* due to word
        boundary rounding).
    """
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Walk back to the last whitespace to avoid cutting mid-word
    last_space = cut.rfind(" ")
    return cut[:last_space].rstrip() if last_space > 0 else cut


# ---------------------------------------------------------------------------
# LlamaProcessor
# ---------------------------------------------------------------------------

class LlamaProcessor:
    """
    Wrapper around a local llama.cpp model for structured page summarisation.

    Tries ``llama-cpp-python`` bindings first; falls back to the CLI binary
    via :mod:`subprocess`.  If neither is available, :meth:`is_available`
    returns ``False`` and every call to :meth:`run_summary` returns a safe
    empty fallback dict.

    Parameters
    ----------
    model_path : str, optional
        Path to the GGUF model file.  Defaults to
        ``settings.LLAMA_MODEL_PATH``.
    n_ctx : int, optional
        Context window size in tokens.  Defaults to ``settings.LLAMA_N_CTX``.
    n_threads : int, optional
        Number of CPU threads for inference.  Defaults to
        ``settings.LLAMA_N_THREADS``.
    temperature : float, optional
        Sampling temperature (lower = more deterministic).  Defaults to
        ``settings.LLAMA_TEMPERATURE``.
    max_tokens : int, optional
        Maximum tokens to generate.  Defaults to
        ``settings.LLAMA_MAX_TOKENS``.
    enabled : bool, optional
        Master on/off switch.  Set to ``False`` (or via the
        ``AGRIINDEX_LLAMA_ENABLED`` env var) to skip all model calls
        without changing other configuration.  Defaults to
        ``settings.LLAMA_ENABLED``.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.model_path: str = model_path or settings.LLAMA_MODEL_PATH
        self.n_ctx: int = n_ctx if n_ctx is not None else settings.LLAMA_N_CTX
        self.n_threads: int = (
            n_threads if n_threads is not None else settings.LLAMA_N_THREADS
        )
        self.temperature: float = (
            temperature if temperature is not None else settings.LLAMA_TEMPERATURE
        )
        self.max_tokens: int = (
            max_tokens if max_tokens is not None else settings.LLAMA_MAX_TOKENS
        )
        self.enabled: bool = (
            enabled if enabled is not None else settings.LLAMA_ENABLED
        )

        # Cached binding instance (populated lazily by _get_binding())
        self._binding: Any = None
        self._binding_loaded: bool = False

        logger.debug(
            "LlamaProcessor init: model=%s  n_ctx=%d  n_threads=%d  "
            "temperature=%.2f  max_tokens=%d  enabled=%s",
            self.model_path,
            self.n_ctx,
            self.n_threads,
            self.temperature,
            self.max_tokens,
            self.enabled,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """
        Return ``True`` when the processor is ready to run inference.

        Checks (in order):

        1. The ``enabled`` flag must be ``True``.
        2. The model file must exist on disk.
        3. Either the ``llama_cpp`` Python package is importable **or**
           the CLI binary at ``settings.LLAMA_BIN`` is executable.

        Returns
        -------
        bool
        """
        if not self.enabled:
            logger.debug("LlamaProcessor.is_available: disabled by config")
            return False

        if not os.path.isfile(self.model_path):
            logger.debug(
                "LlamaProcessor.is_available: model file not found: %s",
                self.model_path,
            )
            return False

        # Check Python bindings
        if self._has_python_bindings():
            return True

        # Check CLI binary
        if os.path.isfile(settings.LLAMA_BIN) and os.access(
            settings.LLAMA_BIN, os.X_OK
        ):
            return True

        logger.debug(
            "LlamaProcessor.is_available: neither llama-cpp-python nor "
            "CLI binary found (bin=%s)",
            settings.LLAMA_BIN,
        )
        return False

    def build_prompt(
        self,
        page_data: Dict[str, Any],
        keyword_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Construct a compact prompt for the model from *page_data*.

        The prompt prioritises:
        - Page title (truncated to :data:`_MAX_TITLE_CHARS`)
        - Meta description (truncated to :data:`_MAX_META_CHARS`)
        - A truncated slice of ``cleaned_text`` (at most
          :data:`_MAX_BODY_CHARS` characters)
        - Top matched keywords from *keyword_data* (optional)

        The prompt instructs the model to return **only** compact JSON
        matching :data:`_JSON_SCHEMA_HINT`.

        Parameters
        ----------
        page_data : dict
            Output from ``parse_html()``.  Expected keys: ``page_title``,
            ``meta_description``, ``cleaned_text``.
        keyword_data : dict, optional
            Output from ``extract_keywords()``.  Used to add a hint about
            already-matched agricultural terms.

        Returns
        -------
        str
            Fully formatted prompt string ready to be sent to the model.
        """
        title = _safe_truncate(
            (page_data.get("page_title") or "").strip(), _MAX_TITLE_CHARS
        )
        meta = _safe_truncate(
            (page_data.get("meta_description") or "").strip(), _MAX_META_CHARS
        )
        body = _safe_truncate(
            (page_data.get("cleaned_text") or "").strip(), _MAX_BODY_CHARS
        )

        # Collect the top keyword hits (flat list, max 10) as a hint so the
        # model does not have to re-discover obvious agricultural terms.
        kw_hint = ""
        if keyword_data and isinstance(keyword_data, dict):
            matched: List[str] = []
            for hits in keyword_data.values():
                if isinstance(hits, list):
                    matched.extend(str(h) for h in hits[:5])
                elif isinstance(hits, dict):
                    matched.extend(str(k) for k in list(hits.keys())[:5])
            if matched:
                kw_hint = "Matched keywords: " + ", ".join(matched[:10]) + "\n"

        parts: List[str] = [_SYSTEM_INSTRUCTION, "", f"Schema: {_JSON_SCHEMA_HINT}", ""]
        if title:
            parts.append(f"Title: {title}")
        if meta:
            parts.append(f"Description: {meta}")
        if kw_hint:
            parts.append(kw_hint.rstrip())
        if body:
            parts.append(f"Content:\n{body}")
        parts.append("\nJSON:")

        return "\n".join(parts)

    def run_summary(
        self,
        page_data: Dict[str, Any],
        keyword_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the local model and return a structured summary dict.

        Tries the Python bindings first; falls back to the subprocess CLI.
        Returns a safe empty fallback if the model is unavailable or if
        inference fails for any reason.

        Parameters
        ----------
        page_data : dict
            Page data from ``parse_html()``.
        keyword_data : dict, optional
            Keyword hits from ``extract_keywords()``.

        Returns
        -------
        dict with keys:
            ``summary``              – str
            ``topics``               – list of str
            ``keywords``             – list of str
            ``entities``             – list of dicts with ``name`` and ``type``
            ``page_type_suggestion`` – str
            ``why_relevant``         – str
            ``llm_success``          – bool
            ``raw_response``         – str (raw model output for debugging)
        """
        if not self.is_available():
            logger.debug(
                "LlamaProcessor.run_summary: model not available; returning fallback"
            )
            return _make_fallback("model_unavailable")

        prompt = self.build_prompt(page_data, keyword_data)
        logger.debug(
            "LlamaProcessor.run_summary: prompt_length=%d chars", len(prompt)
        )

        raw: str = ""
        try:
            if self._has_python_bindings():
                raw = self._call_via_bindings(prompt)
            else:
                raw = self._call_via_subprocess(prompt)
        except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
            logger.warning("LlamaProcessor.run_summary: inference error: %s", exc)
            return _make_fallback(f"inference_error: {exc}")

        if not raw:
            logger.warning("LlamaProcessor.run_summary: empty response from model")
            return _make_fallback("empty_response")

        result = self.parse_json_response(raw)
        result["raw_response"] = raw
        return result

    def parse_json_response(self, raw_text: str) -> Dict[str, Any]:
        """
        Extract and parse the JSON object from *raw_text*.

        Handles common model output problems:

        - Extra prose before or after the JSON object
        - Markdown code fences (`` ```json … ``` ``)
        - Empty or whitespace-only responses

        On any parse failure the method logs a warning and returns the safe
        fallback structure with ``llm_success=False``.

        Parameters
        ----------
        raw_text : str
            Raw text output from the model.

        Returns
        -------
        dict
            Parsed result with ``llm_success`` set to ``True`` on success or
            ``False`` on failure.  Always contains all expected keys.
        """
        if not raw_text or not raw_text.strip():
            logger.warning("LlamaProcessor.parse_json_response: empty raw_text")
            return _make_fallback("empty_raw_text")

        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()

        # Find the outermost JSON object via brace matching
        json_str = self._extract_json_object(cleaned)
        if not json_str:
            logger.warning(
                "LlamaProcessor.parse_json_response: no JSON object found in: %r",
                raw_text[:200],
            )
            return _make_fallback("no_json_object")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning(
                "LlamaProcessor.parse_json_response: JSON decode error: %s  "
                "raw=%r",
                exc,
                json_str[:200],
            )
            return _make_fallback(f"json_decode_error: {exc}")

        if not isinstance(data, dict):
            logger.warning(
                "LlamaProcessor.parse_json_response: expected dict, got %s",
                type(data).__name__,
            )
            return _make_fallback("unexpected_json_type")

        # Normalise and coerce field types defensively
        result: Dict[str, Any] = {
            "summary": str(data.get("summary") or ""),
            "topics": self._to_str_list(data.get("topics")),
            "keywords": self._to_str_list(data.get("keywords")),
            "entities": self._to_entity_list(data.get("entities")),
            "page_type_suggestion": str(data.get("page_type_suggestion") or ""),
            "why_relevant": str(data.get("why_relevant") or ""),
            "llm_success": True,
            "raw_response": "",
        }
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _has_python_bindings(self) -> bool:
        """Return ``True`` if the ``llama_cpp`` package can be imported."""
        try:
            import llama_cpp  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_binding(self) -> Any:
        """
        Lazily load and cache the ``llama_cpp.Llama`` model instance.

        Returns ``None`` if loading fails for any reason.
        """
        if self._binding_loaded:
            return self._binding

        self._binding_loaded = True
        try:
            from llama_cpp import Llama  # type: ignore[import]

            logger.info(
                "LlamaProcessor: loading model via llama-cpp-python: %s",
                self.model_path,
            )
            self._binding = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
            logger.info("LlamaProcessor: model loaded successfully")
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            logger.error(
                "LlamaProcessor: failed to load model via bindings: %s", exc
            )
            self._binding = None

        return self._binding

    def _call_via_bindings(self, prompt: str) -> str:
        """
        Run inference using ``llama-cpp-python`` bindings.

        Parameters
        ----------
        prompt : str
            Fully formatted prompt string.

        Returns
        -------
        str
            Generated text, or an empty string on failure.
        """
        llm = self._get_binding()
        if llm is None:
            raise RuntimeError("llama-cpp-python binding failed to load")

        output = llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["\n\n", "```"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    def _call_via_subprocess(self, prompt: str) -> str:
        """
        Run inference by invoking the llama.cpp CLI binary as a subprocess.

        Parameters
        ----------
        prompt : str
            Fully formatted prompt string.

        Returns
        -------
        str
            Generated text, or an empty string on failure.

        Raises
        ------
        RuntimeError
            If the subprocess exits with a non-zero return code.
        subprocess.TimeoutExpired
            If the model takes longer than 120 seconds.
        """
        cmd: List[str] = [
            settings.LLAMA_BIN,
            "--model", self.model_path,
            "--ctx-size", str(self.n_ctx),
            "--threads", str(self.n_threads),
            "--n-predict", str(self.max_tokens),
            "--temp", str(self.temperature),
            "--n-gpu-layers", str(settings.LLAMA_GPU_LAYERS),
            "--prompt", prompt,
            "--log-disable",
        ]
        logger.debug("LlamaProcessor: subprocess cmd=%s", cmd[:4])

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"llama CLI exited with code {proc.returncode}: {proc.stderr[:200]}"
            )
        return proc.stdout.strip()

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """
        Return the first complete ``{…}`` JSON object found in *text*.

        Uses simple brace-counting rather than a full parser so it works even
        when the model emits prose before the opening brace.

        Returns an empty string if no complete object is found.
        """
        start = text.find("{")
        if start == -1:
            return ""
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return ""

    @staticmethod
    def _to_str_list(value: Any) -> List[str]:
        """Coerce *value* to a list of strings, ignoring non-string items."""
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if item is not None]

    @staticmethod
    def _to_entity_list(value: Any) -> List[Dict[str, str]]:
        """
        Coerce *value* to a list of ``{"name": str, "type": str}`` dicts.

        Silently drops malformed entries.
        """
        if not isinstance(value, list):
            return []
        result: List[Dict[str, str]] = []
        for item in value:
            if isinstance(item, dict):
                name = str(item.get("name") or "")
                kind = str(item.get("type") or "")
                if name:
                    result.append({"name": name, "type": kind})
        return result


# ---------------------------------------------------------------------------
# Module-level default instance (shared across the pipeline)
# ---------------------------------------------------------------------------

_default_processor: Optional[LlamaProcessor] = None


def _get_default_processor() -> LlamaProcessor:
    """Return (creating if necessary) the module-level default processor."""
    global _default_processor  # noqa: PLW0603
    if _default_processor is None:
        _default_processor = LlamaProcessor()
    return _default_processor


# ---------------------------------------------------------------------------
# Backward-compatible public API (used by phase1_pipeline.py)
# ---------------------------------------------------------------------------

def run_llama_summary(text: str) -> Dict[str, Any]:
    """
    Generate a short summary of *text* using the local llama.cpp model.

    This is a thin backward-compatible wrapper around
    :class:`LlamaProcessor`.  It accepts a plain text string (rather than
    a full ``page_data`` dict) and returns a dict with at least a
    ``summary`` key so existing pipeline code continues to work unchanged.

    Parameters
    ----------
    text : str
        Cleaned page text.

    Returns
    -------
    dict
        Always contains a ``summary`` key.  When the model is available
        the full structured result is returned; otherwise only
        ``{"summary": ""}`` is returned to maintain backward compatibility.
    """
    if not text:
        return {"summary": ""}

    processor = _get_default_processor()
    if not processor.is_available():
        logger.debug(
            "run_llama_summary: model not available; returning empty summary. "
            "Set AGRIINDEX_LLAMA_BIN, AGRIINDEX_LLAMA_MODEL, and ensure the "
            "model file exists to enable LLM summarisation."
        )
        return {"summary": ""}

    page_data = {"cleaned_text": text}
    result = processor.run_summary(page_data)
    return result


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    # Sample page_data that would come from parse_html()
    sample_page_data: Dict[str, Any] = {
        "page_title": "Grain Futures Rally on Drought Concerns in the Corn Belt",
        "meta_description": (
            "Corn and soybean futures surged Monday as traders cited worsening "
            "drought conditions across Iowa, Illinois, and Indiana."
        ),
        "cleaned_text": (
            "Chicago Board of Trade corn futures rose 3.2% on Monday following "
            "a USDA crop progress report showing 68% of the crop rated good-to-"
            "excellent, down from 74% last week. Analysts expect further "
            "volatility as La Niña conditions persist through the growing "
            "season. Ag-tech venture investors are watching the situation "
            "closely, with precision irrigation startups seeing increased "
            "inbound interest. Key players include Valley Irrigation, Lindsay "
            "Corporation, and several early-stage companies backed by "
            "Cultivian Sandbox Ventures."
        ),
    }

    sample_keyword_data: Dict[str, Any] = {
        "crops": ["corn", "soybean"],
        "investor_terms": ["venture", "investors"],
    }

    processor = LlamaProcessor()
    print("=" * 60)
    print(f"Model path : {processor.model_path}")
    print(f"Available  : {processor.is_available()}")
    print("=" * 60)

    # Always show prompt construction – does not require a model
    prompt = processor.build_prompt(sample_page_data, sample_keyword_data)
    print("\n--- Generated prompt ---")
    print(prompt)
    print("--- end of prompt ---\n")

    if processor.is_available():
        print("Running real inference …")
        result = processor.run_summary(sample_page_data, sample_keyword_data)
    else:
        print("Model not available – demonstrating parse_json_response() instead.")
        # Simulate what a well-behaved model might return
        simulated_raw = json.dumps(
            {
                "summary": (
                    "Corn and soybean futures rose sharply amid drought concerns "
                    "in the Corn Belt. USDA data showed declining crop ratings, "
                    "prompting investor interest in precision irrigation startups."
                ),
                "topics": ["commodity markets", "drought", "precision agriculture"],
                "keywords": ["corn futures", "USDA", "La Niña", "irrigation"],
                "entities": [
                    {"name": "Valley Irrigation", "type": "company"},
                    {"name": "Lindsay Corporation", "type": "company"},
                    {"name": "Cultivian Sandbox Ventures", "type": "investor"},
                    {"name": "USDA", "type": "organization"},
                ],
                "page_type_suggestion": "market_news",
                "why_relevant": (
                    "Covers Corn Belt commodity volatility and ag-tech investor activity."
                ),
            }
        )
        result = processor.parse_json_response(simulated_raw)
        result["raw_response"] = simulated_raw

    print("--- run_summary result ---")
    print(json.dumps(result, indent=2))
    sys.exit(0)
