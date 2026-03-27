"""
smoke_test.py
-------------
Phase 1 smoke test for CornScout.

Runs a minimal end-to-end validation of the pipeline to confirm that the
local environment is correctly configured before larger runs.

What it checks
--------------
1. Settings load without errors.
2. The SQLite database initialises successfully.
3. A very small DuckDuckGo query runs through the full Phase 1 pipeline.
4. At least one record is written to the ``pages`` table.

Usage::

    python scripts/smoke_test.py

Environment variables (optional):
    AGRIINDEX_DB_PATH   – Override the database path (default: /tmp/cornscout_smoke.db).
    AGRIINDEX_LOG_LEVEL – Override the log level   (default: WARNING).

Exit codes:
    0 – all checks passed
    1 – one or more checks failed
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
import traceback

# Ensure the project root is importable when the script is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use a temporary database by default so the smoke test doesn't pollute the
# regular development database.
_DEFAULT_SMOKE_DB = "/tmp/cornscout_smoke.db"
os.environ.setdefault("AGRIINDEX_DB_PATH", _DEFAULT_SMOKE_DB)
# Keep log output quiet during the smoke test unless the caller opts in.
os.environ.setdefault("AGRIINDEX_LOG_LEVEL", "WARNING")

from agriindex.config.settings import load_settings  # noqa: E402
from agriindex.db.database import init_db  # noqa: E402
from agriindex.pipeline.phase1_pipeline import run_pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

_SMOKE_QUERY = "corn disease management"
_SMOKE_LIMIT = 3  # Keep the smoke test fast: fetch at most 3 URLs.


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def _check(label: str, passed: bool, detail: str = "") -> bool:
    """Print a single check result and return *passed*."""
    status = "PASS" if passed else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return passed


def _fail(label: str, exc: Exception) -> bool:
    """Print a FAIL result for *label* and, when DEBUG logging is enabled, print the traceback."""
    _check(label, False, str(exc))
    if os.environ.get("AGRIINDEX_LOG_LEVEL", "").upper() == "DEBUG":
        traceback.print_exc()
    return False


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_settings() -> bool:
    """Verify that settings load without raising an exception."""
    try:
        cfg = load_settings()
        return _check(
            "Settings load",
            True,
            f"db={cfg.DATABASE_PATH}  limit={cfg.DEFAULT_QUERY_LIMIT}",
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("Settings load", exc)


def check_db_init(db_path: str) -> bool:
    """Verify that the database initialises and all expected tables exist."""
    try:
        init_db(db_path=db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        expected = {"contacts", "entities", "page_entities", "pages", "search_discovery", "urls"}
        missing = expected - tables
        if missing:
            return _check("Database init", False, f"missing tables: {missing}")
        return _check("Database init", True, f"tables={sorted(tables)}")
    except Exception as exc:  # noqa: BLE001
        return _fail("Database init", exc)


def check_pipeline(db_path: str) -> tuple[bool, int]:
    """Run a tiny pipeline query and return (passed, url_count)."""
    try:
        results = run_pipeline(
            query=_SMOKE_QUERY,
            limit=_SMOKE_LIMIT,
            db_path=db_path,
        )
        url_count = len(results)
        passed = _check(
            "Pipeline run",
            url_count > 0,
            f"query={_SMOKE_QUERY!r}  urls_attempted={url_count}",
        )
        return passed, url_count
    except Exception as exc:  # noqa: BLE001
        _fail("Pipeline run", exc)
        return False, 0


def check_records_written(db_path: str) -> bool:
    """Verify that at least one page record was written to the database."""
    try:
        conn = sqlite3.connect(db_path)
        (count,) = conn.execute("SELECT COUNT(*) FROM pages").fetchone()
        conn.close()
        passed = count > 0
        return _check(
            "Records written",
            passed,
            f"pages.count={count}",
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("Records written", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all smoke-test checks and print a pass/fail summary.

    Returns
    -------
    int
        Exit code: 0 on success, 1 if any check failed.
    """
    cfg = load_settings()
    db_path = cfg.DATABASE_PATH

    print("=" * 60)
    print("CornScout – Phase 1 Smoke Test")
    print("=" * 60)
    print(f"  database : {db_path}")
    print(f"  query    : {_SMOKE_QUERY!r}")
    print(f"  limit    : {_SMOKE_LIMIT}")
    print()

    start = time.monotonic()

    results: list[bool] = [
        check_settings(),
        check_db_init(db_path),
    ]

    pipeline_ok, _ = check_pipeline(db_path)
    results.append(pipeline_ok)

    results.append(check_records_written(db_path))

    elapsed = time.monotonic() - start
    passed_count = sum(results)
    total = len(results)
    all_passed = passed_count == total

    print()
    print("=" * 60)
    if all_passed:
        print(f"✓  ALL {total} CHECKS PASSED  ({elapsed:.1f}s)")
    else:
        failed = total - passed_count
        print(f"✗  {failed} of {total} CHECKS FAILED  ({elapsed:.1f}s)")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
