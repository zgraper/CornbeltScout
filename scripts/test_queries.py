"""
test_queries.py
---------------
Test runner script for the CornScout Phase 1 pipeline.

Runs a predefined list of agricultural queries through the pipeline to quickly
build an initial dataset.  Results are stored in the configured SQLite database.

Usage::

    python scripts/test_queries.py

Environment variables (optional):
    AGRIINDEX_DB_PATH   – Override the database path (default: agriindex.db).
    AGRIINDEX_LOG_LEVEL – Override the log level   (default: INFO).
"""

import os
import sys

# Ensure the project root is on the path when the script is run directly
# (e.g. ``python scripts/test_queries.py`` from the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agriindex.config.settings import load_settings  # noqa: E402
from agriindex.db.database import init_db  # noqa: E402
from agriindex.pipeline.phase1_pipeline import run_pipeline  # noqa: E402
from agriindex.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Queries to run
# ---------------------------------------------------------------------------

QUERIES = [
    "corn disease",
    "soybean aphid management",
    "precision agriculture startups",
    "ag venture capital funds",
    "soil nutrient deficiency corn",
    "crop yield analytics companies",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run each query through the Phase 1 pipeline and store results.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on unhandled error.
    """
    # 1. Load settings
    cfg = load_settings()
    logger.info("Settings loaded  db=%s  limit=%d", cfg.DATABASE_PATH, cfg.DEFAULT_QUERY_LIMIT)

    # 2. Initialize the database
    init_db(db_path=cfg.DATABASE_PATH)
    logger.info("Database initialized: %s", cfg.DATABASE_PATH)

    # 3. Initialize the pipeline (imported above) and run it for each query
    total_discovered = 0
    total_processed = 0
    failed_queries: list = []

    for query in QUERIES:
        logger.info("─── Query: %r ───", query)

        try:
            # 4. Run the pipeline (results are stored inside run_pipeline).
            # Each element is a context dict built throughout the pipeline; the
            # "fetch_error" key is set to a non-None string when the HTTP fetch
            # failed, or None / absent when the page was retrieved successfully.
            results = run_pipeline(
                query=query,
                limit=cfg.DEFAULT_QUERY_LIMIT,
                db_path=cfg.DATABASE_PATH,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Pipeline failed for query=%r: %s", query, exc)
            failed_queries.append(query)
            continue

        # 5. Logging summary for this query
        urls_discovered = len(results)
        urls_processed = sum(1 for r in results if not r.get("fetch_error"))

        logger.info(
            "Query=%r  urls_discovered=%d  urls_processed=%d",
            query,
            urls_discovered,
            urls_processed,
        )

        total_discovered += urls_discovered
        total_processed += urls_processed

    logger.info(
        "All queries complete  total_discovered=%d  total_processed=%d",
        total_discovered,
        total_processed,
    )

    if failed_queries:
        logger.warning("Queries that failed: %s", failed_queries)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
