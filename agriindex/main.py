"""
main.py
-------
AgriIndex command-line entrypoint.

Usage
-----
Run a DuckDuckGo search and process the results through the Phase 1 pipeline::

    python main.py --query "corn disease management" --limit 20

Options
-------
--query   TEXT   Search query to run (required).
--limit   INT    Maximum number of URLs to process (default: 20).
--db      PATH   Path to the SQLite database file (default: agriindex.db).
--log-level LEVEL  Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO).
"""

import argparse
import os
import sys

# Ensure the repo root (parent of the agriindex package) is importable
# when running this file directly: `python agriindex/main.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="agriindex",
        description=(
            "AgriIndex Phase 1 – discover and index agricultural web pages "
            "via DuckDuckGo searches."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "corn disease management" --limit 20
  python main.py --query "soybean yield 2024" --limit 10 --db mydata.db
        """,
    )

    parser.add_argument(
        "--query", "-q",
        required=True,
        help="DuckDuckGo search query (e.g. 'corn disease management').",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=20,
        help="Maximum number of search results to process (default: 20).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the SQLite database file (default: agriindex.db).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level (default: INFO).",
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    """
    Main entrypoint for the AgriIndex CLI.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    args = parse_args(argv)

    # Apply log level before importing pipeline (which imports logging_utils)
    os.environ.setdefault("AGRIINDEX_LOG_LEVEL", args.log_level)
    if args.log_level:
        os.environ["AGRIINDEX_LOG_LEVEL"] = args.log_level

    # Late import so that LOG_LEVEL env var is set first
    from agriindex.pipeline.phase1_pipeline import run_pipeline
    from agriindex.utils.logging_utils import get_logger

    logger = get_logger("agriindex.main")
    logger.info(
        "Starting AgriIndex  query=%r  limit=%d  db=%s",
        args.query,
        args.limit,
        args.db or "agriindex.db",
    )

    try:
        results = run_pipeline(
            query=args.query,
            limit=args.limit,
            db_path=args.db,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Fatal error: {exc}", file=sys.stderr)
        logger.exception("Unhandled exception in pipeline")
        return 1

    # Summary report
    successful = [r for r in results if not r.get("fetch_error")]
    print(
        f"\nDone.  Processed {len(results)} URL(s), "
        f"{len(successful)} fetched successfully."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
