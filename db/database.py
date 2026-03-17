"""
database.py
-----------
SQLite database layer for CornScout.

This module provides the :class:`AgriIndexDB` class which wraps a single
SQLite connection and exposes helper methods for every write operation
performed by the Phase 1 pipeline.

Design goals
------------
* **Portability** – SQL statements are kept in named constants or inline
  triple-quoted strings so they can be swapped for PostgreSQL equivalents
  with minimal effort.  SQLite-specific syntax (``INSERT OR IGNORE``,
  ``AUTOINCREMENT``) is isolated and clearly commented.
* **Safety** – every mutating operation runs inside an explicit transaction
  via the :meth:`AgriIndexDB._transaction` context manager so failures
  are rolled back automatically.
* **Simplicity** – the class holds a single long-lived connection; callers
  are responsible for closing it via :meth:`AgriIndexDB.close` or the
  context-manager protocol (``with AgriIndexDB(path) as db: ...``).

Later phases can extend or replace this module to:
- Support PostgreSQL by swapping ``sqlite3`` for ``psycopg2`` / ``asyncpg``.
- Add connection pooling.
- Add full-text search helpers.
- Add a re-crawl scheduling method (``get_stale_urls``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


# ---------------------------------------------------------------------------
# AgriIndexDB
# ---------------------------------------------------------------------------


class AgriIndexDB:
    """
    SQLite database interface for the CornScout project.

    Opens (or creates) a SQLite database file, initialises the schema from
    ``schema.sql``, and exposes helper methods for all Phase 1 write
    operations.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite ``.db`` file.  Defaults to ``agriindex.db`` in
        the current working directory.
    auto_init : bool
        When ``True`` (default), call :meth:`initialize_schema` immediately
        after opening the connection.

    Examples
    --------
    Use as a context manager to ensure the connection is always closed::

        with AgriIndexDB("agriindex.db") as db:
            url_id = db.upsert_url({"original_url": "https://example.com", ...})
    """

    def __init__(
        self,
        db_path: str = "agriindex.db",
        auto_init: bool = True,
    ) -> None:
        self._db_path = db_path
        logger.debug("Opening SQLite database at %s", db_path)
        self._conn = self._open_connection(db_path)
        if auto_init:
            self.initialize_schema()

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "AgriIndexDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @staticmethod
    def _open_connection(db_path: str) -> sqlite3.Connection:
        """
        Open a SQLite connection with sensible defaults.

        Sets WAL journal mode for better concurrent read performance and
        enables foreign-key enforcement.  The row factory is set to
        :class:`sqlite3.Row` so rows behave like read-only dictionaries.

        Parameters
        ----------
        db_path : str

        Returns
        -------
        sqlite3.Connection
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # WAL mode improves concurrent read throughput.
        # SQLite-specific: PostgreSQL does not need this pragma.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def close(self) -> None:
        """Close the underlying database connection."""
        try:
            self._conn.close()
            logger.debug("Database connection closed (%s)", self._db_path)
        except Exception:  # noqa: BLE001
            logger.warning("Error closing database connection", exc_info=True)

    # ------------------------------------------------------------------
    # Transaction helper
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """
        Yield the connection inside an explicit transaction.

        Commits on clean exit, rolls back on any exception.

        Yields
        ------
        sqlite3.Connection
        """
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def initialize_schema(self, schema_path: Optional[str] = None) -> None:
        """
        Create all tables and indexes defined in ``schema.sql``.

        Safe to call multiple times — every statement uses ``IF NOT EXISTS``.

        Parameters
        ----------
        schema_path : str, optional
            Override the default path to ``schema.sql``.
        """
        path = schema_path or _SCHEMA_PATH
        logger.debug("Initialising schema from %s", path)
        with open(path, "r", encoding="utf-8") as fh:
            sql = fh.read()
        # executescript issues an implicit COMMIT first, which is fine here.
        # SQLite-specific: PostgreSQL would use psycopg2's executescript equivalent.
        self._conn.executescript(sql)
        logger.info("Schema initialised (%s)", self._db_path)

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_json(value: Any) -> Optional[str]:
        """
        Serialise *value* to a JSON string.

        Returns ``None`` if *value* is ``None``.  Lists and dicts are
        serialised; strings are returned as-is (assumed already serialised).

        Parameters
        ----------
        value : any

        Returns
        -------
        str or None
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _from_json(value: Optional[str]) -> Any:
        """
        Deserialise a JSON string back to a Python object.

        Returns ``None`` if *value* is ``None`` or empty.

        Parameters
        ----------
        value : str or None

        Returns
        -------
        Any
        """
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning("Failed to deserialise JSON value: %r", value[:120])
            return value

    # ------------------------------------------------------------------
    # Timestamp helper
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        """Return the current UTC time as an ISO-8601 string."""
        return datetime.now(tz=timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Row → dict helper
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        """Convert a :class:`sqlite3.Row` to a plain ``dict``, or ``None``."""
        return dict(row) if row is not None else None

    # ------------------------------------------------------------------
    # search_discovery
    # ------------------------------------------------------------------

    def insert_search_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Bulk-insert raw DuckDuckGo search result rows into ``search_discovery``.

        Each dict in *results* should contain the keys produced by
        :func:`agriindex.search.duckduckgo_search.normalize_search_result`:
        ``query``, ``rank``, ``title``, ``url`` (or ``discovered_url``),
        ``snippet``, and optionally ``normalized_url`` and ``discovered_at``.

        Duplicate rows (same query + rank) are silently ignored.

        Parameters
        ----------
        results : list of dict
            Normalised search result records.
        """
        if not results:
            return

        now = self._now()
        rows = []
        for r in results:
            rows.append(
                (
                    r.get("query", ""),
                    r.get("rank"),
                    r.get("title"),
                    r.get("snippet"),
                    r.get("url") or r.get("discovered_url", ""),
                    r.get("normalized_url"),
                    r.get("discovered_at") or now,
                )
            )

        sql = """
            INSERT OR IGNORE INTO search_discovery
                (query, rank, title, snippet, discovered_url, normalized_url, discovered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        # SQLite-specific: "INSERT OR IGNORE" → PostgreSQL uses
        # "INSERT … ON CONFLICT DO NOTHING".
        with self._transaction() as conn:
            conn.executemany(sql, rows)
        logger.debug("Inserted %d search_discovery rows", len(rows))

    # ------------------------------------------------------------------
    # urls
    # ------------------------------------------------------------------

    def upsert_url(self, url_record: Dict[str, Any]) -> int:
        """
        Insert a new URL row or return the existing row's id.

        Uses ``INSERT OR IGNORE`` semantics: if a row with the same
        ``canonical_url`` already exists the existing row is returned
        unchanged and its ``id`` is returned.

        Parameters
        ----------
        url_record : dict
            Must contain ``original_url`` and ``canonical_url``.
            Optional keys: ``domain``, ``source_query``, ``search_rank``,
            ``discovered_at``, ``status``, ``content_hash``.

        Returns
        -------
        int
            Primary key of the (possibly pre-existing) ``urls`` row.
        """
        canonical_url = url_record["canonical_url"]
        now = self._now()

        # SQLite-specific: "INSERT OR IGNORE" → PostgreSQL uses
        # "INSERT … ON CONFLICT (canonical_url) DO NOTHING".
        sql_insert = """
            INSERT OR IGNORE INTO urls
                (original_url, canonical_url, domain,
                 source_query, search_rank,
                 discovered_at, status, content_hash,
                 created_at, updated_at)
            VALUES
                (:original_url, :canonical_url, :domain,
                 :source_query, :search_rank,
                 :discovered_at, :status, :content_hash,
                 :created_at, :updated_at)
        """
        params: Dict[str, Any] = {
            "original_url": url_record.get("original_url", canonical_url),
            "canonical_url": canonical_url,
            "domain": url_record.get("domain", ""),
            "source_query": url_record.get("source_query"),
            "search_rank": url_record.get("search_rank"),
            "discovered_at": url_record.get("discovered_at") or now,
            "status": url_record.get("status", "pending"),
            "content_hash": url_record.get("content_hash"),
            "created_at": now,
            "updated_at": now,
        }

        with self._transaction() as conn:
            conn.execute(sql_insert, params)

        row = self._conn.execute(
            "SELECT id FROM urls WHERE canonical_url = ?", (canonical_url,)
        ).fetchone()
        url_id: int = row["id"]
        logger.debug("upsert_url id=%d  canonical_url=%s", url_id, canonical_url)
        return url_id

    # ------------------------------------------------------------------
    # pages
    # ------------------------------------------------------------------

    def insert_or_update_page(self, page_record: Dict[str, Any]) -> int:
        """
        Insert a page record or update it if one already exists for the URL.

        If a row with the same ``url_id`` already exists, all provided
        fields are overwritten and ``processed_at`` is refreshed.

        Parameters
        ----------
        page_record : dict
            Must contain ``url_id``.  Optional keys match ``pages`` columns:
            ``title``, ``meta_description``, ``headings_json``,
            ``cleaned_text``, ``word_count``, ``summary``,
            ``topics_json``, ``keywords_json``, ``page_type``,
            ``relevance_cornbelt_ai``, ``relevance_investor``,
            ``confidence_score``, ``parse_success``, ``llm_success``,
            ``fetched_at``.  List/dict values for ``*_json`` fields are
            serialised automatically.

        Returns
        -------
        int
            Primary key of the inserted or updated ``pages`` row.
        """
        now = self._now()

        # Serialise JSON fields
        headings = self._to_json(page_record.get("headings_json"))
        topics = self._to_json(page_record.get("topics_json"))
        keywords = self._to_json(page_record.get("keywords_json"))

        # Check whether a page row already exists for this url_id
        existing = self._conn.execute(
            "SELECT id FROM pages WHERE url_id = ?", (page_record["url_id"],)
        ).fetchone()

        if existing is None:
            sql = """
                INSERT INTO pages
                    (url_id, title, meta_description, headings_json,
                     cleaned_text, word_count, summary,
                     topics_json, keywords_json, page_type,
                     relevance_cornbelt_ai, relevance_investor, confidence_score,
                     parse_success, llm_success,
                     fetched_at, processed_at)
                VALUES
                    (:url_id, :title, :meta_description, :headings_json,
                     :cleaned_text, :word_count, :summary,
                     :topics_json, :keywords_json, :page_type,
                     :relevance_cornbelt_ai, :relevance_investor, :confidence_score,
                     :parse_success, :llm_success,
                     :fetched_at, :processed_at)
            """
        else:
            sql = """
                UPDATE pages SET
                    title               = :title,
                    meta_description    = :meta_description,
                    headings_json       = :headings_json,
                    cleaned_text        = :cleaned_text,
                    word_count          = :word_count,
                    summary             = :summary,
                    topics_json         = :topics_json,
                    keywords_json       = :keywords_json,
                    page_type           = :page_type,
                    relevance_cornbelt_ai = :relevance_cornbelt_ai,
                    relevance_investor  = :relevance_investor,
                    confidence_score    = :confidence_score,
                    parse_success       = :parse_success,
                    llm_success         = :llm_success,
                    fetched_at          = :fetched_at,
                    processed_at        = :processed_at
                WHERE url_id = :url_id
            """

        params: Dict[str, Any] = {
            "url_id": page_record["url_id"],
            "title": page_record.get("title"),
            "meta_description": page_record.get("meta_description"),
            "headings_json": headings,
            "cleaned_text": page_record.get("cleaned_text"),
            "word_count": page_record.get("word_count"),
            "summary": page_record.get("summary"),
            "topics_json": topics,
            "keywords_json": keywords,
            "page_type": page_record.get("page_type"),
            "relevance_cornbelt_ai": page_record.get("relevance_cornbelt_ai"),
            "relevance_investor": page_record.get("relevance_investor"),
            "confidence_score": page_record.get("confidence_score"),
            "parse_success": int(bool(page_record.get("parse_success", False))),
            "llm_success": int(bool(page_record.get("llm_success", False))),
            "fetched_at": page_record.get("fetched_at") or now,
            "processed_at": now,
        }

        with self._transaction() as conn:
            cursor = conn.execute(sql, params)

        if existing is None:
            page_id: int = cursor.lastrowid  # type: ignore[assignment]
        else:
            page_id = existing["id"]

        logger.debug("insert_or_update_page page_id=%d  url_id=%d", page_id, page_record["url_id"])
        return page_id

    # ------------------------------------------------------------------
    # contacts
    # ------------------------------------------------------------------

    def insert_contacts(self, page_id: int, contacts: Dict[str, List[str]]) -> None:
        """
        Persist contact information (emails, phone numbers) found on a page.

        Duplicate (page_id, contact_type, contact_value) triplets are
        silently ignored.

        Parameters
        ----------
        page_id : int
            Primary key of the parent ``pages`` row.
        contacts : dict
            Expected structure::

                {
                    "emails":        ["alice@example.com", ...],
                    "phone_numbers": ["+1-800-555-0100", ...],
                }

            Any other key whose value is a list of strings is also accepted
            and stored under that key as ``contact_type``.
        """
        if not contacts:
            return

        rows: List[tuple] = []
        for contact_type, values in contacts.items():
            if not isinstance(values, list):
                continue
            # Normalise key → contact_type label
            label = "phone" if "phone" in contact_type else contact_type.rstrip("s")
            for value in values:
                if value:
                    rows.append((page_id, label, str(value)))

        if not rows:
            return

        sql = """
            INSERT OR IGNORE INTO contacts (page_id, contact_type, contact_value)
            VALUES (?, ?, ?)
        """
        # SQLite-specific: "INSERT OR IGNORE" → PostgreSQL:
        # "INSERT … ON CONFLICT DO NOTHING".
        with self._transaction() as conn:
            conn.executemany(sql, rows)
        logger.debug("Inserted %d contact row(s) for page_id=%d", len(rows), page_id)

    # ------------------------------------------------------------------
    # entities
    # ------------------------------------------------------------------

    def upsert_entities(
        self,
        page_id: int,
        entities: List[Dict[str, Any]],
    ) -> None:
        """
        Upsert named entities and create ``page_entities`` links.

        Each entity dict should contain at least ``canonical_name`` and
        ``entity_type``.  Optional keys: ``mention_text``, ``confidence``.

        The ``entities`` table is deduplicated on
        ``(canonical_name, entity_type)``; the ``page_entities`` table
        is deduplicated on ``(page_id, entity_id, mention_text)``.

        Parameters
        ----------
        page_id : int
            Primary key of the parent ``pages`` row.
        entities : list of dict
            Each dict may contain:
            ``canonical_name``, ``entity_type``, ``mention_text``,
            ``confidence``.
        """
        if not entities:
            return

        # SQLite-specific upsert via "INSERT OR IGNORE".
        # PostgreSQL equivalent: INSERT … ON CONFLICT DO NOTHING.
        sql_entity = """
            INSERT OR IGNORE INTO entities (canonical_name, entity_type)
            VALUES (?, ?)
        """
        sql_lookup = """
            SELECT id FROM entities
            WHERE canonical_name = ? AND entity_type = ?
        """
        sql_link = """
            INSERT OR IGNORE INTO page_entities
                (page_id, entity_id, mention_text, confidence)
            VALUES (?, ?, ?, ?)
        """

        with self._transaction() as conn:
            for ent in entities:
                cname = ent.get("canonical_name", "")
                etype = ent.get("entity_type", "")
                if not cname or not etype:
                    continue
                conn.execute(sql_entity, (cname, etype))
                entity_row = conn.execute(sql_lookup, (cname, etype)).fetchone()
                if entity_row is None:
                    continue
                conn.execute(
                    sql_link,
                    (
                        page_id,
                        entity_row["id"],
                        ent.get("mention_text"),
                        ent.get("confidence"),
                    ),
                )

        logger.debug(
            "upsert_entities: %d entity/entities linked to page_id=%d",
            len(entities),
            page_id,
        )

    # ------------------------------------------------------------------
    # URL status
    # ------------------------------------------------------------------

    def mark_url_status(self, url_id: int, status: str) -> None:
        """
        Update the ``status`` column for a ``urls`` row.

        Also refreshes ``updated_at`` to the current UTC time.

        Parameters
        ----------
        url_id : int
            Primary key of the ``urls`` row to update.
        status : str
            New status value, e.g. ``"fetched"``, ``"failed"``,
            ``"skipped"``, ``"pending"``.
        """
        sql = """
            UPDATE urls
            SET    status     = :status,
                   updated_at = :updated_at
            WHERE  id         = :id
        """
        with self._transaction() as conn:
            conn.execute(sql, {"status": status, "updated_at": self._now(), "id": url_id})
        logger.debug("mark_url_status url_id=%d  status=%s", url_id, status)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_unprocessed_urls(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Return URL rows whose status is ``"pending"``.

        Parameters
        ----------
        limit : int
            Maximum number of rows to return (default 50).

        Returns
        -------
        list of dict
            Each dict contains all ``urls`` columns.
        """
        sql = """
            SELECT *
            FROM   urls
            WHERE  status = 'pending'
            ORDER  BY discovered_at ASC
            LIMIT  :limit
        """
        rows = self._conn.execute(sql, {"limit": limit}).fetchall()
        result = [dict(r) for r in rows]
        logger.debug("get_unprocessed_urls returning %d row(s)", len(result))
        return result


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        demo_db_path = tmp.name

    print(f"\n=== AgriIndexDB demo  (db={demo_db_path}) ===\n")

    with AgriIndexDB(demo_db_path) as db:
        # ----------------------------------------------------------------
        # 1. Insert a sample search result
        # ----------------------------------------------------------------
        sample_results = [
            {
                "query": "corn disease management",
                "rank": 1,
                "title": "Managing Corn Diseases – Purdue Extension",
                "snippet": "Best practices for identifying and managing common corn diseases.",
                "url": "https://extension.purdue.edu/corn-disease-management",
                "normalized_url": "https://extension.purdue.edu/corn-disease-management",
            }
        ]
        db.insert_search_results(sample_results)
        print("✓  Inserted 1 search_discovery row")

        # ----------------------------------------------------------------
        # 2. Upsert a URL
        # ----------------------------------------------------------------
        url_id = db.upsert_url(
            {
                "original_url": "https://extension.purdue.edu/corn-disease-management",
                "canonical_url": "https://extension.purdue.edu/corn-disease-management",
                "domain": "extension.purdue.edu",
                "source_query": "corn disease management",
                "search_rank": 1,
                "status": "pending",
            }
        )
        print(f"✓  Upserted URL  url_id={url_id}")

        # ----------------------------------------------------------------
        # 3. Insert a sample page
        # ----------------------------------------------------------------
        page_id = db.insert_or_update_page(
            {
                "url_id": url_id,
                "title": "Managing Corn Diseases – Purdue Extension",
                "meta_description": "Comprehensive guide to corn disease management.",
                "headings_json": ["Introduction", "Gray Leaf Spot", "Northern Corn Leaf Blight"],
                "cleaned_text": "Corn diseases can significantly reduce yield ...",
                "word_count": 1450,
                "summary": "This page covers identification and management of common corn diseases.",
                "topics_json": ["plant pathology", "corn", "disease management"],
                "keywords_json": {"corn": 12, "disease": 8, "yield": 4},
                "page_type": "extension",
                "relevance_cornbelt_ai": 0.92,
                "relevance_investor": 0.45,
                "confidence_score": 0.88,
                "parse_success": True,
                "llm_success": True,
            }
        )
        print(f"✓  Inserted page  page_id={page_id}")

        # ----------------------------------------------------------------
        # 4. Insert contacts
        # ----------------------------------------------------------------
        db.insert_contacts(
            page_id,
            {"emails": ["contact@extension.purdue.edu"], "phone_numbers": ["+1-765-494-4773"]},
        )
        print("✓  Inserted contacts")

        # ----------------------------------------------------------------
        # 5. Upsert entities
        # ----------------------------------------------------------------
        db.upsert_entities(
            page_id,
            [
                {"canonical_name": "Purdue University", "entity_type": "ORG",
                 "mention_text": "Purdue Extension", "confidence": 0.97},
                {"canonical_name": "Indiana", "entity_type": "GPE",
                 "mention_text": "Indiana", "confidence": 0.99},
            ],
        )
        print("✓  Upserted entities")

        # ----------------------------------------------------------------
        # 6. Read unprocessed URLs (still "pending")
        # ----------------------------------------------------------------
        unprocessed = db.get_unprocessed_urls(limit=10)
        print(f"✓  Unprocessed URL(s): {len(unprocessed)}")
        for row in unprocessed:
            print(f"     id={row['id']}  status={row['status']}  url={row['canonical_url']}")

        # ----------------------------------------------------------------
        # 7. Mark the URL as fetched
        # ----------------------------------------------------------------
        db.mark_url_status(url_id, "fetched")
        print(f"✓  Marked url_id={url_id} as 'fetched'")

        # Confirm no more pending URLs
        still_pending = db.get_unprocessed_urls(limit=10)
        print(f"✓  Unprocessed URL(s) after mark: {len(still_pending)}")

    print("\n=== Demo complete ===\n")

    os.unlink(demo_db_path)
