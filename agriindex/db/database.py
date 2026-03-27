"""
database.py
-----------
SQLite connection management and helper functions for CornScout.

This module provides:
- get_connection()      – returns a configured sqlite3.Connection
- init_db()             – creates all tables from schema.sql if they don't exist
- insert_url()          – insert or ignore a canonical URL record
- insert_page()         – upsert a full page record (insert or update by url_id)
- insert_entities()     – bulk-insert spaCy entities and page_entity links
- insert_contacts()     – bulk-insert emails / phone numbers
- insert_discovery()    – record that a URL was found by a specific DDG query
- mark_url_status()     – update the processing status of a URL
- url_exists()          – check whether a canonical URL is already known
- get_url_id()          – return the PK for a canonical URL
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from agriindex.config import settings
from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _schema_path() -> str:
    """Return the absolute path to schema.sql."""
    return os.path.join(os.path.dirname(__file__), "schema.sql")


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Open (or create) the CornScout SQLite database and return a connection.

    The connection is configured with:
    - WAL journal mode for better concurrent read performance
    - Foreign key enforcement
    - Row factory set to sqlite3.Row for dict-like row access

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite file.  Defaults to ``settings.DB_PATH``.

    Returns
    -------
    sqlite3.Connection
    """
    path = db_path or settings.DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialise the database by executing schema.sql.

    Safe to call multiple times — all CREATE TABLE statements use
    ``IF NOT EXISTS``.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite file.  Defaults to ``settings.DB_PATH``.
    """
    schema_file = _schema_path()
    with open(schema_file, "r", encoding="utf-8") as fh:
        sql = fh.read()

    conn = get_connection(db_path)
    with conn:
        conn.executescript(sql)
    conn.close()


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def url_exists(canonical_url: str, db_path: Optional[str] = None) -> bool:
    """
    Return True if *canonical_url* is already present in the ``urls`` table.

    Parameters
    ----------
    canonical_url : str
    db_path : str, optional
    """
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT id FROM urls WHERE canonical_url = ?", (canonical_url,)
    ).fetchone()
    conn.close()
    return row is not None


def get_url_id(canonical_url: str, db_path: Optional[str] = None) -> Optional[int]:
    """
    Return the primary key for *canonical_url*, or None if not found.

    Parameters
    ----------
    canonical_url : str
    db_path : str, optional
    """
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT id FROM urls WHERE canonical_url = ?", (canonical_url,)
    ).fetchone()
    conn.close()
    return row["id"] if row else None


def insert_url(
    url: str,
    canonical_url: str,
    domain: str,
    db_path: Optional[str] = None,
) -> int:
    """
    Insert a URL into the ``urls`` table.

    If *canonical_url* already exists the existing row is returned unchanged
    (INSERT OR IGNORE semantics).

    Parameters
    ----------
    url : str
        Raw URL as returned by the search engine.
    canonical_url : str
        Normalised / deduplicated form of the URL.
    domain : str
        Extracted domain (e.g. ``"example.com"``).
    db_path : str, optional

    Returns
    -------
    int
        The primary key of the (possibly pre-existing) row.
    """
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO urls (url, canonical_url, domain, first_seen)
            VALUES (?, ?, ?, ?)
            """,
            (url, canonical_url, domain, _now_iso()),
        )
    row = conn.execute(
        "SELECT id FROM urls WHERE canonical_url = ?", (canonical_url,)
    ).fetchone()
    conn.close()
    return row["id"]


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------

def insert_page(page_data: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """
    Upsert a page record into the ``pages`` table.

    On repeated pipeline runs for the same URL the existing row is updated
    rather than a new duplicate row being inserted (the ``pages`` table has
    a UNIQUE constraint on ``url_id``).

    Parameters
    ----------
    page_data : dict
        Keys map directly to ``pages`` columns.  Required key: ``url_id``.
        JSON-serialisable values (topics, keywords) will be serialised
        automatically if they are passed as Python objects.
    db_path : str, optional

    Returns
    -------
    int
        The primary key of the row (new or pre-existing).
    """
    # Serialise any dict/list values to JSON strings
    topics = page_data.get("topics")
    keywords = page_data.get("keywords")
    if isinstance(topics, (list, dict)):
        topics = json.dumps(topics)
    if isinstance(keywords, (list, dict)):
        keywords = json.dumps(keywords)

    fetched_at = page_data.get("fetched_at", _now_iso())
    params = {
        "url_id": page_data["url_id"],
        "fetched_at": fetched_at,
        "http_status": page_data.get("http_status"),
        "page_title": page_data.get("page_title"),
        "meta_description": page_data.get("meta_description"),
        "cleaned_text": page_data.get("cleaned_text"),
        "word_count": page_data.get("word_count"),
        "summary": page_data.get("summary"),
        "topics": topics,
        "keywords": keywords,
        "relevance_cornbelt_ai": page_data.get("relevance_cornbelt_ai"),
        "relevance_investor": page_data.get("relevance_investor"),
        "page_type": page_data.get("page_type"),
        "content_hash": page_data.get("content_hash"),
    }

    conn = get_connection(db_path)
    with conn:
        # Insert if url_id is new; the UNIQUE(url_id) constraint silently
        # ignores the insert when a row already exists for this URL.
        conn.execute(
            """
            INSERT OR IGNORE INTO pages (
                url_id, fetched_at, http_status,
                page_title, meta_description, cleaned_text, word_count,
                summary, topics, keywords,
                relevance_cornbelt_ai, relevance_investor,
                page_type, content_hash
            ) VALUES (
                :url_id, :fetched_at, :http_status,
                :page_title, :meta_description, :cleaned_text, :word_count,
                :summary, :topics, :keywords,
                :relevance_cornbelt_ai, :relevance_investor,
                :page_type, :content_hash
            )
            """,
            params,
        )
        # Always update so that re-crawls refresh the stored content.
        conn.execute(
            """
            UPDATE pages SET
                fetched_at            = :fetched_at,
                http_status           = :http_status,
                page_title            = :page_title,
                meta_description      = :meta_description,
                cleaned_text          = :cleaned_text,
                word_count            = :word_count,
                summary               = :summary,
                topics                = :topics,
                keywords              = :keywords,
                relevance_cornbelt_ai = :relevance_cornbelt_ai,
                relevance_investor    = :relevance_investor,
                page_type             = :page_type,
                content_hash          = :content_hash
            WHERE url_id = :url_id
            """,
            params,
        )
        row = conn.execute(
            "SELECT id FROM pages WHERE url_id = ?", (params["url_id"],)
        ).fetchone()
    if row is None:
        raise RuntimeError(
            f"insert_page: could not retrieve page row for url_id={params['url_id']}"
        )
    page_id = row["id"]
    conn.close()
    return page_id


# ---------------------------------------------------------------------------
# Entity helpers
# ---------------------------------------------------------------------------

def insert_entities(
    page_id: int,
    entities: List[Tuple[str, str]],
    db_path: Optional[str] = None,
) -> None:
    """
    Insert named entities and link them to a page.

    Parameters
    ----------
    page_id : int
        Primary key of the parent ``pages`` row.
    entities : list of (text, label) tuples
        ``label`` is a spaCy entity label such as ``"ORG"``, ``"GPE"``, etc.
    db_path : str, optional
    """
    if not entities:
        return

    conn = get_connection(db_path)
    with conn:
        for text, label in entities:
            conn.execute(
                "INSERT OR IGNORE INTO entities (text, label) VALUES (?, ?)",
                (text, label),
            )
            entity_row = conn.execute(
                "SELECT id FROM entities WHERE text = ? AND label = ?",
                (text, label),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO page_entities (page_id, entity_id, count)
                VALUES (?, ?, 1)
                ON CONFLICT (page_id, entity_id) DO UPDATE SET count = count + 1
                """,
                (page_id, entity_row["id"]),
            )
    conn.close()


# ---------------------------------------------------------------------------
# Contact helpers
# ---------------------------------------------------------------------------

def insert_contacts(
    page_id: int,
    contacts: List[Tuple[str, str]],
    db_path: Optional[str] = None,
) -> None:
    """
    Insert contact information (emails, phone numbers) for a page.

    Parameters
    ----------
    page_id : int
        Primary key of the parent ``pages`` row.
    contacts : list of (type, value) tuples
        ``type`` is either ``"email"`` or ``"phone"``.
    db_path : str, optional
    """
    if not contacts:
        return

    conn = get_connection(db_path)
    with conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO contacts (page_id, type, value)
            VALUES (?, ?, ?)
            """,
            [(page_id, ctype, cvalue) for ctype, cvalue in contacts],
        )
    conn.close()


# ---------------------------------------------------------------------------
# Search discovery helpers
# ---------------------------------------------------------------------------

def insert_discovery(
    url_id: int,
    query: str,
    result_rank: Optional[int] = None,
    db_path: Optional[str] = None,
) -> None:
    """
    Record that a URL was discovered via a specific DDG query.

    Uses INSERT OR IGNORE so that repeated pipeline runs for the same
    (url_id, query) pair do not create duplicate discovery records.

    Parameters
    ----------
    url_id : int
        Primary key of the ``urls`` row.
    query : str
        The search query string used.
    result_rank : int, optional
        Position (1-based) of the URL in the search result list.
    db_path : str, optional
    """
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO search_discovery (url_id, query, result_rank, discovered_at)
            VALUES (?, ?, ?, ?)
            """,
            (url_id, query, result_rank, _now_iso()),
        )
    conn.close()


# ---------------------------------------------------------------------------
# URL status helpers
# ---------------------------------------------------------------------------

def mark_url_status(
    url_id: int,
    status: str,
    db_path: Optional[str] = None,
) -> None:
    """
    Update the processing status of a URL record.

    Parameters
    ----------
    url_id : int
        Primary key of the ``urls`` row.
    status : str
        New status value.  Expected values: ``"pending"``, ``"fetched"``,
        ``"failed"``, ``"skipped"``.
    db_path : str, optional
    """
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "UPDATE urls SET status = ?, last_crawled = ? WHERE id = ?",
            (status, _now_iso(), url_id),
        )
    conn.close()
    logger.debug("mark_url_status: url_id=%d  status=%r", url_id, status)
