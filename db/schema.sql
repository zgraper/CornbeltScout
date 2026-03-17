-- schema.sql
-- ----------
-- AgriIndex SQLite database schema.
--
-- Tables
-- ------
--   urls             – Every unique URL ever discovered (canonical + original)
--   pages            – Full page content and extracted metadata
--   entities         – Unique named entities (canonical form + type)
--   page_entities    – Many-to-many link between pages and entities
--   contacts         – Emails and phone numbers found on pages
--   search_discovery – Raw DuckDuckGo result rows, one per (query, rank) pair
--
-- Design notes
-- ------------
-- * All tables use INTEGER PRIMARY KEY AUTOINCREMENT so they are easy to
--   migrate to a PostgreSQL SERIAL / BIGSERIAL column.
-- * Timestamps are stored as ISO-8601 text strings (UTC).
-- * JSON arrays/objects are stored as TEXT (headings_json, topics_json, etc.).
-- * All CREATE TABLE / CREATE INDEX statements use IF NOT EXISTS so the file
--   is safe to re-execute multiple times.

-- ---------------------------------------------------------------------------
-- urls
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS urls (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    original_url   TEXT    NOT NULL,           -- raw URL as returned by search engine
    canonical_url  TEXT    NOT NULL UNIQUE,    -- normalised / deduplicated form
    domain         TEXT    NOT NULL,
    source_query   TEXT,                       -- DDG query that first found this URL
    search_rank    INTEGER,                    -- 1-based position in search results
    discovered_at  TEXT    NOT NULL,           -- ISO-8601 UTC timestamp
    status         TEXT    NOT NULL DEFAULT 'pending',
                                               -- pending | fetched | failed | skipped
    content_hash   TEXT,                       -- SHA-256 of cleaned_text (dedup)
    created_at     TEXT    NOT NULL,           -- ISO-8601 UTC timestamp (row created)
    updated_at     TEXT    NOT NULL            -- ISO-8601 UTC timestamp (last update)
);

CREATE INDEX IF NOT EXISTS idx_urls_domain        ON urls (domain);
CREATE INDEX IF NOT EXISTS idx_urls_canonical_url ON urls (canonical_url);
CREATE INDEX IF NOT EXISTS idx_urls_status        ON urls (status);

-- ---------------------------------------------------------------------------
-- pages
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pages (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    url_id                INTEGER NOT NULL REFERENCES urls (id) ON DELETE CASCADE,
    title                 TEXT,
    meta_description      TEXT,
    headings_json         TEXT,     -- JSON array of heading strings
    cleaned_text          TEXT,
    word_count            INTEGER,
    summary               TEXT,     -- LLM-generated summary
    topics_json           TEXT,     -- JSON array of topic strings
    keywords_json         TEXT,     -- JSON object: {keyword_set: hit_count, …}
    page_type             TEXT,     -- e.g. "news", "product", "research"
    relevance_cornbelt_ai REAL,     -- 0.0 – 1.0
    relevance_investor    REAL,     -- 0.0 – 1.0
    confidence_score      REAL,     -- overall confidence in classification
    parse_success         INTEGER   NOT NULL DEFAULT 0,  -- 1 = HTML parsed OK
    llm_success           INTEGER   NOT NULL DEFAULT 0,  -- 1 = LLM summary OK
    fetched_at            TEXT,     -- ISO-8601 UTC timestamp
    processed_at          TEXT      -- ISO-8601 UTC timestamp
);

CREATE INDEX IF NOT EXISTS idx_pages_url_id    ON pages (url_id);
CREATE INDEX IF NOT EXISTS idx_pages_page_type ON pages (page_type);

-- ---------------------------------------------------------------------------
-- entities
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entities (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT    NOT NULL,
    entity_type    TEXT    NOT NULL,   -- e.g. ORG, GPE, PERSON, PRODUCT
    UNIQUE (canonical_name, entity_type)
);

-- ---------------------------------------------------------------------------
-- page_entities
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS page_entities (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id      INTEGER NOT NULL REFERENCES pages (id)    ON DELETE CASCADE,
    entity_id    INTEGER NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    mention_text TEXT,     -- exact text as it appeared on the page
    confidence   REAL,     -- 0.0 – 1.0 extraction confidence
    UNIQUE (page_id, entity_id, mention_text)
);

CREATE INDEX IF NOT EXISTS idx_page_entities_page_id   ON page_entities (page_id);
CREATE INDEX IF NOT EXISTS idx_page_entities_entity_id ON page_entities (entity_id);

-- ---------------------------------------------------------------------------
-- contacts
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS contacts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id       INTEGER NOT NULL REFERENCES pages (id) ON DELETE CASCADE,
    contact_type  TEXT    NOT NULL,   -- "email" or "phone"
    contact_value TEXT    NOT NULL,
    UNIQUE (page_id, contact_type, contact_value)
);

CREATE INDEX IF NOT EXISTS idx_contacts_page_id ON contacts (page_id);

-- ---------------------------------------------------------------------------
-- search_discovery
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS search_discovery (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    query          TEXT    NOT NULL,    -- DuckDuckGo search query string
    rank           INTEGER,             -- 1-based position in result list
    title          TEXT,                -- page title from search snippet
    snippet        TEXT,                -- short description from search result
    discovered_url TEXT    NOT NULL,    -- raw URL as returned by search engine
    normalized_url TEXT,                -- canonical / normalised form of the URL
    discovered_at  TEXT    NOT NULL     -- ISO-8601 UTC timestamp
);

CREATE INDEX IF NOT EXISTS idx_search_discovery_query          ON search_discovery (query);
CREATE INDEX IF NOT EXISTS idx_search_discovery_normalized_url ON search_discovery (normalized_url);
