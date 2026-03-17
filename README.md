# CornbeltScout – CornScout

**CornScout** is a modular agricultural web intelligence crawler designed to index ag-related webpages discovered through DuckDuckGo searches.

## Phase 1 Scope

Phase 1 focuses on:

- Discovering URLs via DuckDuckGo
- Filtering and normalising URLs
- Fetching page HTML
- Extracting readable text
- Running lightweight metadata extraction
- Generating short summaries using a local llama.cpp model *(stub in Phase 1)*
- Storing results in a SQLite database

## Project Structure

```
agriindex/
├── main.py                         # CLI entrypoint
├── config/
│   ├── settings.py                 # program configuration
│   ├── blocked_domains.yaml        # domains we do not crawl
│   └── keyword_sets.yaml           # agriculture keyword dictionaries
├── db/
│   ├── schema.sql                  # database schema
│   └── database.py                 # connection + helper functions
├── search/
│   └── duckduckgo_search.py        # run searches and return URLs
├── filters/
│   └── url_filters.py              # normalisation, deduplication, domain filtering
├── fetchers/
│   └── page_fetcher.py             # download HTML pages
├── parsers/
│   └── html_parser.py              # extract title, meta, readable text
├── extractors/
│   ├── contact_extractor.py        # email + phone regex
│   ├── keyword_extractor.py        # simple keyword detection
│   └── entity_extractor.py         # spaCy-based entity extraction (optional)
├── classifiers/
│   └── rule_classifier.py          # page type and relevance scoring
├── llm/
│   └── llama_processor.py          # summarisation via llama.cpp (stub)
├── pipeline/
│   └── phase1_pipeline.py          # orchestrates Phase 1 flow
└── utils/
    ├── hashing.py
    ├── logging_utils.py
    └── text_utils.py

docs/
└── phase1_checklist.md             # developer onboarding and validation checklist

scripts/
├── smoke_test.py                   # lightweight end-to-end smoke test
└── test_queries.py                 # bulk query runner for initial dataset builds

ui/
└── streamlit_app.py                # web dashboard
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the smoke test to verify your environment
python scripts/smoke_test.py

# 3. Run a search and process results
python agriindex/main.py --query "corn disease management" --limit 20

# Optional flags
python agriindex/main.py \
    --query "soybean yield 2024" \
    --limit 10 \
    --db my_data.db \
    --log-level DEBUG

# 4. Launch the web dashboard
streamlit run ui/streamlit_app.py
```

See [`docs/phase1_checklist.md`](docs/phase1_checklist.md) for a full
developer onboarding checklist, including expected output and common failure
points.

## Configuration

All tuneable parameters are in `agriindex/config/settings.py`.  Most can be
overridden with environment variables:

| Variable | Default | Description |
|---|---|---|
| `AGRIINDEX_DB_PATH` | `agriindex.db` | SQLite database file path |
| `AGRIINDEX_REQUEST_TIMEOUT` | `15` | HTTP request timeout (seconds) |
| `AGRIINDEX_SEARCH_LIMIT` | `20` | Default search result limit |
| `AGRIINDEX_LOG_LEVEL` | `INFO` | Logging level |
| `AGRIINDEX_LOG_FILE` | `agriindex.log` | Log file path (set to empty string to disable) |
| `AGRIINDEX_LLAMA_BIN` | `/usr/local/bin/llama` | llama.cpp binary path |
| `AGRIINDEX_LLAMA_MODEL` | `models/llama-ag-7b.gguf` | GGUF model file path |

## Optional Dependencies

| Package | Purpose |
|---|---|
| `spacy` + `en_core_web_sm` | Named entity extraction |
| `llama-cpp-python` | LLM summarisation (alternative to CLI) |

Install spaCy model:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Smoke Test

Run a lightweight end-to-end check before larger jobs:

```bash
python scripts/smoke_test.py
```

Expected output on success:

```
============================================================
CornScout – Phase 1 Smoke Test
============================================================
  [PASS] Settings load
  [PASS] Database init
  [PASS] Pipeline run
  [PASS] Records written

✓  ALL 4 CHECKS PASSED
============================================================
```

## Database Schema

SQLite tables:

- **urls** – every unique URL (canonical form, domain, first seen)
- **pages** – full page content and extracted metadata
- **entities** – unique spaCy named entities
- **page_entities** – many-to-many link between pages and entities
- **contacts** – emails and phone numbers
- **search_discovery** – records each URL's first discovery via a DDG query

## Pipeline Flow

```
run_search()
  → filter_urls()
  → fetch_page()
  → parse_html()
  → extract_contacts()
  → extract_keywords()
  → run_rule_classifier()
  → run_llama_summary()   (stub in Phase 1)
  → store_results()
```
