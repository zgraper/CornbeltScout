# CornScout – Phase 1 Developer Checklist

Use this checklist to verify that your local environment is correctly set up
and that the Phase 1 pipeline is working end-to-end before running larger jobs.

---

## 1. Environment Setup

- [ ] Python 3.8 or newer is installed (`python --version`)
- [ ] A virtual environment is created and activated:

  ```bash
  python -m venv venv
  source venv/bin/activate        # macOS / Linux
  # venv\Scripts\activate         # Windows
  ```

- [ ] You are running commands from the **repository root** (the directory that
  contains `agriindex/`, `scripts/`, `ui/`, etc.)

---

## 2. Required Packages

- [ ] Core dependencies are installed:

  ```bash
  pip install -r requirements.txt
  ```

- [ ] (Optional) spaCy model installed for named-entity extraction:

  ```bash
  pip install spacy
  python -m spacy download en_core_web_sm
  ```

- [ ] (Optional) `llama-cpp-python` installed for LLM summarisation:

  ```bash
  pip install llama-cpp-python
  ```

  > **Note:** LLM summarisation is a stub in Phase 1.  The pipeline runs
  > correctly without it; summaries will simply be empty.

---

## 3. Configuration (Optional)

All settings have sensible defaults and require no changes for local
development.  Override them via environment variables when needed:

| Variable | Default | Description |
|---|---|---|
| `AGRIINDEX_DB_PATH` | `agriindex.db` | SQLite database file path |
| `AGRIINDEX_REQUEST_TIMEOUT` | `15` | HTTP timeout in seconds |
| `AGRIINDEX_SEARCH_LIMIT` | `20` | Default result limit per query |
| `AGRIINDEX_LOG_LEVEL` | `INFO` | Logging verbosity |
| `AGRIINDEX_LOG_FILE` | `agriindex.log` | Log file path (empty = disabled) |
| `AGRIINDEX_LLAMA_ENABLED` | `true` | Set to `false` to skip LLM step |

A `.env` file in the project root is automatically loaded if `python-dotenv`
is installed.

---

## 4. How to Run the CLI

```bash
# Basic run – 20 results (default limit)
python agriindex/main.py --query "corn disease management"

# Custom limit and database path
python agriindex/main.py --query "soybean yield 2024" --limit 10 --db mydata.db

# Debug logging
python agriindex/main.py --query "ag tech startups" --log-level DEBUG
```

**Expected output (success):**

```
Done.  Processed 20 URL(s), 14 fetched successfully.
```

The exact numbers vary by network conditions and search results.

---

## 5. How to Run the Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

Then open <http://localhost:8501> in a browser.

**Expected behaviour:**

- The page title reads **"🌽 CornScout Web Intelligence Crawler"**.
- Entering a query and clicking **▶ Run Crawl** triggers the Phase 1 pipeline.
- After the crawl completes, the **Crawl Results Summary** metrics update.
- The **📄 Recent Pages** tab shows a table of fetched pages.

---

## 6. How to Run the Smoke Test

The smoke test is the fastest way to validate the environment before a larger
run.  It executes a minimal 3-URL query and verifies database writes.

```bash
python scripts/smoke_test.py
```

To use a specific database instead of the default `/tmp/cornscout_smoke.db`:

```bash
AGRIINDEX_DB_PATH=/tmp/my_test.db python scripts/smoke_test.py
```

---

## 7. Successful Output

A passing smoke test looks like this:

```
============================================================
CornScout – Phase 1 Smoke Test
============================================================
  database : /tmp/cornscout_smoke.db
  query    : 'corn disease management'
  limit    : 3

  [PASS] Settings load — db=/tmp/cornscout_smoke.db  limit=20
  [PASS] Database init — tables=['contacts', 'entities', 'page_entities', 'pages', 'search_discovery', 'urls']
  [PASS] Pipeline run — query='corn disease management'  urls_attempted=3
  [PASS] Records written — pages.count=2

============================================================
✓  ALL 4 CHECKS PASSED  (8.3s)
============================================================
```

> **Note:** `urls_attempted` and `pages.count` will vary.  As long as
> `pages.count` is ≥ 1, the pipeline is working correctly.

A successful CLI run produces a final summary line:

```
Done.  Processed 20 URL(s), 14 fetched successfully.
```

---

## 8. Running the Full Test Query Set

For a broader validation, run the predefined query set:

```bash
python scripts/test_queries.py
```

This runs 6 agricultural queries and logs per-query URL counts.

---

## 9. Common Failure Points

### `ModuleNotFoundError: No module named 'agriindex'`

You are not running the command from the repository root, or the virtual
environment is not activated.

```bash
# Fix: run from the repo root with venv active
cd /path/to/CornbeltScout   # the cloned repository directory
source venv/bin/activate
python scripts/smoke_test.py
```

### `ModuleNotFoundError: No module named 'duckduckgo_search'` (or similar)

Dependencies are not installed.

```bash
pip install -r requirements.txt
```

### `[FAIL] Records written — pages.count=0`

The DuckDuckGo search returned results, but all page fetches failed (e.g. due
to network restrictions, SSL errors, or all URLs being blocked).

- Check that you have outbound internet access.
- Run with `AGRIINDEX_LOG_LEVEL=DEBUG` to see detailed fetch errors.
- Try a different query or increase `--limit`.

### `[FAIL] Pipeline run — urls_attempted=0`

DuckDuckGo returned no results for the smoke-test query.

- This usually means a rate-limit or transient DDG API issue.  Wait a moment
  and retry.
- Alternatively set `_SMOKE_QUERY` in `scripts/smoke_test.py` to a broader
  search term.

### `sqlite3.OperationalError: unable to open database file`

The configured `AGRIINDEX_DB_PATH` is not writable.

```bash
# Use a writable path explicitly
AGRIINDEX_DB_PATH=/tmp/test.db python scripts/smoke_test.py
```

### `OSError: Could not open log file …`

The default `agriindex.log` cannot be written to the current directory (e.g.
read-only filesystem).  Disable file logging:

```bash
AGRIINDEX_LOG_FILE="" python scripts/smoke_test.py
```

### Streamlit page does not load / shows import errors

Ensure you run `streamlit run ui/streamlit_app.py` from the **repository root**
so that the `agriindex` package is importable.

---

## 10. Verifying the Database After a Run

```bash
sqlite3 agriindex.db "SELECT COUNT(*) FROM pages;"
sqlite3 agriindex.db "SELECT canonical_url, page_type FROM urls u JOIN pages p ON p.url_id = u.id LIMIT 5;"
```
