# Frontend and Backend Code Flow Deep Dive

## 1. End-to-End Runtime Flow

At runtime, the project behaves in this sequence:

1. FastAPI backend starts (`backend/main.py`) with lifespan hook.
2. Backend checks vector DB readiness via `get_stats()`.
3. If DB is empty, background scraper/indexing is launched.
4. Frontend loads and fetches `/stats`.
5. User submits a question in chat UI.
6. Frontend sends `POST /chat` with `{question}`.
7. Backend routes question through policy guardrails.
8. If no policy match, backend runs RAG pipeline in `brain.py`.
9. Backend returns `{answer, sources}`.
10. Frontend renders assistant message and source links.

## 2. Backend Detailed Flow (`backend/main.py`)

## 2.1 Startup and lifecycle

Key parts:

- `load_dotenv()` loads environment values.
- `DEFAULT_STARTUP_URLS` defines seed URLs.
- `lifespan(...)` is called by FastAPI at startup.

Startup logic:

1. call `get_stats()`.
2. if chunks exist -> mark status loaded, skip scrape.
3. if no chunks -> start background thread with `_run_scraper_background(...)`.

Operational reason:

- API remains responsive while ingestion runs in background.

## 2.2 Scrape worker path

`_run_scraper_background(...)`:

1. sets `scraping_status.running=True`.
2. calls `run_scraper_for_urls(...)`.
3. if documents found -> `store_documents(...)`.
4. updates human-readable status messages.
5. finalizes status in `finally` block.

Design note:

- status dict enables basic observability without external task queue.

## 2.3 Mandatory policy router

Before RAG, `/chat` calls:

- `_route_mandatory_case(question)`
- which delegates to `_detect_mandatory_case(question)`.

If matched, backend returns deterministic `MANDATORY_RESPONSES[case]` with no retrieval.

This prevents RAG from being used where fixed policy behavior is preferred.

## 2.4 API endpoint behavior

- `GET /stats`: returns chunk count and readiness.
- `GET /health`: liveness probe.
- `POST /chat`: main answer pipeline.
- `POST /chat-policy-test`: debug policy matching only.
- `GET /scrape-status`: background scrape progress.
- `POST /rescrape`: rerun startup URL scrape.
- `POST /scrape-urls`: targeted URL scrape.

## 3. Scraping and Normalization Flow (`backend/scraper.py`)

## 3.1 Page ingestion steps

For each URL:

1. fetch HTML (`requests.get`) with browser-like headers,
2. decode robustly with fallback encoding,
3. parse with `BeautifulSoup`,
4. extract tables as markdown,
5. extract cleaned main text,
6. fallback to heading/list extraction if main text is empty,
7. discover linked PDFs/images for optional OCR,
8. combine all sections into one document payload.

Output document shape:

- `url`
- `title`
- `content`

## 3.2 Why table extraction matters in flow

Fee and structure pages are often represented in HTML tables. If flattened naively, row/column semantics are lost and LLM responses degrade. Markdown conversion preserves enough structure for better summarization.

## 4. Retrieval and Generation Flow (`backend/brain.py`)

## 4.1 Storage/index path

`store_documents(documents)`:

1. split each document with `chunk_text` (1000/200),
2. generate embeddings,
3. upsert IDs/documents/embeddings/metadata in batches.

Metadata includes `url`, `title`, `chunk_index` for traceability.

## 4.2 Query path

`generate_response(question)` does:

1. detect intent class (fees/programs/hostel/overview),
2. choose retrieval depth (`top_k` dynamic),
3. retrieve semantic top chunks,
4. add intent-pinned chunks (for fees/programs/hostel),
5. filter excluded sources (`deans/directors/team-cdpc`),
6. apply relevance filtering with fuzzy overlap + distance fallback,
7. apply intent-specific post-filters (fee-only, program-only, hostel-only as needed),
8. build prompt with context,
9. call HF router,
10. on failure, build retrieval-only fallback answer,
11. return answer with deduplicated source URLs.

## 4.3 Relevance subsystem behavior

Primary relevance check combines:

- lexical token overlap,
- fuzzy token similarity for typo tolerance,
- embedding distance threshold.

This hybrid prevents both over-strict lexical misses and purely semantic drift.

## 4.4 Intent-specific flow branches

Fees branch:

- broadens query terms,
- increases retrieval depth,
- pins fee pages,
- asks model for structured fee output,
- uses larger token budget.

Programs branch:

- typo-tolerant trigger (`program/progrms/...`),
- pins `program-list` URLs,
- asks model for grouped UG/PG/PhD coverage.

Hostel-fees branch:

- detects hostel + fee combination,
- pins hostel page,
- avoids accidental restriction to academic fee pages.

Overview branch:

- supports broad asks like "tell me about rbu/college",
- includes fallback behavior when strict filtering is too aggressive.

## 4.5 LLM failure handling

If HF call fails (HTTP/network/DNS/token issues), code returns `_build_fallback_answer(...)` from retrieved contexts so users still get useful output.

This keeps user flow alive during external dependency outages.

## 5. Frontend Detailed Flow (`frontend/src/components/ChatInterface.jsx`)

## 5.1 Initialization

At load:

1. resolve `API_BASE_URL` from `import.meta.env.VITE_API_BASE_URL` fallback.
2. initialize default assistant welcome message.
3. call `/stats` once in `useEffect`.
4. store chunk count/state for the side panel.

## 5.2 Send message path

On form submit:

1. trim input and guard against empty/duplicate send while loading,
2. append user message locally,
3. send `POST /chat` JSON payload,
4. parse backend response,
5. append assistant message,
6. on fetch failure, append backend-unreachable message,
7. clear loading state.

## 5.3 Rendering behavior

- user and assistant messages use separate visual styles,
- loading animation shown during pending request,
- auto-scroll to latest message after updates.

## 5.4 Why this frontend flow is effective

- optimistic message append improves responsiveness,
- minimal state model keeps code maintainable,
- backend-driven answers avoid frontend business logic duplication.

## 6. Data Contracts Across Layers

## 6.1 Chat request/response

Request:

```json
{ "question": "..." }
```

Response:

```json
{ "answer": "...", "sources": ["..."] }
```

## 6.2 Status contracts

`/stats`:

```json
{ "total_chunks": 70, "status": "ready" }
```

`/scrape-status`:

```json
{ "running": false, "done": true, "message": "Done - indexed ..." }
```

These simple contracts make frontend integration straightforward and robust.

## 7. Where Inconsistencies Previously Came From (and How Flow Was Corrected)

Previously observed inconsistent outputs were caused by one of these flow gaps:

1. missing URL in index (for example `program-list/` before targeted scrape),
2. intent not detected (for example typo like `progrms`),
3. strict post-filter removing relevant context,
4. source suppression triggered by over-broad no-info marker.

Corrections were made in retrieval branching, intent triggers, source filtering, and targeted indexing.

## 8. Practical Debugging Sequence for This Codebase

When an answer is incomplete, the fastest reliable debug sequence is:

1. check `/stats` and `/scrape-status`,
2. verify source URL is in vector metadata,
3. run targeted `/scrape-urls` for missing page,
4. test intent route using representative query variants,
5. validate resulting `sources` list,
6. tune intent-specific pinning/filter logic if needed.

This sequence matches the architecture and avoids blind prompt tweaking.

## 9. Summary

The code flow is intentionally layered:

- backend policy router for deterministic constraints,
- retrieval layer for grounded knowledge access,
- generation layer for natural-language synthesis,
- fallback layer for resilience,
- frontend layer for simple, reliable interaction.

Most quality gains in this project came from improving retrieval flow and intent-aware context selection, not from changing model providers. That is expected for RAG systems where data coverage and context routing dominate answer quality.
