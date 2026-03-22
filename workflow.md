# RBU AI Assistant: End-to-End Workflow and System Theory

## 1. Purpose and System Context
The project is a domain-focused RAG assistant for RBU Nagpur. Its objective is not generic conversation quality, but reliable, source-grounded answers for university queries such as admissions, eligibility, fees, placements, recruiters, and hostel facilities.

From an engineering perspective, this is a three-stage information system:
1. Ingestion stage: collect and normalize raw web content.
2. Knowledge indexing stage: convert normalized text into vector-searchable units.
3. Answering stage: retrieve relevant units and generate grounded natural-language responses.

The system is optimized for practical local deployment:
- Local persistent vector DB (`ChromaDB`) to avoid rebuilding context every run.
- Configurable startup scraping to seed knowledge base once, then reuse.
- Lightweight embedding path (Chroma default embedding function) to avoid heavy Torch setup.
- HF Router model invocation for generation.

---

## 2. Architecture at a Glance

## 2.1 Runtime Components
- Backend API and orchestration: `backend/main.py`
- Scraping and parsing: `backend/scraper.py`
- RAG core (chunking, embeddings, retrieval, generation): `backend/brain.py`
- Chroma telemetry patch: `backend/chroma_noop.py`
- Vector store persistence directory: `backend/rbu_chroma_db/`
- Frontend chat UX: `frontend/src/components/ChatInterface.jsx`

## 2.2 Data and Control Planes
- Control plane: API routes, lifecycle logic, background threads, status reporting.
- Data plane: scraped documents -> chunks -> embeddings -> vector search -> prompt -> model output.

## 2.3 Core Design Principle
The backend follows a one-time-seed + reuse model:
- If vector DB has chunks, skip scraping.
- If empty, scrape configured URLs and index.
This minimizes startup delay and external dependency calls during normal operation.

---

## 3. Detailed Start-to-End Flow

## 3.1 Startup Flow (Application Lifespan)
File: `backend/main.py`

1. FastAPI app boots via Uvicorn.
2. Lifespan hook executes:
   - Calls `get_stats()` from `brain.py`.
   - Computes `has_chunks = total_chunks > 0`.
3. Branching behavior:
   - If `has_chunks` is true:
     - Marks status as loaded.
     - Skips scraping/indexing.
   - If false:
     - Reads startup URL list from `.env` (`STARTUP_SCRAPE_URLS`) or default URL fallback.
     - Starts daemon background thread for scrape + indexing.
4. API becomes responsive immediately while background ingestion progresses.

Theoretical note:
- This is an asynchronous warmup pattern where read endpoints remain available even during index population. It favors low startup blocking at the cost of temporary partial readiness.

## 3.2 Scrape and Parse Flow
File: `backend/scraper.py`

Given a URL list, `run_scraper_for_urls` performs:
1. URL normalization and deduplication.
2. Per-URL `scrape_page` execution:
   - HTTP fetch with timeout and browser-like headers.
   - Robust decoding (`_decode_response`) to survive mixed encodings.
   - HTML parse with BeautifulSoup.
3. Content extraction layers:
   - Main textual content extraction (`clean_text`).
   - Tabular extraction (`extract_tables_as_markdown`) to preserve fee/eligibility structures.
   - Linked PDF OCR (`ocr_pdf_from_url`) where possible.
   - Linked image OCR (`ocr_image_from_url`) where possible.
4. Content assembly into a single document payload:
   - `title`, `url`, and combined `content` sections.
5. Progress callback emits phase updates to backend status endpoint.

Theoretical note:
- This parser uses structural-denoise heuristics (removing nav/footer/script patterns) instead of strict DOM templates. That improves portability across page templates but can leak some boilerplate content on highly scripted pages.

## 3.3 Index Construction Flow
File: `backend/brain.py`

`store_documents` transforms scraped payloads into searchable vector memory:
1. Chunking:
   - Window size 1000 chars, overlap 200 chars.
   - Overlap preserves cross-boundary semantics (e.g., heading in chunk A, values in chunk B).
2. Embedding:
   - Uses Chroma default embedding function (`all-MiniLM` ONNX path).
3. Persistence:
   - Upserts chunk records into collection `rbu_knowledge`.
   - Batch size 100 for write stability and memory control.

Why chunking + overlap matters:
- Pure large documents hurt retrieval precision due to mixed topics.
- Tiny chunks hurt context continuity.
- 1000/200 is a compromise between retrieval discrimination and semantic continuity.

Why `upsert` instead of `add`:
- Supports repeat ingestion safely.
- Prevents duplicate-ID crashes.
- Allows deterministic re-indexing for changed pages.

## 3.4 Query and Answer Flow
Files: `backend/main.py`, `backend/brain.py`, `frontend/src/components/ChatInterface.jsx`

1. User submits question in frontend.
2. Frontend POSTs `{question}` to `/chat`.
3. Backend validates input and calls `generate_response(question)`.
4. Retrieval stage:
   - Query embedding generated.
   - Top-K (K=4) chunks fetched by similarity.
5. Prompt construction:
   - System prompt enforces context-grounded assistant behavior.
   - Retrieved chunks injected as context text.
6. Generation stage:
   - Requests HF Router endpoint (`/v1/chat/completions`) with configured `HF_MODEL`.
7. Backend returns:
   - `answer` (model response)
   - `sources` (distinct URLs from retrieved chunks)
8. Frontend renders answer and source list.

Theoretical note:
- This is late-fusion RAG: retrieval and generation are coupled at query time without retraining. Knowledge updates are immediate after re-index, unlike fine-tuned models requiring retraining cycles.

---

## 4. Component-by-Component Deep Theory

## 4.1 `main.py`: Orchestration and System State

Primary role:
- Provides stable API contract, startup lifecycle policy, and background work scheduling.

Internal state model:
- `scraping_status` acts as a lightweight in-memory state machine:
  - Idle: `running=false, done=true`
  - Working: `running=true, done=false`
  - Completed: `running=false, done=true`
  - Error: `running=false, done=true` + error message

Concurrency model:
- Uses Python threads for non-blocking background ingest.
- Thread writes status updates; request handlers read status.
- For this scale, shared dict is acceptable; at larger scale, use locked structures or external state store.

Route semantics:
- `/stats`: knowledge readiness indicator.
- `/scrape-status`: operational progress indicator.
- `/rescrape`: deterministic refresh using configured startup URLs.
- `/scrape-urls`: ad hoc ingestion route for operator-provided targets.

Design strengths:
- Simple operational model.
- Good developer ergonomics.
- Startup auto-seeding behavior fits local workflows.

Known tradeoffs:
- In-memory status resets on process restart.
- Single-process assumptions (no distributed worker coordination).

## 4.2 `scraper.py`: Information Extraction Engine

Primary role:
- Convert unstructured web pages into model-friendly, semantically dense text.

Extraction strategy:
- Hybrid extraction:
  - Structural text extraction (DOM text)
  - Structural table extraction (markdown for schema-like data)
  - OCR for non-selectable text assets

Why table markdown is important:
- LLMs reason better with explicit row/column boundaries than flattened cell text.
- Fees and placement stats are often table-centric.

OCR branch theory:
- OCR expands recall for scanned documents but introduces noise and latency.
- OCR quality depends on source resolution and preprocessing; current implementation is minimal preprocessing, so extraction is pragmatic rather than perfect.

Current denoise heuristic:
- Drops typical non-content tags/classes.
- Still susceptible to dynamic widget/form residues in some WordPress pages.

Potential parser evolution:
- Readability-style content scoring.
- Boilerplate classifier.
- Domain-specific rules for RBU page templates.

## 4.3 `brain.py`: Retrieval and Generation Intelligence

Primary role:
- Build and query vector memory, then mediate LLM generation with context.

Subsystems:
1. Vector memory subsystem
   - Chroma persistent collection
   - cosine similarity space
2. Retrieval subsystem
   - embedding query
   - nearest-neighbor search
3. Prompting subsystem
   - role-constrained system instruction
   - context serialization
4. Generation subsystem
   - model invocation via HF Router
   - error-aware fallback messaging

Embedding model path:
- Chroma default embedding function avoids heavy dependency tree.
- Appropriate for low-to-medium semantic retrieval quality with lightweight setup.

Generation model path:
- Configurable by `HF_MODEL`.
- Router support differs by account/provider; model availability must be validated.

Error handling quality:
- HTTP errors from router converted into user-visible details.
- Improves operator diagnosis compared with opaque failure strings.

## 4.4 `chroma_noop.py`: Operational Stability Patch

Primary role:
- Override telemetry capture to avoid runtime telemetry integration breakage.

Why needed:
- Certain environment/package combinations caused telemetry capture signature issues.
- A no-op telemetry client stabilizes startup and runtime behavior without affecting retrieval functionality.

## 4.5 Frontend Chat UI (`ChatInterface.jsx`)

Primary role:
- Human-facing interaction layer and status surface.

State model:
- `messages`: chat transcript
- `loading`: async request state
- `stats`: chunk count and readiness state

Interaction flow:
1. User sends question.
2. Input disabled during request.
3. Response added to transcript with source references.
4. Auto-scroll ensures latest context visibility.

UX properties:
- Minimal but effective operational transparency (chunk count + source URLs).
- Error fallback distinguishes backend reachability issues.

Limitations:
- Stats fetched once (not polled).
- No streaming token output.
- Error display not yet structured by error type.

---

## 5. Operational Sequences

## 5.1 First-Time Setup Sequence
1. Configure `.env` (`HF_TOKEN`, `HF_MODEL`, URL list).
2. Start backend.
3. Observe auto startup scrape if DB empty.
4. Confirm `/stats` shows chunk count > 0.
5. Start frontend and test `/chat`.

## 5.2 Fresh Rebuild Sequence
1. Stop backend process.
2. Delete `backend/rbu_chroma_db/`.
3. Restart backend.
4. Auto-scrape + re-index runs from configured URLs.

## 5.3 Manual Targeted Ingestion Sequence
1. Call `/scrape-urls` with URL list.
2. Poll `/scrape-status`.
3. Validate with `/stats` and chat sources.

---

## 6. Configuration Theory and Best Practices

## 6.1 Environment Variables
- `HF_TOKEN`: authentication secret for router calls.
- `HF_MODEL`: runtime-selectable generation model.
- `STARTUP_SCRAPE_URLS`: deterministic ingestion scope at startup.

Best practices:
- Keep startup URL list domain-focused and high-signal.
- Use a model that your token/provider combination supports.
- Rotate token if unauthorized errors appear unexpectedly.

## 6.2 Model Selection Guidance
Selection criteria:
- Router availability for your token
- Instruction-following quality
- Response latency
- Cost constraints (if billing applies)

Practical current default:
- `meta-llama/Llama-3.1-8B-Instruct`

Fallback candidate:
- `Qwen/Qwen2.5-7B-Instruct`

---

## 7. Failure Modes and Component-Level Mitigation

## 7.1 `main.py` Failures
1. Symptom: `409 Scrape already running`
- Cause: concurrent trigger during active thread.
- Mitigation: poll `/scrape-status`, trigger only when `running=false`.

2. Symptom: startup appears idle with empty DB
- Cause: missing/empty startup URL config.
- Mitigation: set `STARTUP_SCRAPE_URLS` and restart.

3. Symptom: stale status after crash
- Cause: in-memory state not persisted.
- Mitigation: restart process, then verify `/stats` as source of truth.

## 7.2 `scraper.py` Failures
1. Symptom: content full of scripts/forms/widgets
- Cause: noisy HTML templates.
- Mitigation: expand pruning rules; apply readability extraction.

2. Symptom: OCR empty or skipped
- Cause: missing local binaries or unsupported assets.
- Mitigation: install/verify Tesseract and Poppler; validate PATH.

3. Symptom: scrape slow
- Cause: OCR-heavy pages and network variance.
- Mitigation: reduce URL set; cap OCR scope; run targeted ingestion only.

4. Symptom: timeout failures
- Cause: endpoint latency or blocking.
- Mitigation: retry with backoff; log failed URL list for replay.

## 7.3 `brain.py` Failures
1. Symptom: chunk count remains 0 after scrape
- Cause: no valid document payloads or indexing failure.
- Mitigation: inspect scraper outputs, then re-run indexing.

2. Symptom: unauthorized/model-not-supported errors
- Cause: token invalid or model unavailable.
- Mitigation: verify `HF_TOKEN`; switch `HF_MODEL` to supported option.

3. Symptom: irrelevant answers
- Cause: poor retrieval quality or noisy chunks.
- Mitigation: improve cleanup, adjust chunking, increase/decrease `n_results`, add metadata filters.

## 7.4 Frontend Failures
1. Symptom: cannot reach backend
- Cause: API base mismatch or backend down.
- Mitigation: set `VITE_API_BASE_URL`; verify backend health.

2. Symptom: users see raw backend error text
- Cause: backend returns error as answer string.
- Mitigation: standardize backend error schema and frontend rendering.

---

## 8. Performance and Scalability Analysis

## 8.1 Current Complexity Profile
- Scrape stage: O(U) network-bound, where U is number of URLs.
- Chunking stage: O(T), where T is total text size.
- Embedding stage: O(C), where C is chunk count.
- Retrieval per query: approximately O(log N) to O(N) depending on ANN index behavior and collection size.

## 8.2 Current Bottlenecks
- OCR latency per asset.
- Network fetch latency and page variability.
- Batch embedding throughput.
- Single-threaded ingestion loop in scraper.

## 8.3 Scaling Paths
Near-term:
- parallel URL fetch with bounded worker pool
- failed URL retry queue
- incremental indexing per page/batch

Mid-term:
- metadata-aware retrieval (`topic=fees/placement`)
- chunk deduplication and compression
- scheduled refresh pipeline

Long-term:
- separate worker service for ingestion
- durable task queue
- persistent status/event store and metrics dashboard

---

## 9. Security and Reliability Considerations

Token management:
- Store only in `.env`, not hardcoded in source.
- Avoid logging token values.

Input safety:
- URL ingestion endpoint should be domain-restricted in production.
- Add URL allowlist (`rbunagpur.in`) to prevent misuse.

Operational reliability:
- Add structured logs with correlation IDs.
- Add health checks for external dependencies (router reachability).

---

## 10. Improvement Blueprint (Detailed)

## 10.1 Ingestion Quality Improvements
- Add HTML boilerplate classifier.
- Remove known plugin/script signatures before text extraction.
- Parse fee/placement tables into typed schema before markdown serialization.

## 10.2 Retrieval Quality Improvements
- Tune chunk size by section type (tables vs prose).
- Hybrid retrieval (keyword + vector fusion).
- Add reranker for top-K retrieved chunks.

## 10.3 Answer Quality Improvements
- Add citation mapping from answer sentences to source chunks.
- Add hallucination guardrails (answer only from retrieved context).
- Add deterministic answer templates for fee/placement questions.

## 10.4 DevEx and Observability
- Add unit tests for parser and chunking.
- Add integration tests for `/chat` and `/scrape-urls`.
- Add scrape report endpoint with failed URL diagnostics.

---

## 11. Run and Validation Checklist

1. Backend start:
```powershell
Set-Location "d:\Projects\Mohit_Rudrakar\backend"
d:\Projects\Mohit_Rudrakar\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

2. Verify readiness:
```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/stats
curl.exe http://127.0.0.1:8000/scrape-status
```

3. Frontend start:
```powershell
Set-Location "d:\Projects\Mohit_Rudrakar\frontend"
npm install
npm run dev
```

4. Functional checks:
- `/stats.total_chunks > 0`
- chat returns answer and source URLs
- no auth/model errors from HF router

---

## 12. Quick Troubleshooting Matrix

- `401 Unauthorized` on chat
: invalid/misloaded token; update `HF_TOKEN`, restart backend.

- `model_not_supported`
: set supported `HF_MODEL` (for current token/provider profile).

- `Scrape already running` (409)
: wait until `/scrape-status.running` is false.

- `total_chunks = 0` after scrape
: inspect scrape outputs; ensure valid docs extracted; rerun scrape.

- Cannot delete `rbu_chroma_db`
: stop backend process first; then delete folder.

---

## 13. Summary
This project implements a production-style but lightweight RAG pipeline specialized for RBU content. The architecture separates ingestion, indexing, and answering concerns clearly, enabling easy iteration. The current system is already operational and resilient for local use, and its next quality leap will come from stronger HTML denoising, retrieval tuning, and incremental indexing/observability enhancements.
