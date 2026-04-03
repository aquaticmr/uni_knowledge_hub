# RBU AI Assistant - Implementation Reference

## 1. What has been implemented

This repository now contains a complete RBU-focused RAG assistant with:

- FastAPI backend for chat, stats, scraping, and policy testing.
- ChromaDB persistent vector store (`backend/rbu_chroma_db/`).
- Web scraper with HTML parsing, table extraction, and OCR hooks.
- Policy router in `main.py` for mandatory fixed responses before RAG.
- Retrieval + generation pipeline in `brain.py`.
- Resilience fallback in `brain.py`: if Hugging Face router is unreachable (DNS/network/HTTP issues), the API returns a retrieval-based summary with sources instead of a raw exception string.
- React frontend chat interface.

## 2. Core tech stack

- Backend API: FastAPI + Uvicorn
- Retrieval store: ChromaDB
- Embeddings: Chroma default ONNX embedding function (`all-MiniLM` family)
- LLM endpoint: Hugging Face Router (`https://router.huggingface.co/v1/chat/completions`)
- Scraping/parsing: `requests`, `beautifulsoup4`, `lxml`, `markdownify`
- OCR helpers: `pdf2image`, `pytesseract`, `Pillow`
- Frontend: React + Vite + Tailwind

## 3. File map (what each file does)

- `backend/main.py`
: FastAPI app lifecycle, startup scrape, endpoints (`/chat`, `/stats`, `/health`, `/scrape-status`, `/rescrape`, `/scrape-urls`, `/chat-policy-test`), and mandatory policy routing.

- `backend/brain.py`
: Chunking, Chroma indexing, retrieval, prompt building, HF model call, source extraction, deterministic people responses, and LLM outage fallback summarization.

- `backend/scraper.py`
: URL scraping, clean text extraction, table extraction, OCR-assisted content extraction, and batch document return.

- `backend/chroma_noop.py`
: No-op telemetry shim to avoid telemetry-related runtime issues.

- `backend/test_policy_router.py`
: Unit tests for mandatory policy case detection logic.

- `backend/requirements.txt`
: Python dependencies for backend and scraper pipeline.

- `frontend/src/components/ChatInterface.jsx`
: Chat UI, message rendering, source display, and API calls.

- `frontend/src/App.jsx`
: App shell and layout integration.

- `workflow.md`
: End-to-end architecture and operational flow documentation.

## 4. Runtime flow (short)

1. Backend starts.
2. Lifespan checks vector DB chunk count.
3. If empty, startup URLs are scraped and indexed in background.
4. User sends question to `/chat`.
5. Policy router checks mandatory response cases.
6. If no policy hit, RAG retrieval runs from Chroma.
7. Backend tries Hugging Face generation.
8. If HF fails, backend returns retrieval-only fallback summary with source URLs.

## 5. Required setup

## 5.1 Python and Node

- Python 3.11+
- Node.js 18+

## 5.2 Backend dependencies

Install from:

- `backend/requirements.txt`

## 5.3 Environment variables

Create `backend/.env` with:

- `HF_TOKEN=<your_token>`
- `HF_MODEL=meta-llama/Llama-3.1-8B-Instruct` (or any router-supported model)
- `STARTUP_SCRAPE_URLS=<comma-separated URLs>` (optional; defaults exist in `main.py`)

## 5.4 Optional OCR system tools

- Tesseract OCR installed and available on PATH.
- Poppler installed and available on PATH.

Without these tools, OCR-heavy extraction paths may be limited.

## 6. Run commands

Backend:

```powershell
Set-Location "d:\Projects\Mohit_Rudrakar\backend"
d:\Projects\Mohit_Rudrakar\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
Set-Location "d:\Projects\Mohit_Rudrakar\frontend"
npm install
npm run dev
```

## 7. Health checks

- `GET /health` -> backend liveness
- `GET /stats` -> indexed chunk count
- `GET /scrape-status` -> current ingestion status
- `POST /chat-policy-test` -> verify mandatory route behavior

## 8. Current known behavior

- If RBU data is indexed, startup skip logic avoids re-scraping.
- If RBU data is empty, auto-scrape starts in background.
- If HF Router DNS/network fails, user sees an informative fallback answer from retrieved context with citations instead of raw `HTTPSConnectionPool` exception text.

