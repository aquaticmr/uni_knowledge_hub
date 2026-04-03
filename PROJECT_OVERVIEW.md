# RBU AI Assistant - Separate Project Overview

## 1. What has been done

- Built a full RBU-focused RAG assistant with FastAPI backend and React frontend.
- Added scraping + indexing pipeline for RBU pages into local ChromaDB.
- Added mandatory policy routing in backend for fixed-response safety/behavior cases.
- Added deterministic responses for deans/directors/CDPC queries.
- Added resilient fallback in `backend/brain.py` for Hugging Face outages (DNS/network/HTTP/token issues):
  - No raw exception strings shown to user.
  - Retrieval-based answer is returned with sources.

## 2. Tech stack

- Backend: FastAPI, Uvicorn
- Retrieval DB: ChromaDB (persistent)
- Embeddings: Chroma default ONNX embedding function
- LLM: Hugging Face Router Chat Completions API
- Scraping/Parsing: requests, BeautifulSoup, lxml, markdownify
- OCR: pytesseract, pdf2image, Pillow
- Frontend: React + Vite + Tailwind

## 3. Important files and their roles

- `backend/main.py`
  - App lifecycle, auto-start scrape behavior, API endpoints, mandatory policy router.
- `backend/brain.py`
  - Chunking, indexing, retrieval, generation, deterministic people responses, outage fallback.
- `backend/scraper.py`
  - URL fetch, extraction, table parsing, optional OCR enrichment.
- `backend/chroma_noop.py`
  - Telemetry no-op helper for Chroma stability.
- `backend/test_policy_router.py`
  - Unit tests for mandatory policy-case detection.
- `frontend/src/components/ChatInterface.jsx`
  - Chat UI and backend API integration.
- `detail.md`
  - Consolidated project implementation reference.
- `workflow.md`
  - Deep architecture and end-to-end flow documentation.

## 4. End-to-end flow

1. Backend starts and checks vector DB chunk count.
2. If empty, startup URLs are scraped and indexed in a background thread.
3. User asks a question in frontend; request goes to `POST /chat`.
4. Backend first checks mandatory policy routing.
5. If no policy match, retrieval runs from ChromaDB.
6. Backend attempts Hugging Face generation.
7. If HF fails, retrieval-only fallback answer is returned with source URLs.

## 5. Required setup

## 5.1 Environment variables (`backend/.env`)

- `HF_TOKEN=<your_token>`
- `HF_MODEL=meta-llama/Llama-3.1-8B-Instruct` (or another supported model)
- `STARTUP_SCRAPE_URLS=<comma-separated URLs>` (optional; defaults exist)

## 5.2 Dependencies

- Install backend packages from `backend/requirements.txt`.
- Install frontend packages from `frontend/package.json`.

## 5.3 Optional system tools for OCR

- Tesseract OCR in PATH.
- Poppler in PATH.

## 6. Run

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

## 7. Validation endpoints

- `GET /health`
- `GET /stats`
- `GET /scrape-status`
- `POST /chat-policy-test`
