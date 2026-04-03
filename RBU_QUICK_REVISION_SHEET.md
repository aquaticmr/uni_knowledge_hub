# RBU Assistant Quick Revision Sheet

## 1. Project in One Line

RBU Assistant is a FastAPI + React RAG chatbot that scrapes official RBU pages, stores embeddings in ChromaDB, retrieves relevant context, and generates source-grounded answers via Hugging Face LLM with fallback mode.

## 2. Core Stack

- Backend: FastAPI (`backend/main.py`)
- RAG Engine: `backend/brain.py`
- Scraper: `backend/scraper.py`
- Vector DB: ChromaDB (local persistent)
- Frontend: React + Vite (`frontend/src/components/ChatInterface.jsx`)
- LLM API: Hugging Face Router Chat Completions

## 3. End-to-End Flow

1. App starts -> checks vector DB chunk count.
2. If empty -> background scraping/indexing runs.
3. User asks question in frontend.
4. Backend policy router checks mandatory cases.
5. If no policy match -> retrieval from Chroma.
6. Intent-aware pinning adds key contexts (fees/programs/hostel).
7. Prompt built with retrieved context.
8. LLM generates answer.
9. If LLM fails -> retrieval-only fallback answer.
10. Frontend shows answer + source URLs.

## 4. Why This Architecture

- Scraping: keeps data current from official website.
- Vector retrieval: handles wording variation better than keyword-only search.
- RAG over fine-tuning: faster updates, no retraining cycle.
- Chroma local persistence: simple setup, fast local iteration.
- Fallback mode: graceful behavior during LLM outages.

## 5. Important APIs

- `GET /health` -> liveness
- `GET /stats` -> chunk count/status
- `POST /chat` -> main Q&A
- `GET /scrape-status` -> scraper state
- `POST /rescrape` -> full refresh
- `POST /scrape-urls` -> targeted ingestion
- `POST /chat-policy-test` -> policy debug

## 6. Current Key Retrieval Specializations

- Fees queries:
  - Boosts `fees-structure-26-27` and `fees-structure`
  - Uses higher retrieval depth and token budget
- Program-list queries:
  - Handles typo variants like `progrms`
  - Boosts `program-list` and `program-list-2026-2027`
- Hostel fee queries:
  - Prioritizes `hostel-facilities` context
  - Avoids wrong restriction to academic fee pages
- Overview queries:
  - Handles broad asks like `tell me about rbu/college`

## 7. Mandatory Policy Router Cases

- short confirmation
- small talk
- competitor comparison
- admission probability
- eligibility doubt
- academic assistance
- identity origin
- irrelevant requests
- privacy request
- financial bargaining
- future speculation
- direct transactions

Tests live in `backend/test_policy_router.py`.

## 8. Source Constraints Applied

These source URL groups are excluded from retrieval/sources:

- `/deans/`
- `/directors/`
- `/team-cdpc/`

## 9. High-Value Commands

Backend run:

```powershell
Set-Location d:\Projects\Mohit_Rudrakar\backend
d:\Projects\Mohit_Rudrakar\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8080
```

Frontend run:

```powershell
Set-Location d:\Projects\Mohit_Rudrakar\frontend
npx vite --host 127.0.0.1 --port 5173
```

Targeted scrape:

```powershell
$body = @{ urls = @('https://rbunagpur.in/program-list/') } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8080/scrape-urls' -ContentType 'application/json' -Body $body
```

## 10. Common Failure Patterns

- Incomplete answer:
  - URL not indexed -> run targeted scrape
  - intent not detected -> update intent keywords/typo handling
  - strict filtering too aggressive -> add intent-specific fallback
- 404 on frontend localhost:
  - Vite started with wrong args
  - run `npx vite --host 127.0.0.1 --port 5173`
- Uvicorn start failure:
  - port already in use
  - stop old listeners and restart

## 11. Viva Quick Pitch

"We built a domain-specific RAG assistant for RBU with a retrieval-first architecture. We scrape official pages, index chunks in local ChromaDB, retrieve relevant context per user query, and generate grounded responses using HF LLM. We added intent-aware retrieval tuning for fees, programs, and hostel flows, plus robust fallback handling when LLM API is down. This gives better accuracy, transparency, and operational resilience than a generic chatbot setup."
