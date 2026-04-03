"""
main.py â€” FastAPI Backend for RBU AI Assistant
Provides /stats and /chat endpoints.
Runs the scraper on startup if needed (in a background thread).
"""

import os
import re
import sys
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper import (
    run_scraper_for_urls,
    extract_urls_from_text,
    CHROMA_DIR,
)
from brain import store_documents, get_stats, generate_response

load_dotenv()

DEFAULT_STARTUP_URLS = [
    "https://rbunagpur.in/overview",
    "https://rbunagpur.in/admission-process/",
    "https://rbunagpur.in/eligibility-criteria/",
    "https://rbunagpur.in/program-list-2026-2027/",
    "https://rbunagpur.in/program-list/",
    "https://rbunagpur.in/fees-structure-26-27/",
    "https://rbunagpur.in/fees-structure/",
    "https://rbunagpur.in/direct-second-year-admissions/",
    "https://rbunagpur.in/hostel-facilities/",
    "https://rbunagpur.in/accreditation-rankings/",
    "https://rbunagpur.in/interdisciplinary-studies/",
    "https://rbunagpur.in/scholarships-financial-aid/",
    "https://rbunagpur.in/higher-education/"

]


MANDATORY_RESPONSES = {
    "small_talk": "Hello! I am Niaa, the official RBU Information Assistant. I can help you with admissions, fees, placements, eligibility, and campus details.",
    "short_confirmation": "Sure. Please tell me what you want to know about RBU, for example admissions, fees, placements, hostel, scholarships, or eligibility.",
    "competitor_comparison": "Comparing colleges is personal. I can walk you through RBU's specific strengths, like our A+ NAAC grade and placement trends, so you can weigh them against your other options. Want me to pull those up?",
    "admission_probability": "I can't guarantee admission as cutoffs shift yearly based on applicants. However, I can show you the previous year's cutoff trends to give you a realistic sense of the competition. Shall we look at the 2024-25 table?",
    "eligibility_doubt": "Eligibility for unique cases like gap years is decided by the admissions committee. I can show you the general criteria, but for your specific situation, it's best to contact the RBU office directly. Want their number?",
    "academic_assistance": "I'm built specifically for RBU admissions and campus info, not for solving homework or writing assignments. However, I can explain the curriculum of any RBU course if you are curious!",
    "identity_origin": "I am Niaa, the official RBU Information Assistant. I was created specifically to help students with admissions, fees, and placement details at RBU.",
    "irrelevant_requests": "That's outside my scope! I can't help with that, but if you're coming to Nagpur, I can give you the campus address and hostel details. Interested?",
    "privacy_request": "For privacy reasons, I cannot share personal mobile numbers. For official queries, you can reach the RBU General Office at +91 9156288990 or use the departmental emails I have on file.",
    "financial_bargaining": "Fees are fixed according to the official RBU structure. However, I can help you find scholarship opportunities or financial aid that might apply to you. Should I look that up?",
    "future_speculation": "I only work with verified data from the current 2026-27 cycle. I can't predict future changes, but I can show you the most recent placement statistics and fee charts. Want to see those?",
    "direct_transactions": "I cannot process payments or registrations directly. However, I can provide you with the link to the official RBU Application Portal and a guide on how to apply. Shall I share the link?",
}


def _normalize_query(text: str) -> str:
    """Lowercase and normalize punctuation/whitespace for robust keyword matching."""
    normalized = (text or "").lower().strip()
    normalized = re.sub(r"[^a-z0-9+\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _contains_any_phrase(text: str, phrases: list[str]) -> bool:
    """Match full words/phrases instead of raw substring fragments."""
    for phrase in phrases:
        pattern = rf"(?:^|\s){re.escape(phrase)}(?:$|\s)"
        if re.search(pattern, text):
            return True
    return False


def _contains_verb_object_intent(text: str, verbs: list[str], objects: list[str]) -> bool:
    """Detect requests like 'write a 300-word essay' even when words are separated."""
    return _contains_any(text, verbs) and _contains_any(text, objects)


def _detect_mandatory_case(question: str) -> str | None:
    """Return mandatory case key for policy routing; otherwise None."""
    q = _normalize_query(question)

    if q in {"yes", "yeah", "yup", "ok", "okay", "hmm", "hmm ok"}:
        return "short_confirmation"

    small_talk_words = [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "who is there",
    ]
    if _contains_any_phrase(q, small_talk_words):
        return "small_talk"

    compare_words = ["compare", "comparison", "vs", "versus", "better than", "best college"]
    college_words = ["college", "university", "institute", "iit", "nit", "vit", "mit", "iiit", "bits"]
    if _contains_any(q, compare_words) and _contains_any(q, college_words):
        return "competitor_comparison"

    admission_prob_words = [
        "guarantee admission",
        "guaranteed admission",
        "will i get admission",
        "chance of admission",
        "chances of admission",
        "admission probability",
        "confirm admission",
        "sure admission",
        "pakka admission",
        "can i get in",
    ]
    if _contains_any(q, admission_prob_words):
        return "admission_probability"

    eligibility_words = ["eligible", "eligibility", "can i apply", "can i get admission"]
    edge_case_words = [
        "gap year",
        "drop year",
        "diploma",
        "lateral entry",
        "backlog",
        "backlogs",
        "year gap",
        "special case",
        "migration",
    ]
    if _contains_any(q, eligibility_words) and _contains_any(q, edge_case_words):
        return "eligibility_doubt"

    academic_help_words = [
        "homework",
        "assignment",
        "write essay",
        "essay for me",
        "solve this",
        "project report",
        "lab report",
        "exam answer",
        "complete my",
    ]
    academic_gen_verbs = ["write", "draft", "generate", "create", "make"]
    academic_gen_objects = ["essay", "assignment", "homework", "report", "project", "answer", "solution"]
    if _contains_any(q, academic_help_words) or _contains_verb_object_intent(q, academic_gen_verbs, academic_gen_objects):
        return "academic_assistance"

    identity_words = [
        "who are you",
        "what are you",
        "your name",
        "who made you",
        "are you chatgpt",
        "who created you",
        "what is niaa",
        "are you ai",
    ]
    if _contains_any(q, identity_words):
        return "identity_origin"

    irrelevant_words = [
        "weather",
        "recipe",
        "movie",
        "song",
        "joke",
        "news",
        "stock price",
        "cricket score",
        "horoscope",
        "translate this",
    ]
    rbu_context_words = ["rbu", "ramdeobaba", "admission", "fees", "placement", "hostel"]
    if _contains_any(q, irrelevant_words) and not _contains_any(q, rbu_context_words):
        return "irrelevant_requests"

    privacy_words = [
        "personal number",
        "mobile number",
        "phone number",
        "whatsapp number",
        "student number",
        "faculty number",
        "teacher number",
        "professor number",
        "private contact",
    ]
    if _contains_any(q, privacy_words):
        return "privacy_request"

    bargaining_words = [
        "discount",
        "reduce fee",
        "lower fee",
        "fee concession",
        "waive fee",
        "less fees",
        "negotiate fee",
        "can you reduce",
        "special discount",
    ]
    if _contains_any(q, bargaining_words):
        return "financial_bargaining"

    future_words = [
        "predict",
        "future",
        "in 2028",
        "in 2029",
        "in 2030",
        "next 5 years",
        "next year fees",
        "future fees",
        "future placement",
        "expected package in",
    ]
    if _contains_any(q, future_words):
        return "future_speculation"

    transaction_words = [
        "take payment",
        "pay fees here",
        "can i pay you",
        "register me",
        "do my registration",
        "complete my application",
        "book my seat",
        "admit me now",
        "payment through you",
    ]
    if _contains_any(q, transaction_words):
        return "direct_transactions"

    return None


def _route_mandatory_case(question: str) -> str | None:
    """Return fixed response for mandatory policy cases; otherwise None."""
    case_key = _detect_mandatory_case(question)
    return MANDATORY_RESPONSES.get(case_key) if case_key else None


def _get_startup_urls() -> list[str]:
    """Resolve startup scrape URLs from env or default list."""
    raw = os.getenv("STARTUP_SCRAPE_URLS", "").strip()
    if raw:
        urls = [item.strip() for item in raw.split(",") if item.strip()]
        return list(dict.fromkeys(urls))
    return DEFAULT_STARTUP_URLS

scraping_status = {"running": False, "done": False, "message": ""}


def _run_scraper_background(
    force: bool = False,
    urls: list[str] | None = None,
):
    """Run scraper and indexing in a background thread."""
    scraping_status["running"] = True
    scraping_status["message"] = "Scraping started..."

    def update_status(message: str) -> None:
        scraping_status["message"] = message

    try:
        documents = run_scraper_for_urls(urls=urls or [], progress_callback=update_status)

        if documents:
            scraping_status["message"] = f"Indexing {len(documents)} documents..."
            count = store_documents(documents)
            scraping_status["message"] = f"Done â€” indexed {count} chunks."
            print(f"[Server] Indexed {count} chunks into ChromaDB.")
        else:
            scraping_status["message"] = "No documents scraped."
            print("[Server] No documents scraped.")
    except Exception as e:
        scraping_status["message"] = f"Error: {str(e)}"
        print(f"[Server] Scraper error: {e}")
    finally:
        scraping_status["running"] = False
        scraping_status["done"] = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Auto-seed the DB from startup URLs when empty; skip when indexed."""
    stats = get_stats()
    has_chunks = stats.get("total_chunks", 0) > 0

    if has_chunks:
        print("[Server] Existing indexed knowledge base found. Skipping scrape.")
        scraping_status["done"] = True
        scraping_status["message"] = "Knowledge base loaded."
    else:
        startup_urls = _get_startup_urls()
        if startup_urls:
            print(f"[Server] Knowledge base empty. Auto-scraping {len(startup_urls)} startup URLs...")
            scraping_status["done"] = False
            scraping_status["message"] = f"Auto startup scrape for {len(startup_urls)} URLs started..."
            t = threading.Thread(
                target=_run_scraper_background,
                kwargs={"force": True, "urls": startup_urls},
                daemon=True,
            )
            t.start()
        else:
            print("[Server] Knowledge base is empty and no startup URLs configured.")
            scraping_status["done"] = True
            scraping_status["message"] = "Knowledge base is empty. Configure STARTUP_SCRAPE_URLS or call /scrape-urls."

    print(f"[Server] Knowledge base stats: {stats}")
    yield


app = FastAPI(
    title="RBU Nagpur AI Assistant",
    description="RAG-powered chatbot for RBU Nagpur university information",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []


class URLScrapeRequest(BaseModel):
    urls: list[str] | None = None
    url_text: str | None = None


class PolicyTestRequest(BaseModel):
    question: str


@app.get("/stats")
async def stats():
    """Return knowledge base statistics."""
    return get_stats()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a user question and return an AI-generated response."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    policy_answer = _route_mandatory_case(request.question)
    if policy_answer:
        return ChatResponse(answer=policy_answer, sources=[])

    result = generate_response(request.question)
    return ChatResponse(answer=result["answer"], sources=result["sources"])


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat-policy-test")
async def chat_policy_test(request: PolicyTestRequest):
    """Debug endpoint to validate mandatory policy routing without invoking RAG."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    case_key = _detect_mandatory_case(question)
    return {
        "matched": bool(case_key),
        "case": case_key,
        "answer": MANDATORY_RESPONSES.get(case_key, ""),
    }


@app.get("/scrape-status")
async def scrape_status():
    """Return current scraping progress."""
    return scraping_status


@app.post("/rescrape")
async def rescrape():
    """Force a background re-scrape using configured startup URLs."""
    if scraping_status["running"]:
        raise HTTPException(status_code=409, detail="Scrape already running.")

    startup_urls = _get_startup_urls()
    if not startup_urls:
        raise HTTPException(status_code=400, detail="No startup URLs configured.")

    scraping_status["done"] = False
    scraping_status["message"] = "Rescrape requested..."
    t = threading.Thread(
        target=_run_scraper_background,
        kwargs={"force": True, "urls": startup_urls},
        daemon=True,
    )
    t.start()
    return {
        "status": "started",
        "message": "Background rescrape started.",
        "url_count": len(startup_urls),
    }


@app.post("/scrape-urls")
async def scrape_urls(request: URLScrapeRequest):
    """Scrape only user-provided URLs and index them in background."""
    if scraping_status["running"]:
        raise HTTPException(status_code=409, detail="Scrape already running.")

    url_list = request.urls or []
    if request.url_text:
        url_list.extend(extract_urls_from_text(request.url_text))

    url_list = list(dict.fromkeys([u.strip() for u in url_list if isinstance(u, str) and u.strip()]))
    if not url_list:
        raise HTTPException(status_code=400, detail="No valid URLs provided.")

    scraping_status["done"] = False
    scraping_status["message"] = f"Targeted scrape requested for {len(url_list)} URLs..."
    t = threading.Thread(
        target=_run_scraper_background,
        kwargs={"force": True, "urls": url_list},
        daemon=True,
    )
    t.start()
    return {
        "status": "started",
        "message": "Background targeted scrape started.",
        "url_count": len(url_list),
    }

