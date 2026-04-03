"""
brain.py â€” RAG Logic with ChromaDB and HuggingFace
Handles text chunking, embedding, storage, retrieval,
and LLM response generation.
"""

import os
import re
from difflib import SequenceMatcher
import requests

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("CHROMA_PRODUCT_TELEMETRY_IMPL", "chroma_noop.NoOpTelemetry")

import chromadb
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from chromadb.config import Settings

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "rbu_chroma_db")
COLLECTION_NAME = "rbu_knowledge"
HF_TOKEN = os.getenv("HF_TOKEN", "")
LLM_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embedder = embedding_functions.DefaultEmbeddingFunction()

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)
print("[Brain] Chroma telemetry disabled.")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "i", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "was", "were", "what", "when", "where", "who", "why", "how", "with", "about",
    "please", "tell", "me", "details", "detail", "give", "everything", "evrthing", "all"
}

EXCLUDED_SOURCE_URL_PARTS = (
    "/deans/",
    "/directors/",
    "/team-cdpc/",
)


def _is_excluded_source(url: str) -> bool:
    value = (url or "").lower()
    return any(part in value for part in EXCLUDED_SOURCE_URL_PARTS)


def _is_fees_source(url: str) -> bool:
    value = (url or "").lower()
    return "/fees-structure" in value


def _is_programs_query(question: str) -> bool:
    q = (question or "").lower()
    has_program_word = any(
        token in q
        for token in [
            "program",
            "programme",
            "progrms",
            "programe",
            "course",
            "courses",
            "branch",
            "branches",
        ]
    )
    has_offer_intent = any(token in q for token in ["offered", "offer", "available", "list", "all", "what are"])
    has_entity = any(token in q for token in ["rbu", "ramdeobaba", "college", "university"])
    return has_program_word and (has_offer_intent or has_entity)


def _is_programs_source(url: str) -> bool:
    value = (url or "").lower()
    return "/program-list" in value


def _sentences(text: str) -> list[str]:
    """Split text into simple sentence-like fragments."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _build_fallback_answer(question: str, contexts: list[dict], reason: str | None = None) -> str:
    """Build a retrieval-only answer when the remote LLM is unavailable."""
    terms = _query_terms(question)
    selected: list[str] = []

    for ctx in contexts:
        title = (ctx.get("title") or "RBU source").strip()
        text = ctx.get("text") or ""
        sentences = _sentences(text)

        matching = []
        for sentence in sentences:
            lower = sentence.lower()
            if not terms or any(term in lower for term in terms):
                matching.append(sentence)

        candidates = matching[:2] if matching else sentences[:2]
        if not candidates:
            continue

        snippet = " ".join(candidates).strip()
        if snippet:
            selected.append(f"{title}: {snippet}")

        if len(selected) >= 3:
            break

    if not selected:
        return "I could not generate a detailed answer right now, but relevant RBU sources were found. Please try again in a moment."

    preface = "I am having trouble reaching the language model right now, but here is what I found from official RBU data:"
    if reason:
        preface = f"I am having trouble reaching the language model right now ({reason}), but here is what I found from official RBU data:"

    lines = [f"{idx}. {item}" for idx, item in enumerate(selected, 1)]
    return preface + "\n" + "\n".join(lines)


def _to_plain_text(answer: str) -> str:
    """Convert common markdown emphasis to plain text for cleaner UI output."""
    if not answer:
        return ""
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", answer, flags=re.DOTALL)
    cleaned = cleaned.replace("**", "")
    return cleaned.strip()


def _query_terms(text: str) -> set[str]:
    """Extract meaningful terms from user query for relevance checks."""
    words = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}


def _fuzzy_token_overlap(query_terms: set[str], context_text: str) -> bool:
    """Return True if query terms approximately match context terms."""
    if not query_terms:
        return True

    context_terms = set(re.findall(r"[a-zA-Z0-9]+", (context_text or "").lower()))
    context_terms = {w for w in context_terms if len(w) >= 3}
    if not context_terms:
        return False

    if query_terms.intersection(context_terms):
        return True

    for q in query_terms:
        for c in context_terms:
            if abs(len(q) - len(c)) > 3:
                continue
            if SequenceMatcher(None, q, c).ratio() >= 0.78:
                return True

    return False


def _is_relevant_context(question: str, context_text: str, distance: float | None) -> bool:
    """Hybrid relevance check using lexical/fuzzy overlap and embedding distance."""
    terms = _query_terms(question)
    if not terms:
        return True

    if _fuzzy_token_overlap(terms, context_text):
        return True

    if distance is not None and distance <= 0.65:
        return True

    return False


def _is_no_info_answer(answer: str) -> bool:
    """Detect fallback/no-info responses and suppress unrelated source citations."""
    text = (answer or "").lower()
    markers = [
        "could not find",
        "no relevant information",
        "i don't know",
        "not available in the context",
        "don't have any information",
        "do not have any information",
        "please provide more details",
    ]
    return any(marker in text for marker in markers)


def _is_rbu_overview_query(question: str) -> bool:
    """Detect broad overview asks like 'tell me about RBU/college'."""
    q = (question or "").lower()
    has_rbu_entity = any(token in q for token in ["rbu", "ramdeobaba", "nagpur university", "college", "university"])
    has_overview_intent = any(
        token in q
        for token in [
            "tell me about",
            "about rbu",
            "about college",
            "about university",
            "tell me everything",
            "evrthing",
            "everything",
            "all about",
            "overview",
            "full details",
            "complete details",
        ]
    )
    return has_rbu_entity and has_overview_intent


def _is_fees_query(question: str) -> bool:
    q = (question or "").lower()
    return any(token in q for token in ["fee", "fees", "fee structure", "tuition", "caution money", "development fee"])


def _is_hostel_query(question: str) -> bool:
    q = (question or "").lower()
    return any(token in q for token in ["hostel", "accommodation", "mess", "room", "boarding"])


def _is_hostel_fees_query(question: str) -> bool:
    return _is_hostel_query(question) and _is_fees_query(question)


def _expand_query_for_retrieval(question: str) -> str:
    """Add intent-specific terms to improve retrieval for known RBU asks."""
    q = (question or "").lower()
    expanded = [question]

    if "scholarship" in q or "financial aid" in q:
        expanded.append("rbu scholarships financial aid")

    if _is_fees_query(question):
        expanded.append("rbu full fees structure 2026 27 tuition development caution total all programs schools")

    if _is_hostel_query(question):
        expanded.append("rbu hostel facilities accommodation mess room charges fee structure")

    if _is_programs_query(question):
        expanded.append("rbu complete program list all schools departments ug pg phd programs offered")

    if _is_rbu_overview_query(question):
        expanded.append("rbu overview admissions programs fees placements hostel scholarships accreditation")

    return " ".join(expanded).strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def store_documents(documents: list[dict]) -> int:
    """
    Chunk and store documents in ChromaDB.
    Each document dict has: url, title, content
    Returns total number of chunks stored.
    """
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for doc in documents:
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['url']}__chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "url": doc["url"],
                "title": doc["title"],
                "chunk_index": i
            })

    if all_chunks:
        print(f"[Brain] Embedding {len(all_chunks)} chunks...")
        embeddings = embedder(all_chunks)

        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            end = min(i + batch_size, len(all_chunks))
            collection.upsert(
                ids=all_ids[i:end],
                documents=all_chunks[i:end],
                embeddings=embeddings[i:end],
                metadatas=all_metadatas[i:end]
            )
        print(f"[Brain] Stored {len(all_chunks)} chunks in ChromaDB.")

    return len(all_chunks)


def get_stats() -> dict:
    """Return stats about the knowledge base."""
    try:
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        count = collection.count()
        return {"total_chunks": count, "status": "ready" if count > 0 else "empty"}
    except Exception as e:
        return {"total_chunks": 0, "status": f"error: {str(e)}"}


def retrieve_context(query: str, n_results: int = 4) -> list[dict]:
    """Retrieve the top-N most relevant chunks for a query."""
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        return []

    query_embedding = embedder([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    contexts = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            contexts.append({
                "text": doc,
                "url": meta.get("url", ""),
                "title": meta.get("title", ""),
                "distance": (
                    results["distances"][0][i]
                    if results.get("distances") and results["distances"][0]
                    else None
                )
            })

    if _is_fees_query(query):
        for fee_url in [
            "https://rbunagpur.in/fees-structure-26-27/",
            "https://rbunagpur.in/fees-structure/",
        ]:
            try:
                pinned = collection.get(where={"url": fee_url}, include=["documents", "metadatas"])
                docs = pinned.get("documents") or []
                metas = pinned.get("metadatas") or []
                for i, doc in enumerate(docs[:6]):
                    meta = metas[i] if i < len(metas) else {}
                    contexts.append({
                        "text": doc,
                        "url": meta.get("url", fee_url),
                        "title": meta.get("title", ""),
                        "distance": None,
                    })
            except Exception:
                continue

    if _is_hostel_query(query):
        for h_url in ["https://rbunagpur.in/hostel-facilities/"]:
            try:
                pinned = collection.get(where={"url": h_url}, include=["documents", "metadatas"])
                docs = pinned.get("documents") or []
                metas = pinned.get("metadatas") or []
                for i, doc in enumerate(docs[:6]):
                    meta = metas[i] if i < len(metas) else {}
                    contexts.append({
                        "text": doc,
                        "url": meta.get("url", h_url),
                        "title": meta.get("title", ""),
                        "distance": None,
                    })
            except Exception:
                continue

    if _is_programs_query(query):
        for p_url in [
            "https://rbunagpur.in/program-list-2026-2027/",
            "https://rbunagpur.in/program-list/",
        ]:
            try:
                pinned = collection.get(where={"url": p_url}, include=["documents", "metadatas"])
                docs = pinned.get("documents") or []
                metas = pinned.get("metadatas") or []
                for i, doc in enumerate(docs[:8]):
                    meta = metas[i] if i < len(metas) else {}
                    contexts.append({
                        "text": doc,
                        "url": meta.get("url", p_url),
                        "title": meta.get("title", ""),
                        "distance": None,
                    })
            except Exception:
                continue

    if contexts:
        seen = set()
        deduped = []
        for c in contexts:
            key = (c.get("url"), c.get("text"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        contexts = deduped

    contexts = [c for c in contexts if not _is_excluded_source(c.get("url", ""))]

    return contexts


def generate_response(question: str) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant context
    2. Build prompt
    3. Call HuggingFace LLM
    4. Return response
    """
    top_k = 10 if (_is_fees_query(question) or _is_programs_query(question) or _is_hostel_query(question)) else 4
    contexts = retrieve_context(question, n_results=top_k)
    if not contexts:
        expanded_query = _expand_query_for_retrieval(question)
        if expanded_query != question:
            contexts = retrieve_context(expanded_query, n_results=top_k)

    relevant_contexts = [
        c
        for c in contexts
        if _is_relevant_context(
            question,
            f"{c.get('title', '')}\n{c.get('text', '')}",
            c.get("distance"),
        )
    ]

    if not relevant_contexts:
        expanded_query = _expand_query_for_retrieval(question)
        if expanded_query != question:
            expanded_contexts = retrieve_context(expanded_query, n_results=top_k)
            relevant_contexts = [
                c
                for c in expanded_contexts
                if _is_relevant_context(
                    question,
                    f"{c.get('title', '')}\n{c.get('text', '')}",
                    c.get("distance"),
                )
            ]

    if not relevant_contexts and _is_rbu_overview_query(question) and contexts:
        relevant_contexts = contexts

    if not relevant_contexts and contexts and len(_query_terms(question)) <= 2:
        relevant_contexts = contexts

    if _is_fees_query(question) and not _is_hostel_fees_query(question):
        fee_only = [c for c in relevant_contexts if _is_fees_source(c.get("url", ""))]
        if fee_only:
            relevant_contexts = fee_only

    if _is_programs_query(question):
        program_only = [c for c in relevant_contexts if _is_programs_source(c.get("url", ""))]
        if program_only:
            relevant_contexts = program_only

    if _is_hostel_query(question):
        hostel_only = [c for c in relevant_contexts if "hostel-facilities" in (c.get("url", ""))]
        if hostel_only:
            relevant_contexts = hostel_only

    if not relevant_contexts:
        return {
            "answer": "I could not find relevant information for that question in the current university data.",
            "sources": []
        }

    source_urls = list(
        set([
            c["url"]
            for c in relevant_contexts
            if c["url"] and not _is_excluded_source(c["url"])
            and (not _is_fees_query(question) or _is_hostel_fees_query(question) or _is_fees_source(c["url"]))
            and (not _is_programs_query(question) or _is_programs_source(c["url"]))
        ])
    )

    if not HF_TOKEN:
        return {
            "answer": _build_fallback_answer(question, relevant_contexts, reason="HF token missing"),
            "sources": source_urls,
        }

    context_text = "\n\n---\n\n".join([
        f"Source: {c['title']}\n{c['text']}" for c in relevant_contexts
    ])

    system_prompt = (
        "You are the RBU Nagpur Assistant. You help students and parents with "
        "information about RBU Nagpur. "
        "Use the following context to answer the question accurately. "
        "If the data is in a table, provide a structured summary. "
        "Return plain text only; do not use markdown syntax like **bold**. "
        "If you don't know the answer from the context, say so honestly.\n\n"
        f"Context:\n{context_text}"
    )

    if _is_fees_query(question):
        system_prompt += (
            "\n\nFor fee-structure questions, provide complete coverage of all programs visible in context. "
            "Use a clean structured list with Tuition, Development, Caution, and Total wherever available."
        )

    if _is_programs_query(question):
        system_prompt += (
            "\n\nFor program-list questions, provide a comprehensive list of programs from all schools visible in context. "
            "Group by school/department when possible and include UG, PG, and PhD entries available in context."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    response = None
    answer = ""
    try:
        response = requests.post(
            HF_ROUTER_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "max_tokens": 1500 if (_is_fees_query(question) or _is_programs_query(question)) else 1024,
                "temperature": 0.7,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        answer = _to_plain_text(data["choices"][0]["message"]["content"])
    except requests.HTTPError:
        detail = "LLM HTTP error"
        try:
            detail = response.json().get("error", {}).get("message", "") or detail
        except Exception:
            detail = response.text[:300] if response is not None else detail
        answer = _build_fallback_answer(question, relevant_contexts, reason=detail)
    except Exception as e:
        answer = _build_fallback_answer(question, relevant_contexts, reason=str(e))

    if _is_no_info_answer(answer):
        sources = []
    else:
        sources = source_urls

    return {
        "answer": answer,
        "sources": sources
    }
