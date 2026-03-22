"""
brain.py — RAG Logic with ChromaDB and HuggingFace
Handles text chunking, embedding, storage, retrieval,
and LLM response generation.
"""

import os
import re
from difflib import SequenceMatcher
import requests
from bs4 import BeautifulSoup

# Disable Chroma telemetry early (must be set before importing chromadb).
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

# Use Chroma's ONNX MiniLM embedding function.
# This avoids pulling full PyTorch from sentence-transformers.
embedder = embedding_functions.DefaultEmbeddingFunction()

# Initialize ChromaDB client (persistent) with telemetry disabled
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

    # Exact overlap first.
    if query_terms.intersection(context_terms):
        return True

    # Fuzzy overlap to tolerate typos like "acadmic" or "scholership".
    for q in query_terms:
        for c in context_terms:
            # Skip very different token lengths to reduce false matches.
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

    # Distance fallback: allow semantically close matches even with typos.
    # Chroma cosine distance tends to be smaller for better matches.
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
        "from the provided context",
        "please provide more details",
    ]
    return any(marker in text for marker in markers)


def _is_rbu_overview_query(question: str) -> bool:
    """Detect broad 'tell me everything about RBU' style questions."""
    q = (question or "").lower()
    has_rbu_entity = any(token in q for token in ["rbu", "ramdeobaba", "nagpur university"])
    has_overview_intent = any(
        token in q
        for token in [
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


def _is_deans_query(question: str) -> bool:
    q = (question or "").lower()
    return "dean" in q


def _is_directors_query(question: str) -> bool:
    q = (question or "").lower()
    return "director" in q


def _is_cdpc_query(question: str) -> bool:
    q = (question or "").lower()
    return "cdpc" in q or ("team" in q and "placement" in q)


def _extract_people_names_from_url(url: str) -> list[str]:
    """Fetch a page and extract person names prefixed with Dr."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        text = "\n".join(
            node.get_text(" ", strip=True)
            for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li"])
        )
    except Exception:
        return []

    heading_matches = []
    for node in soup.find_all(["h3", "h4", "h5"]):
        line = node.get_text(" ", strip=True)
        if "dr" in line.lower():
            heading_matches.append(line)

    raw_matches = heading_matches + re.findall(r"\bDr\.?\s*[A-Z][A-Za-z.\s]{2,60}", text)
    cleaned = []
    for item in raw_matches:
        name = re.sub(r"\s+", " ", item).strip()
        # Remove role/title tails if they are glued to the name segment.
        name = re.split(
            r"\b(Dean|Director|School|Career|Development|Placement|Research|Admissions|Quality|Students)\b",
            name,
            maxsplit=1,
        )[0].strip()
        # Normalize common missing-space variant, e.g. 'Dr.Sanjay'.
        name = re.sub(r"^Dr\.", "Dr. ", name)
        name = re.sub(r"^Dr\s+", "Dr. ", name)
        name = re.sub(r"\s{2,}", " ", name)
        if len(name) >= 6:
            cleaned.append(name)

    # Keep order while removing duplicates.
    seen = set()
    unique = []
    for name in cleaned:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(name)
    return unique


def _deterministic_people_response(question: str) -> dict | None:
    """Return deterministic answers for governance/team people-list asks."""
    if _is_deans_query(question):
        url = "https://rbunagpur.in/deans/"
        names = _extract_people_names_from_url(url)
        if names:
            lines = [f"{idx}. {name}" for idx, name in enumerate(names, 1)]
            return {
                "answer": "As per the official RBU Deans page, the dean names are:\n" + "\n".join(lines),
                "sources": [url],
            }

    if _is_directors_query(question):
        url = "https://rbunagpur.in/directors/"
        names = _extract_people_names_from_url(url)
        if names:
            lines = [f"{idx}. {name}" for idx, name in enumerate(names, 1)]
            return {
                "answer": "As per the official RBU Directors page, the director names are:\n" + "\n".join(lines),
                "sources": [url],
            }

    if _is_cdpc_query(question):
        url = "https://rbunagpur.in/team-cdpc/"
        names = _extract_people_names_from_url(url)
        if names:
            lead = names[0]
            return {
                "answer": (
                    f"From the official Team CDPC page, a listed key contact is {lead}. "
                    "I can also summarize placement initiatives and recruiter highlights if you want."
                ),
                "sources": [url],
            }

    return None


def _expand_query_for_retrieval(question: str) -> str:
    """Add intent-specific terms to improve retrieval for known RBU asks."""
    q = (question or "").lower()
    expanded = [question]

    if any(token in q for token in ["dean", "deans"]):
        expanded.append("rbu deans directors academic leadership")

    if any(token in q for token in ["director", "directors"]):
        expanded.append("rbu directors leadership administration")

    if "cdpc" in q or ("team" in q and "placement" in q):
        expanded.append("rbu team cdpc placement cell")

    if "scholarship" in q or "financial aid" in q:
        expanded.append("rbu scholarships financial aid")

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
        # Embed all chunks with Chroma's built-in embedding model.
        print(f"[Brain] Embedding {len(all_chunks)} chunks...")
        embeddings = embedder(all_chunks)

        # Store in batches (ChromaDB has limits)
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

    # Embed the query
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

    # Boost known governance/team pages for intent-specific queries.
    q = (query or "").lower()
    target_urls = []
    if any(token in q for token in ["dean", "deans"]):
        target_urls.append("https://rbunagpur.in/deans/")
    if any(token in q for token in ["director", "directors"]):
        target_urls.append("https://rbunagpur.in/directors/")
    if "cdpc" in q or ("team" in q and "placement" in q):
        target_urls.append("https://rbunagpur.in/team-cdpc/")

    boosted_contexts = []
    for url in target_urls:
        try:
            pinned = collection.get(where={"url": url}, include=["documents", "metadatas"])
            docs = pinned.get("documents") or []
            metas = pinned.get("metadatas") or []
            for i, doc in enumerate(docs[:2]):  # First chunks usually carry headings/names.
                meta = metas[i] if i < len(metas) else {}
                boosted_contexts.append({
                    "text": doc,
                    "url": meta.get("url", url),
                    "title": meta.get("title", ""),
                    "distance": None,
                })
        except Exception:
            continue

    if boosted_contexts:
        # Keep boosted contexts first, then append non-duplicate semantic results.
        seen = {(c.get("url"), c.get("text")) for c in boosted_contexts}
        deduped = boosted_contexts[:]
        for c in contexts:
            key = (c.get("url"), c.get("text"))
            if key not in seen:
                deduped.append(c)
                seen.add(key)
        contexts = deduped[: max(n_results, len(boosted_contexts))]

    return contexts


def generate_response(question: str) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant context
    2. Build prompt
    3. Call HuggingFace LLM
    4. Return response
    """
    deterministic = _deterministic_people_response(question)
    if deterministic:
        return deterministic

    if not HF_TOKEN:
        return {
            "answer": "Error: HuggingFace API token not configured. Please set HF_TOKEN in your .env file.",
            "sources": []
        }

    # Step 1: Retrieve context
    contexts = retrieve_context(question)
    if not contexts:
        expanded_query = _expand_query_for_retrieval(question)
        if expanded_query != question:
            contexts = retrieve_context(expanded_query)

    # Filter out weak/irrelevant retrievals to avoid unrelated sources.
    relevant_contexts = [
        c
        for c in contexts
        if _is_relevant_context(
            question,
            f"{c.get('title', '')}\n{c.get('text', '')}",
            c.get("distance"),
        )
    ]

    # If strict relevance removed everything, retry once with expanded query.
    if not relevant_contexts:
        expanded_query = _expand_query_for_retrieval(question)
        if expanded_query != question:
            expanded_contexts = retrieve_context(expanded_query)
            relevant_contexts = [
                c
                for c in expanded_contexts
                if _is_relevant_context(
                    question,
                    f"{c.get('title', '')}\n{c.get('text', '')}",
                    c.get("distance"),
                )
            ]

    # For broad overview asks, fall back to top retrievals even if lexical filter is strict.
    if not relevant_contexts and _is_rbu_overview_query(question) and contexts:
        relevant_contexts = contexts

    if not relevant_contexts:
        return {
            "answer": "I could not find relevant information for that question in the current university data.",
            "sources": []
        }

    context_text = "\n\n---\n\n".join([
        f"Source: {c['title']}\n{c['text']}" for c in relevant_contexts
    ])

    # Step 2: Build the prompt
    system_prompt = (
        "You are the RBU Nagpur Assistant. You help students and parents with "
        "information about RBU Nagpur. "
        "Use the following context to answer the question accurately. "
        "If the data is in a table, provide a structured summary. "
        "Return plain text only; do not use markdown syntax like **bold**. "
        "If you don't know the answer from the context, say so honestly.\n\n"
        f"Context:\n{context_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    # Step 3: Call the LLM
    response = None
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
                "max_tokens": 1024,
                "temperature": 0.7,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        answer = _to_plain_text(data["choices"][0]["message"]["content"])
    except requests.HTTPError:
        detail = ""
        try:
            detail = response.json().get("error", {}).get("message", "")
        except Exception:
            detail = response.text[:300] if response is not None else ""
        answer = f"Error generating response: HTTP {response.status_code}. {detail}".strip()
    except Exception as e:
        answer = f"Error generating response: {str(e)}"

    # Step 4: Return with sources
    if _is_no_info_answer(answer):
        sources = []
    else:
        sources = list(set([c["url"] for c in relevant_contexts if c["url"]]))

    return {
        "answer": answer,
        "sources": sources
    }
