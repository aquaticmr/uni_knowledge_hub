"""
scraper.py — RBU Nagpur Website Scraper
Scrapes provided URLs, extracts clean text, tables, and OCR content.
"""

import os
import re
import io
import warnings
import requests
from typing import Callable
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urljoin

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

try:
    import pytesseract
except Exception:
    pytesseract = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "rbu_chroma_db")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Silence repeated BeautifulSoup decode warnings after we apply controlled decoding.
warnings.filterwarnings(
    "ignore",
    message="Some characters could not be decoded, and were replaced with REPLACEMENT CHARACTER.*"
)


def _decode_response(resp: requests.Response) -> str:
    """Decode HTTP response with best-effort encoding fallback."""
    encoding = resp.encoding or resp.apparent_encoding or "utf-8"
    return resp.content.decode(encoding, errors="replace")


def extract_tables_as_markdown(soup: BeautifulSoup) -> str:
    """Convert all HTML tables to Markdown format."""
    tables_md = []
    for table in soup.find_all("table"):
        try:
            table_md = md(str(table), strip=['img', 'a'])
            tables_md.append(table_md.strip())
        except Exception:
            continue
    return "\n\n".join(tables_md)


def clean_text(soup: BeautifulSoup) -> str:
    """Extract clean text from a page, removing nav, footer, scripts, etc."""
    # Remove unwanted elements
    for tag in soup.find_all(['nav', 'footer', 'script', 'style', 'noscript',
                               'header', 'aside']):
        tag.decompose()

    # Remove social media links and sharing buttons
    for el in soup.find_all(class_=re.compile(
            r'(social|share|footer|menu|nav|sidebar|widget|cookie)', re.I)):
        el.decompose()

    # Get the main content area
    main = soup.find('main') or soup.find('article') or soup.find(
        'div', class_=re.compile(r'(content|entry|post|page)', re.I))

    if main:
        text = main.get_text(separator='\n', strip=True)
    else:
        body = soup.find('body')
        text = body.get_text(separator='\n', strip=True) if body else ""

    # Clean up excessive whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)


def extract_pdf_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Collect PDF links from a page."""
    pdf_links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        full_url = urljoin(base_url, href)
        if full_url.lower().endswith(".pdf"):
            pdf_links.append(full_url)
    return list(dict.fromkeys(pdf_links))


def ocr_pdf_from_url(pdf_url: str, max_pages: int = 3) -> str:
    """Run OCR on a PDF URL and return extracted text."""
    if not convert_from_bytes or not pytesseract:
        return ""

    try:
        resp = requests.get(pdf_url, headers=HEADERS, timeout=40)
        resp.raise_for_status()
        images = convert_from_bytes(resp.content, first_page=1, last_page=max_pages)
        chunks = []
        for idx, image in enumerate(images, 1):
            text = pytesseract.image_to_string(image)
            if text.strip():
                chunks.append(f"[Page {idx}]\n{text.strip()}")
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"[Scraper] PDF OCR failed for {pdf_url}: {e}")
        return ""


def extract_image_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Collect image links from a page for OCR."""
    image_links = []
    for img in soup.find_all("img", src=True):
        src = img["src"].strip()
        full_url = urljoin(base_url, src)
        if re.search(r"\.(png|jpe?g|webp)$", full_url, re.I):
            image_links.append(full_url)
    return list(dict.fromkeys(image_links))


def ocr_image_from_url(image_url: str) -> str:
    """Run OCR on a remote image and return extracted text."""
    if not pytesseract:
        return ""

    try:
        from PIL import Image

        resp = requests.get(image_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"[Scraper] Image OCR failed for {image_url}: {e}")
        return ""


def scrape_page(url: str) -> dict | None:
    """Scrape a single page and return its content."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        html_text = _decode_response(resp)
        soup = BeautifulSoup(html_text, "html.parser")

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        # Extract tables as markdown
        tables_md = extract_tables_as_markdown(soup)

        # Extract clean text
        text = clean_text(soup)

        # Some pages are heavily template-driven; keep a fallback pass.
        if not text:
            fallback_lines = []
            for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li"]):
                line = node.get_text(" ", strip=True)
                if line and len(line) > 2:
                    fallback_lines.append(line)

            # De-duplicate while preserving order.
            seen = set()
            compact_lines = []
            for line in fallback_lines:
                key = line.lower()
                if key in seen:
                    continue
                seen.add(key)
                compact_lines.append(line)

            if compact_lines:
                text = "\n".join(compact_lines[:500])

        # OCR from linked PDFs
        pdf_ocr_texts = []
        for pdf_url in extract_pdf_links(soup, url):
            ocr_text = ocr_pdf_from_url(pdf_url)
            if ocr_text:
                pdf_ocr_texts.append(f"### OCR from PDF: {pdf_url}\n{ocr_text}")

        # OCR from page images
        image_ocr_texts = []
        for image_url in extract_image_links(soup, url)[:5]:
            ocr_text = ocr_image_from_url(image_url)
            if ocr_text:
                image_ocr_texts.append(f"### OCR from Image: {image_url}\n{ocr_text}")

        # Combine: tables are important, put them first
        combined = ""
        if tables_md:
            combined += f"## Tables from: {title}\n\n{tables_md}\n\n"
        if text:
            combined += f"## Content from: {title}\n\n{text}"
        if pdf_ocr_texts:
            combined += "\n\n## PDF OCR Data\n\n" + "\n\n".join(pdf_ocr_texts)
        if image_ocr_texts:
            combined += "\n\n## Image OCR Data\n\n" + "\n\n".join(image_ocr_texts)

        if not combined.strip():
            return None

        return {
            "url": url,
            "title": title,
            "content": combined.strip()
        }

    except Exception as e:
        print(f"[Scraper] Error scraping {url}: {e}")
        return None


def extract_urls_from_text(text: str) -> list[str]:
    """Extract http/https URLs from free-form pasted text."""
    if not text:
        return []
    matches = re.findall(r"https?://[^\s]+", text)
    cleaned = []
    for value in matches:
        # Drop accidental trailing punctuation from copy-paste.
        cleaned.append(value.rstrip(",;)]}"))
    return list(dict.fromkeys(cleaned))


def run_scraper_for_urls(urls: list[str], progress_callback: Callable[[str], None] | None = None) -> list[dict]:
    """Scrape only the provided URLs and return documents."""
    def emit(message: str) -> None:
        if progress_callback:
            try:
                progress_callback(message)
            except Exception:
                pass

    candidates = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        emit("No valid URLs provided.")
        return []

    emit(f"Scraping {len(candidates)} provided URLs...")
    documents = []
    for idx, url in enumerate(candidates, 1):
        emit(f"Scraping provided URL {idx}/{len(candidates)}...")
        doc = scrape_page(url)
        if doc:
            documents.append(doc)

    emit(f"Provided URL scrape complete. Collected {len(documents)} documents.")
    return documents


if __name__ == "__main__":
    docs = run_scraper_for_urls(["https://rbunagpur.in/overview"])
    for d in docs:
        print(f"\n{'='*60}")
        print(f"URL: {d['url']}")
        print(f"Title: {d['title']}")
        print(f"Content preview: {d['content'][:200]}...")
