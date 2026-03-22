
### **Project Title: RBU-Nagpur AI Admission & Placement Assistant**

**Goal:** Create a RAG (Retrieval-Augmented Generation) web application that scrapes https://rbunagpur.in/, processes university-specific data (fees, admissions, placements), and provides a chatbot interface.

---

#### **1. Technical Architecture & Models**
*   **Backend:** FastAPI (Python).
*   **Vector Database:** `ChromaDB` (Persistent local storage).
*   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Local).
*   **LLM:** `mistralai/Mistral-7B-Instruct-v0.2` via **Hugging Face Inference API**.
*   **OCR & Parsing:** `BeautifulSoup`, `pdf2image`, and `pytesseract` (to handle scanned placement graphs and PDF fee tables).
*   **Frontend:** React + Tailwind CSS.

---

#### **2. Detailed Implementation Instructions**

**Phase 1: Specialized Scraper (`scraper.py`)**
1.  **URL Discovery:** Use `https://rbunagpur.in/page-sitemap.xml` as the primary source of URLs.
2.  **Strict Content Filtering:** Only scrape pages that contain these keywords in the URL or Title: `['admission', 'eligibility', 'cutoff', 'fees', 'hostel', 'programs', 'placement', 'recruiters', 'cdpc']`.
3.  **Data Extraction Logic:**
    *   **HTML Tables:** Identify `<table>` tags and convert them into **Markdown format** (essential for the LLM to understand Fee structures).
    *   **Scanned Docs & PDFs:** If a page contains a PDF or image (e.g., placement statistics graphs), use `pdf2image` and `pytesseract` to extract text.
    *   **Clean Text:** Remove navigation menus, social media links, and footer junk.
4.  **One-Time Execution:** Check if the directory `./rbu_chroma_db` exists. If yes, skip scraping. If no, scrape and store.

**Phase 2: Vector Store & RAG Logic (`brain.py`)**
1.  Chunk the text into 1000 characters with 200-character overlap.
2.  Use **`all-MiniLM-L6-v2`** to create embeddings and store them in a local ChromaDB collection.
3.  Implement a retrieval function that takes a user query, finds the top 4 chunks, and constructs a prompt for the Hugging Face API.
4.  **System Prompt:** "You are the RBU Nagpur Assistant. Use the following context to answer the question. If the data is in a table, provide a structured summary. Context: {context}".

**Phase 3: FastAPI Backend (`main.py`)**
1.  **Endpoint `GET /stats`**: Returns the count of unique text chunks/documents in ChromaDB.
2.  **Endpoint `POST /chat`**: Receives the user question and returns the generated LLM response.
3.  Setup CORS to allow communication with the React frontend.

**Phase 4: Frontend Development (`App.jsx`)**
1.  **Sidebar (Left):** A box titled "RBU Knowledge Hub" showing the total number of "Scanned Chunks" (fetched from `/stats`).
2.  **Chat Interface (Right):** 
    *   A scrollable area for chat history.
    *   Distinct message bubbles for "User" and "AI Assistant".
    *   A loading spinner when waiting for a response.
    *   Branding: Use RBU's Navy Blue and White theme.

---

#### **3. Expected Output**
Please provide the code for:
1.  `requirements.txt`
2.  `scraper.py` (Handling sitemap + OCR + Table-to-Markdown).
3.  `main.py` (FastAPI + RAG logic + Hugging Face Integration).
4.  `ChatInterface.js` (The React component).

---

### **Preparation Checklist for You (The Developer)**

Before you run the code that Copilot generates, make sure you have these system-level tools installed:

1.  **Tesseract OCR Engine:** 
    *   Windows: [Download here](https://github.com/UB-Mannheim/tesseract/wiki).
    *   Mac: `brew install tesseract`.
    *   *Crucial:* Note the path where it's installed (usually `C:\Program Files\Tesseract-OCR\tesseract.exe`).
2.  **Poppler (for PDFs):**
    *   Windows: [Download here](https://github.com/oschwartz10612/poppler-windows/releases/), unzip, and add the `bin` folder to your System Environment Path.
    *   Mac: `brew install poppler`.
3.  **Hugging Face API Token:**
    *   Create a free account at [huggingface.co](https://huggingface.co/).
    *   Go to **Settings > Access Tokens** and create a "Read" token. You will paste this into the code where it says `HF_TOKEN`.

