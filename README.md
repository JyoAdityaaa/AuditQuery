# Audit Query — Audit/Tax Document Simplifier RAG Agent

Audit Query is a production-grade Retrieval-Augmented Generation (RAG) agent that helps financial analysts, tax consultants, and businesses simplify dense regulatory texts (like ICAI standards, GST circulars, and annual reports). It takes natural English queries, retrieves relevant clauses, and generates simplified, plain-English explanations backed by page-level citations.

---

## 🛠️ Manual Requirements & Prerequisites

Before launching Audit Query, you must manually satisfy the following prerequisites:

1. **Install Dependencies:**
   Install the required Python packages from the project root directory:
   ```bash
   pip install -r requirements.txt
   ```

2. **Obtain a Gemini API Key:**
   - Go to [Google AI Studio](https://aistudio.google.com/) and create a free API Key.
   - You can provide this key in three ways:
     - **In the `.env` file:** Open the newly created [.env](file:///c:/Users/mjyot/OneDrive/Desktop/Agent%20Reach/.env) file in the project folder and paste your key: `GEMINI_API_KEY=your_actual_key_here`.
     - **As an environment variable:**
       - **Windows (PowerShell):** `$env:GEMINI_API_KEY="your_api_key_here"`
       - **Linux/macOS:** `export GEMINI_API_KEY="your_api_key_here"`
     - **Directly in the Web UI:** Paste it in the **Gemini API Key** text field in the app sidebar.

---

## 🚀 How to Run

Start the FastAPI server:
```bash
python main.py
```
This will spin up a local development server. Automatically open the Audit Query dashboard in your web browser by navigating to `http://127.0.0.1:8000`.

---

## 🏗️ Architecture & Pipeline Details

### 1. Document Extraction & Chunking
- **Extraction:** Powered by `pypdf` which extracts text page-by-page. Pages containing no extractable text are flagged to prevent scanned/image-only PDFs from causing silent failures.
- **Chunking:** Pages are split using a custom character-based sliding window of **800 characters with 150 characters of overlap**. The chunker searches backwards for word boundaries (spaces, newlines) at the end of each slice to ensure that key financial figures, section numbers, or legal clauses are not split in half.

### 2. Local Embeddings & Unified GenAI SDK
- We use the **current, non-deprecated unified Google GenAI Python SDK (`google-genai`)** rather than the deprecated `google-generativeai` package for final generation.
- To prevent exhausting free-tier Google API limits and maximize processing speed, document text chunks are embedded **locally on your CPU** using the lightweight HuggingFace `SentenceTransformer` model: **`all-MiniLM-L6-v2`**.

### 3. Dual-Layer Vector Storage (ChromaDB + Fallback)
- **Primary Database:** ChromaDB is configured to run locally as a persistent store (`./vector_store_chroma`) using `cosine` distance.
- **Resilient Fallback:** Because ChromaDB utilizes compiled C++ extensions that can fail to build on newer Python versions (such as Python 3.13) in environments lacking build tools, Audit Query includes a **drop-in, pure-Python fallback (`SimpleVectorStore`)** built using NumPy. It uses NumPy to compute cosine similarity, provides the exact same API, and serializes state to disk (`./vector_store_simple`). The app switches seamlessly with zero developer friction.

### 4. Grounded Generation
- Queries are answered using **`gemini-2.5-flash`**.
- The system instructions strictly confine the model to the retrieved text context: if the information is not in the document, it states that it is not supported rather than hallucinating.
- The model cites inline claims using standard tags like `[GST_Circular_120.pdf, Page 3]`.
- All external API calls are wrapped in robust **exponential backoff retry logic** to handle the Gemini free-tier rate limits gracefully.

### 5. Incremental Index Syncing
- Every uploaded document is hashed using SHA-256.
- A metadata registry (`vector_store_meta.json`) tracks which file hashes are indexed.
- Clicking **Index & Sync Documents** performs a delta-comparison: new files are indexed, unchanged files are kept, and deleted files are pruned from the vector database automatically without re-processing everything.
