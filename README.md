# 🤖 rag-chatbot

**DocChat AI** — an intelligent document Q&A chatbot powered by a full RAG (Retrieval-Augmented Generation) pipeline. Upload any document and ask questions about it in natural language. The chatbot also handles casual conversation without touching the vector store, thanks to a built-in two-stage intent classifier. Built with FastAPI, FAISS, LangChain, and Groq (Llama 3.3).

---

## 📁 Project Structure

```
rag-chatbot/
├── backend.py                  # FastAPI backend — REST API & route handlers
├── index.html                  # DocChat AI frontend (dark-themed, single-page)
├── .env                        # API keys and model settings (not committed)
├── .gitignore                  # Excludes .env, venv, __pycache__, etc.
├── requirements.txt            # Python dependencies
├── how_to_start.txt            # Quick-start command reference
│
├── src/
│   ├── config.py               # All settings loaded from environment variables
│   ├── document_processor.py   # Document loading, parsing, and chunking
│   ├── embeddings.py           # Cached HuggingFace sentence-transformer loader
│   ├── vector_store.py         # FAISS index lifecycle (create, add, persist, reset)
│   ├── rag_pipeline.py         # Two-stage pipeline (intent classify → answer)
│   ├── prompts.py              # LangChain prompt templates and role descriptions
│   └── utils.py                # File helpers (list, delete, sanitize, format)
│
├── uploads/                    # Uploaded documents (auto-created)
└── vector_store/               # Persisted FAISS index (auto-created)
```

---

## ✨ Features

- **Upload any document** — supports PDF, DOCX, TXT, Markdown, and CSV
- **Ask questions in natural language** — answers are grounded strictly in your uploaded documents
- **Two-stage intent classifier** — distinguishes general chat from document questions; casual messages are answered directly without touching the vector store
- **Persistent FAISS index** — vector store survives server restarts; no need to re-upload documents
- **Incremental ingestion** — upload additional documents without clearing the existing index
- **Document deletion with auto-reindex** — remove a file and the index is rebuilt from remaining documents automatically
- **Configurable AI roles** — choose from Assistant, Teacher, Expert, Analyst, or Summarizer to shape the response style
- **Custom instructions** — pass freeform instructions to the LLM alongside every answer
- **Conversation memory** — retains the last 10 turns of context per session
- **Dark-themed UI** — purple/blue accented single-page app (no framework dependencies)

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-org/rag-chatbot.git
cd rag-chatbot

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# LLM (optional overrides)
GROQ_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048

# Embeddings (optional override)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chunking (optional overrides)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval (optional overrides)
TOP_K_RETRIEVAL=5
MEMORY_WINDOW=10
MAX_FILE_SIZE_MB=50
```

Get a free Groq API key at [console.groq.com](https://console.groq.com). The recommended model is `llama-3.3-70b-versatile`; a faster alternative is `llama-3.1-8b-instant`.

---

### 3. Start the Server

```bash
# Windows
venv\Scripts\python.exe -m uvicorn backend:app --reload --port 8000

# macOS / Linux
uvicorn backend:app --reload --port 8000
```

Open **http://localhost:8000** in your browser. The DocChat AI interface will be ready.

---

### 4. Upload a Document & Start Chatting

1. Click **Upload** in the UI and select a `.pdf`, `.docx`, `.txt`, `.md`, or `.csv` file (max 50 MB)
2. Wait for the indexing confirmation
3. Type any question about the document in the chat box
4. For casual conversation ("hi", "what can you do?") — just type naturally; the intent classifier handles it without retrieval

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the chat UI (`index.html`) |
| `GET` | `/api/health` | Server status, pipeline state, doc count |
| `GET` | `/api/documents` | List all uploaded documents |
| `POST` | `/api/upload` | Upload and index one or more documents |
| `DELETE` | `/api/documents/{filename}` | Delete a document and rebuild the index |
| `GET` | `/api/document/{filename}` | Preview document content |
| `POST` | `/api/chat` | Send a question and receive an answer |
| `POST` | `/api/new-chat` | Clear conversation memory |
| `POST` | `/api/reset` | Delete all documents and wipe the vector store |

### `POST /api/chat`

**Request:**
```json
{
  "question": "What is the refund policy mentioned in the document?",
  "role": "Assistant",
  "instruction": ""
}
```

**Response:**
```json
{
  "answer": "According to the document, the refund policy states...",
  "sources": ["policy.pdf"],
  "num_retrieved": 5
}
```

**Available roles:** `Assistant`, `Teacher`, `Expert`, `Analyst`, `Summarizer`

---

## ⚙️ Configuration (`src/config.py`)

| Setting | Default | Description |
|---|---|---|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq LLM model |
| `LLM_TEMPERATURE` | `0.1` | Low = factual and grounded |
| `LLM_MAX_TOKENS` | `2048` | Max tokens per LLM response |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence transformer |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Character overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Chunks retrieved per query |
| `MEMORY_WINDOW` | `10` | Conversation turns kept in memory |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload file size |
| `UPLOAD_DIR` | `./uploads` | Uploaded document storage path |
| `VECTOR_STORE_DIR` | `./vector_store` | FAISS index persistence path |

All settings are overridable via `.env` — no code changes needed.

---

## 🏗️ How It Works

### Document Ingestion (`document_processor.py` + `vector_store.py`)

1. **Load** — file extension is detected and routed to the correct LangChain loader: `PyPDFLoader` (PDF), `Docx2txtLoader` / python-docx fallback (DOCX), `TextLoader` (TXT/CSV), `UnstructuredMarkdownLoader` / TextLoader fallback (MD)
2. **Chunk** — `RecursiveCharacterTextSplitter` splits documents into 1000-character chunks with 200-character overlap, using natural separators (`\n\n`, `\n`, `. `)
3. **Embed** — chunks are encoded with `sentence-transformers/all-MiniLM-L6-v2` (loaded once and cached via `lru_cache`)
4. **Store** — embeddings are added to a FAISS index and persisted to `./vector_store/`; duplicate uploads are handled gracefully by checking existing IDs

### RAG Pipeline (`rag_pipeline.py`)

**Stage 1 — Intent Classification**
Every incoming message is sent to the LLM with a tight classifier prompt that returns exactly `GENERAL` or `DOCUMENT`. This prevents the RAG chain from being invoked for casual conversation like greetings or small talk.

**Stage 2 — Answer Generation**

- `GENERAL` → direct LLM call with a friendly conversational system prompt; no vector store touched
- `DOCUMENT` → full `ConversationalRetrievalChain`:
  1. Follow-up questions are condensed into standalone queries (using `CONDENSE_QUESTION_PROMPT`)
  2. Top-K chunks are retrieved from FAISS by cosine similarity
  3. Retrieved context is injected into the strict QA prompt, which instructs the LLM to answer only from provided context
  4. If the answer isn't in the documents, the model responds with a fixed apology phrase
  5. The turn is stored in a rolling `ConversationBufferWindowMemory` (last 10 turns)

### Prompt Engineering (`prompts.py`)

- **Role descriptions** — each role (`Teacher`, `Expert`, `Analyst`, etc.) injects a different persona description into the system prompt
- **QA prompt** — strictly forbids outside knowledge; requires source citation; supports optional freeform `instruction` override
- **Condense prompt** — rewrites follow-up questions as standalone queries for accurate vector retrieval

---

## ⚠️ Notes & Limitations

- **`GROQ_API_KEY` is required** — the server starts without it, but all `/api/chat` calls will fail
- **FAISS index persists across restarts** — the `./vector_store/` directory is loaded on startup automatically; delete it manually to start fresh
- **Deleting a document triggers a full reindex** — all remaining files are re-chunked and re-embedded; this may take a few seconds for large document sets
- **CPU-only embeddings** — `all-MiniLM-L6-v2` runs on CPU by default; this is fast enough for typical document sizes but will slow down on very large batches
- **No authentication** — CORS is open to all origins (`*`); add authentication middleware before deploying publicly

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | REST API server |
| `groq` + `langchain-groq` | Groq LLM client (Llama 3.3 / 3.1) |
| `faiss-cpu` | Local vector similarity search |
| `sentence-transformers` | Text embeddings (`all-MiniLM-L6-v2`) |
| `langchain` + `langchain-community` | RAG chain, memory, retrievers, loaders |
| `pypdf` | PDF text extraction |
| `python-docx` + `docx2txt` | DOCX text extraction |
| `unstructured[md]` | Markdown document loading |
| `tiktoken` | Token counting for chunking |
| `python-dotenv` | `.env` file loading |
