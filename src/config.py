"""
Configuration module — loads all settings from environment variables.
All secrets are read from a .env file; no hardcoded values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# ─── Groq / LLM ──────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ─── Embeddings ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── Document Chunking ───────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# ─── Retrieval ───────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# ─── Memory ──────────────────────────────────────────────────────────────────
MEMORY_WINDOW: int = int(os.getenv("MEMORY_WINDOW", "10"))

# ─── File Handling ───────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent.parent
UPLOAD_DIR: str = str(BASE_DIR / "uploads")
VECTOR_STORE_DIR: str = str(BASE_DIR / "vector_store")

ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt", ".md", ".csv"}
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# ─── App ─────────────────────────────────────────────────────────────────────
APP_TITLE: str = "DocChat AI"
APP_ICON: str = "🤖"
APP_VERSION: str = "1.0.0"