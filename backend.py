"""
DocChat AI — FastAPI Backend
════════════════════════════
Serves:
  • index.html at GET /
  • REST API at /api/...
  • Document file previews at /api/document/{name}

Run with:
  C:\rag_chatbot\venv\Scripts\python.exe -m uvicorn backend:app --reload --port 8000
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from src.config import UPLOAD_DIR, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS
from src.document_processor import DocumentProcessor, extract_docx_text
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from src.utils import (
    get_uploaded_files,
    delete_file as delete_file_util,
    clear_upload_directory,
    sanitize_filename,
    format_file_size,
    get_file_icon,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="DocChat AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ──────────────────────────────────────────────────────────────
_processor: DocumentProcessor = DocumentProcessor()
_vs_manager: Optional[VectorStoreManager] = None
_pipeline: Optional[RAGPipeline] = None
_chat_role: str = "Assistant"
_chat_instruction: str = ""

os.makedirs(UPLOAD_DIR, exist_ok=True)


def _get_vsm() -> VectorStoreManager:
    global _vs_manager
    if _vs_manager is None:
        _vs_manager = VectorStoreManager()
    return _vs_manager


def _build_pipeline(role: Optional[str] = None, instruction: Optional[str] = None) -> Optional[RAGPipeline]:
    global _pipeline, _chat_role, _chat_instruction
    if role is not None:
        _chat_role = role
    if instruction is not None:
        _chat_instruction = instruction
    vsm = _get_vsm()
    if not vsm.has_documents():
        _pipeline = None
        return None
    try:
        _pipeline = RAGPipeline(
            vector_store=vsm.get_store(),
            role=_chat_role,
            instruction=_chat_instruction,
        )
        logger.info("RAG pipeline built successfully.")
        return _pipeline
    except Exception as e:
        logger.error(f"Pipeline build failed: {e}")
        _pipeline = None
        return None


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    try:
        vsm = _get_vsm()
        if vsm.has_documents():
            _build_pipeline()
            logger.info("Pipeline restored from existing index.")
    except Exception as e:
        logger.warning(f"Startup init warning: {e}")


# ── Pydantic Models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    role: str = "Assistant"
    instruction: str = ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    vsm = _get_vsm()
    files = get_uploaded_files()
    return {
        "status": "ok",
        "pipeline_ready": _pipeline is not None,
        "has_documents": vsm.has_documents(),
        "doc_count": len(files),
        "role": _chat_role,
        "instruction": _chat_instruction,
    }


@app.get("/api/documents")
async def list_documents():
    files = get_uploaded_files()
    return {
        "documents": [
            {
                "name": f["name"],
                "size": f["size"],
                "size_display": format_file_size(f["size"]),
                "extension": f["extension"],
                "icon": get_file_icon(f["name"]),
            }
            for f in files
        ]
    }


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    global _pipeline
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    saved_paths: List[str] = []
    errors: List[str] = []

    for uf in files:
        ext = Path(uf.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"'{uf.filename}': unsupported type '{ext}'")
            continue

        data = await uf.read()
        size_mb = len(data) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            errors.append(f"'{uf.filename}': too large ({size_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB)")
            continue

        safe_name = sanitize_filename(uf.filename or "file")
        file_path = os.path.join(UPLOAD_DIR, safe_name)
        try:
            with open(file_path, "wb") as fh:
                fh.write(data)
            saved_paths.append(file_path)
        except Exception as e:
            errors.append(f"'{uf.filename}': save error — {e}")

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No files saved. " + "; ".join(errors))

    try:
        chunks = _processor.process_multiple(saved_paths)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text extracted from uploaded files.")
        vsm = _get_vsm()
        vsm.add_documents(chunks)
        _build_pipeline()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {e}")

    return {
        "success": True,
        "saved": len(saved_paths),
        "chunks": len(chunks),
        "errors": errors,
        "message": f"Indexed {len(saved_paths)} file(s) — {len(chunks)} chunks created.",
    }


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    global _pipeline
    safe_name = sanitize_filename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    if not delete_file_util(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    # Rebuild index from remaining files
    vsm = _get_vsm()
    vsm.reset()
    remaining = get_uploaded_files()
    if remaining:
        try:
            chunks = _processor.process_multiple([f["path"] for f in remaining])
            if chunks:
                vsm.add_documents(chunks)
                _build_pipeline()
            else:
                _pipeline = None
        except Exception as e:
            logger.error(f"Reindex failed: {e}")
            _pipeline = None
    else:
        _pipeline = None

    return {"success": True, "message": f"Deleted '{filename}' and rebuilt index."}


@app.get("/api/document/{filename}")
async def preview_document(filename: str):
    safe_name = sanitize_filename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return FileResponse(file_path, media_type="application/pdf",
                            headers={"Content-Disposition": "inline"})
    elif ext in (".txt", ".md", ".csv"):
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        return {"type": "text", "content": content}
    elif ext == ".docx":
        content = extract_docx_text(file_path)
        return {"type": "text", "content": content or "(No text extracted)"}
    else:
        return {"type": "text", "content": "(Preview not available for this file type)"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    global _pipeline, _chat_role, _chat_instruction
    if _pipeline is None:
        vsm = _get_vsm()
        if vsm.has_documents():
            _build_pipeline(req.role, req.instruction)
        if _pipeline is None:
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. Upload documents first."
            )

    # Update settings if changed
    if req.role != _chat_role or req.instruction != _chat_instruction:
        _pipeline.update_settings(req.role, req.instruction)
        _chat_role = req.role
        _chat_instruction = req.instruction

    try:
        result = _pipeline.query(req.question)
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "num_retrieved": result.get("num_retrieved", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {e}")


@app.post("/api/new-chat")
async def new_chat():
    if _pipeline:
        _pipeline.reset_memory()
    return {"success": True}


@app.post("/api/reset")
async def reset_all():
    global _pipeline, _vs_manager
    _pipeline = None
    vsm = _get_vsm()
    vsm.reset()
    _vs_manager = None
    clear_upload_directory()
    return {"success": True}
