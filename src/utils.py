"""
Utilities Module — backend-agnostic file helpers.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.config import UPLOAD_DIR, ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)


def get_uploaded_files() -> List[Dict]:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    files: List[Dict] = []
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        ext = Path(fname).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        files.append({
            "name": fname,
            "path": fpath,
            "size": os.path.getsize(fpath),
            "extension": ext,
        })
    return sorted(files, key=lambda x: x["name"].lower())


def delete_file(file_path: str) -> bool:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Delete failed '{file_path}': {e}")
        return False


def clear_upload_directory() -> None:
    if not os.path.exists(UPLOAD_DIR):
        return
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
            except Exception as e:
                logger.warning(f"Could not remove {fpath}: {e}")


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1_024:
        return f"{size_bytes} B"
    if size_bytes < 1_024 ** 2:
        return f"{size_bytes / 1_024:.1f} KB"
    if size_bytes < 1_024 ** 3:
        return f"{size_bytes / 1_024 ** 2:.1f} MB"
    return f"{size_bytes / 1_024 ** 3:.1f} GB"


def get_file_icon(filename: str) -> str:
    icons = {".pdf": "📕", ".docx": "📘", ".txt": "📄", ".md": "📝", ".csv": "📊"}
    return icons.get(Path(filename).suffix.lower(), "📎")


def sanitize_filename(filename: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9.\-_ ]", "_", filename)
    name = re.sub(r"[_ ]{2,}", "_", name).strip("_ ")
    return name[:255] if name else "unnamed_file"
