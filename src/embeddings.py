"""
Embeddings Module
─────────────────
Provides a cached HuggingFace Sentence Transformer embeddings model.
Uses functools.lru_cache to ensure the heavy model is loaded only once
per process, saving memory and startup time on repeated calls.
"""

import logging
from functools import lru_cache

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Load and cache the HuggingFace Sentence Transformer embeddings model.

    Returns:
        HuggingFaceEmbeddings: A LangChain-compatible embeddings object.

    Raises:
        ImportError: If required packages are not installed.
        Exception: If the model fails to load.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")

    # Try the newer langchain-huggingface package first, fall back to community
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,   # Cosine-similarity friendly
            "batch_size": 32,
        },
    )

    logger.info("✅ Embedding model loaded successfully.")
    return embeddings
