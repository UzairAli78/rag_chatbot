"""
Vector Store Module
────────────────────
Manages the FAISS vector index:
  • Creation from documents
  • Incremental document addition
  • Disk persistence (load/save)
  • Full reset for re-indexing after file deletion
  • Retriever factory method
"""

import os
import logging
from typing import List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from src.embeddings import get_embeddings
from src.config import VECTOR_STORE_DIR, TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)

_INDEX_FAISS = "index.faiss"
_INDEX_PKL   = "index.pkl"


class VectorStoreManager:
    """
    Lifecycle manager for a FAISS vector store.

    Responsibilities:
      - Initialise embeddings once (via cached get_embeddings()).
      - Persist the index to disk after every mutation.
      - Provide a retriever that callers (RAGPipeline) can use.
    """

    def __init__(self) -> None:
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        self._embeddings = get_embeddings()
        self._store: Optional[FAISS] = None
        self._try_load_from_disk()

    # ── Disk I/O ─────────────────────────────────────────────────────────────

    def _try_load_from_disk(self) -> None:
        """Attempt to load a persisted FAISS index; silently skip if absent."""
        faiss_file = os.path.join(VECTOR_STORE_DIR, _INDEX_FAISS)
        if not os.path.exists(faiss_file):
            logger.info("No existing FAISS index found — starting fresh.")
            return

        try:
            self._store = FAISS.load_local(
                VECTOR_STORE_DIR,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("✅ FAISS index loaded from disk.")
        except Exception as e:
            logger.warning(f"Could not load existing FAISS index ({e}). Starting fresh.")
            self._store = None

    def _persist(self) -> None:
        """Save the current in-memory index to disk."""
        if self._store is not None:
            self._store.save_local(VECTOR_STORE_DIR)
            logger.info("FAISS index persisted to disk.")

    # ── Mutations ────────────────────────────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> None:
        """
        Embed and add documents to the store. Creates the store if it doesn't
        yet exist.

        Args:
            docs: Chunked LangChain Document objects.

        Raises:
            ValueError: If docs list is empty.
        """
        if not docs:
            raise ValueError("No documents provided to add_documents().")

        if self._store is None:
            logger.info(f"Creating new FAISS index with {len(docs)} chunks.")
            self._store = FAISS.from_documents(docs, self._embeddings)
        else:
            logger.info(f"Adding {len(docs)} chunks to existing FAISS index.")
            self._store.add_documents(docs)

        self._persist()

    def reset(self) -> None:
        """
        Completely wipe the in-memory store and delete persisted files.
        Call this before rebuilding the index after a document deletion.
        """
        self._store = None
        for fname in [_INDEX_FAISS, _INDEX_PKL]:
            fpath = os.path.join(VECTOR_STORE_DIR, fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError as e:
                    logger.warning(f"Could not delete {fpath}: {e}")
        logger.info("Vector store reset (memory + disk).")

    # ── Accessors ────────────────────────────────────────────────────────────

    def has_documents(self) -> bool:
        """Return True if the store has been populated."""
        return self._store is not None

    def get_store(self) -> Optional[FAISS]:
        """Return the raw FAISS store (or None if not initialised)."""
        return self._store

    def get_retriever(self, k: int = TOP_K_RETRIEVAL):
        """
        Return a LangChain retriever backed by this FAISS index.

        Args:
            k: Number of top documents to retrieve per query.

        Raises:
            RuntimeError: If no documents have been indexed yet.
        """
        if not self.has_documents():
            raise RuntimeError(
                "Vector store is empty. Please upload and index documents first."
            )
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
