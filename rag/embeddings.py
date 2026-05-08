"""
app/rag/embeddings.py
Manages the FAISS vector store backed by OpenAI embeddings.
Supports create / load / save / merge operations.
"""

import logging
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingManager:
    """
    Thin wrapper around LangChain's FAISS vector store.

    Lifecycle
    ---------
    manager = EmbeddingManager()

    # Add documents (creates index first time, merges on subsequent calls)
    manager.add_documents(docs)

    # Persist to disk
    manager.save()

    # Similarity search
    results = manager.search("What is the refund policy?", k=4)
    """

    def __init__(self):
        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._store: FAISS | None = None
        self._index_path = Path(settings.faiss_index_path)
        self._index_path.mkdir(parents=True, exist_ok=True)

        # Try to load a persisted index on startup
        if self._has_saved_index():
            self._load()

    # ─── Public API ──────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> None:
        """Embed *documents* and merge into the FAISS index."""
        if not documents:
            logger.warning("add_documents called with empty list — skipping.")
            return

        logger.info("Embedding %d document chunks…", len(documents))
        if self._store is None:
            self._store = FAISS.from_documents(documents, self._embeddings)
            logger.info("Created new FAISS index.")
        else:
            new_store = FAISS.from_documents(documents, self._embeddings)
            self._store.merge_from(new_store)
            logger.info("Merged %d chunks into existing index.", len(documents))

        self.save()

    def search(
        self, query: str, k: int = 4, score_threshold: float = 0.0
    ) -> list[Document]:
        """
        Return the top-*k* most relevant document chunks for *query*.
        Optionally filter by minimum similarity score.
        """
        if self._store is None:
            logger.warning("No FAISS index loaded — cannot search.")
            return []

        try:
            results_with_scores = self._store.similarity_search_with_score(query, k=k)
            docs = [
                doc
                for doc, score in results_with_scores
                if score >= score_threshold
            ]
            logger.debug(
                "Search for '%s' returned %d/%d results above threshold %.2f",
                query[:60],
                len(docs),
                len(results_with_scores),
                score_threshold,
            )
            return docs
        except Exception as exc:
            logger.error("FAISS search failed: %s", exc)
            return []

    def save(self) -> None:
        """Persist the current FAISS index to disk."""
        if self._store is None:
            return
        self._store.save_local(str(self._index_path))
        logger.info("FAISS index saved to '%s'.", self._index_path)

    def clear(self) -> None:
        """Wipe the in-memory index (does NOT delete files)."""
        self._store = None
        logger.info("In-memory FAISS index cleared.")

    @property
    def is_ready(self) -> bool:
        return self._store is not None

    @property
    def document_count(self) -> int:
        if self._store is None:
            return 0
        return self._store.index.ntotal  # type: ignore[attr-defined]

    # ─── Private helpers ─────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            self._store = FAISS.load_local(
                str(self._index_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                "Loaded FAISS index from '%s' (%d vectors).",
                self._index_path,
                self.document_count,
            )
        except Exception as exc:
            logger.error("Could not load FAISS index: %s — starting fresh.", exc)
            self._store = None

    def _has_saved_index(self) -> bool:
        index_file = self._index_path / "index.faiss"
        return index_file.exists()
