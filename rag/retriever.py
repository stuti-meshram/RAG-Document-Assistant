"""
app/rag/retriever.py
Wraps EmbeddingManager with retrieval-specific logic:
  - query rewriting
  - source deduplication
  - context formatting
"""

import logging

from langchain_core.documents import Document

from app.rag.embeddings import EmbeddingManager
from app.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentRetriever:
    """
    High-level retriever that sits between the user query and the FAISS index.

    Parameters
    ----------
    embedding_manager : EmbeddingManager
        The shared embedding / vector-store instance.
    top_k : int
        Number of chunks to retrieve per query (default 4).
    """

    def __init__(self, embedding_manager: EmbeddingManager, top_k: int = 4):
        self._em = embedding_manager
        self.top_k = top_k

    # ─── Public API ──────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve the most relevant document chunks for *query*.
        Returns deduplicated chunks ordered by relevance.
        """
        if not self._em.is_ready:
            logger.warning("Retriever called but no index is loaded.")
            return []

        docs = self._em.search(query, k=self.top_k)
        docs = self._deduplicate(docs)
        logger.info(
            "Retrieved %d chunks for query: '%s'", len(docs), query[:60]
        )
        return docs

    def format_context(self, docs: list[Document]) -> str:
        """
        Build a single context block from retrieved *docs* for injection
        into the LLM prompt.
        """
        if not docs:
            return "No relevant documents found."

        sections: list[str] = []
        for i, doc in enumerate(docs, start=1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            header = f"[{i}] Source: {src} | Page: {page}"
            sections.append(f"{header}\n{doc.page_content.strip()}")

        return "\n\n---\n\n".join(sections)

    def get_sources(self, docs: list[Document]) -> list[dict]:
        """Return unique source references for citation in the UI."""
        seen: set[str] = set()
        sources: list[dict] = []
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            key = f"{src}::{page}"
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "filename": src,
                        "page": page,
                        "s3_key": doc.metadata.get("s3_key", ""),
                    }
                )
        return sources

    # ─── Private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(docs: list[Document]) -> list[Document]:
        """Remove exact-content duplicates while preserving order."""
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in docs:
            fingerprint = doc.page_content[:200]
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(doc)
        return unique
