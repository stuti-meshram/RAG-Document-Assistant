"""
app/utils/pdf_processor.py
Handles PDF parsing and text chunking for the RAG pipeline.
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class PDFMetadata:
    filename: str
    num_pages: int
    file_size_bytes: int
    s3_key: str = ""


class PDFProcessor:
    """Extracts text from PDFs and splits into overlapping chunks."""

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ─── Public API ──────────────────────────────────────────────────────

    def process(
        self,
        file_bytes: bytes,
        filename: str,
        s3_key: str = "",
    ) -> tuple[list[Document], PDFMetadata]:
        """
        Parse *file_bytes* as a PDF, extract text page-by-page,
        split into chunks and return (documents, metadata).
        """
        pages = self._extract_pages(file_bytes, filename)
        metadata = PDFMetadata(
            filename=filename,
            num_pages=len(pages),
            file_size_bytes=len(file_bytes),
            s3_key=s3_key,
        )
        documents = self._build_documents(pages, metadata)
        logger.info(
            "Processed '%s': %d pages → %d chunks",
            filename,
            metadata.num_pages,
            len(documents),
        )
        return documents, metadata

    # ─── Private helpers ─────────────────────────────────────────────────

    def _extract_pages(self, file_bytes: bytes, filename: str) -> list[dict]:
        """Return list of {page_number, text} dicts."""
        pages: list[dict] = []
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    text = self._clean(text)
                    if text:
                        pages.append({"page_number": i, "text": text})
        except Exception as exc:
            logger.error("Failed to parse '%s': %s", filename, exc)
            raise ValueError(f"Could not read PDF '{filename}': {exc}") from exc
        return pages

    def _build_documents(
        self, pages: list[dict], meta: PDFMetadata
    ) -> list[Document]:
        """Chunk each page and wrap in LangChain Document objects."""
        docs: list[Document] = []
        for page in pages:
            chunks = self.splitter.split_text(page["text"])
            for idx, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": meta.filename,
                            "page": page["page_number"],
                            "chunk": idx,
                            "s3_key": meta.s3_key,
                            "total_pages": meta.num_pages,
                        },
                    )
                )
        return docs

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise whitespace and remove junk characters."""
        lines = (line.strip() for line in text.splitlines())
        return "\n".join(line for line in lines if line)
