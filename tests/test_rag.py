"""
tests/test_rag.py
Unit tests for the RAG pipeline (memory, retriever, processor).
All OpenAI/FAISS calls are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.rag.memory import ConversationMemory, Message
from app.rag.retriever import DocumentRetriever
from app.utils.pdf_processor import PDFProcessor


# ── ConversationMemory ────────────────────────────────────────────────────────

class TestConversationMemory:
    def test_add_and_retrieve(self):
        mem = ConversationMemory(max_messages=5)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there!")
        msgs = mem.to_dict_list()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["content"] == "Hi there!"

    def test_rolling_window(self):
        mem = ConversationMemory(max_messages=2)
        for i in range(6):
            mem.add("user", f"msg {i}")
            mem.add("assistant", f"reply {i}")
        # max_messages=2 means 4 stored (2 pairs)
        assert mem.message_count <= 4

    def test_clear(self):
        mem = ConversationMemory()
        mem.add("user", "Hello")
        mem.clear()
        assert mem.is_empty

    def test_format_for_prompt(self):
        mem = ConversationMemory()
        mem.add("user", "What is RAG?")
        mem.add("assistant", "RAG stands for Retrieval-Augmented Generation.")
        prompt = mem.format_for_prompt()
        assert "User:" in prompt
        assert "Assistant:" in prompt
        assert "RAG" in prompt

    def test_to_langchain_messages(self):
        from langchain_core.messages import AIMessage, HumanMessage
        mem = ConversationMemory()
        mem.add("user", "Hello")
        mem.add("assistant", "Hi")
        lc = mem.to_langchain_messages()
        assert isinstance(lc[0], HumanMessage)
        assert isinstance(lc[1], AIMessage)


# ── DocumentRetriever ─────────────────────────────────────────────────────────

class TestDocumentRetriever:
    def _make_retriever(self, docs=None):
        em = MagicMock()
        em.is_ready = True
        em.search.return_value = docs or []
        return DocumentRetriever(em, top_k=4)

    def test_retrieve_returns_docs(self):
        doc = Document(
            page_content="OpenAI is an AI company.",
            metadata={"source": "report.pdf", "page": 1, "s3_key": ""},
        )
        retriever = self._make_retriever([doc])
        results = retriever.retrieve("What is OpenAI?")
        assert len(results) == 1
        assert results[0].page_content == "OpenAI is an AI company."

    def test_retrieve_returns_empty_when_not_ready(self):
        em = MagicMock()
        em.is_ready = False
        retriever = DocumentRetriever(em)
        assert retriever.retrieve("anything") == []

    def test_deduplication(self):
        same_content = "Duplicate content chunk here."
        docs = [
            Document(page_content=same_content, metadata={"source": "a.pdf", "page": 1}),
            Document(page_content=same_content, metadata={"source": "b.pdf", "page": 2}),
        ]
        retriever = self._make_retriever(docs)
        results = retriever.retrieve("test query")
        assert len(results) == 1  # deduplicated

    def test_format_context(self):
        doc = Document(
            page_content="Some content.",
            metadata={"source": "file.pdf", "page": 3},
        )
        em = MagicMock()
        em.is_ready = True
        retriever = DocumentRetriever(em)
        ctx = retriever.format_context([doc])
        assert "file.pdf" in ctx
        assert "Page: 3" in ctx
        assert "Some content." in ctx

    def test_format_context_empty(self):
        em = MagicMock()
        retriever = DocumentRetriever(em)
        ctx = retriever.format_context([])
        assert "No relevant documents" in ctx

    def test_get_sources_deduplicates(self):
        docs = [
            Document(page_content="a", metadata={"source": "x.pdf", "page": 1, "s3_key": ""}),
            Document(page_content="b", metadata={"source": "x.pdf", "page": 1, "s3_key": ""}),
            Document(page_content="c", metadata={"source": "x.pdf", "page": 2, "s3_key": ""}),
        ]
        em = MagicMock()
        retriever = DocumentRetriever(em)
        sources = retriever.get_sources(docs)
        assert len(sources) == 2  # page 1 deduplicated


# ── PDFProcessor ──────────────────────────────────────────────────────────────

class TestPDFProcessor:
    def test_clean_text(self):
        raw = "  Hello   \n\n  World  \n"
        cleaned = PDFProcessor._clean(raw)
        assert "Hello" in cleaned
        assert "World" in cleaned

    def test_clean_removes_blank_lines(self):
        raw = "\n\n\nonly content\n\n"
        cleaned = PDFProcessor._clean(raw)
        assert cleaned == "only content"

    @patch("app.utils.pdf_processor.pdfplumber.open")
    def test_process_returns_documents(self, mock_open):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is page one content. " * 20
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]
        mock_open.return_value = mock_pdf

        proc = PDFProcessor()
        docs, meta = proc.process(b"fake-bytes", "test.pdf")

        assert len(docs) >= 1
        assert meta.filename == "test.pdf"
        assert meta.num_pages == 1

    @patch("app.utils.pdf_processor.pdfplumber.open")
    def test_process_raises_on_bad_pdf(self, mock_open):
        mock_open.side_effect = Exception("corrupt file")
        proc = PDFProcessor()
        with pytest.raises(ValueError, match="Could not read PDF"):
            proc.process(b"bad", "corrupt.pdf")
