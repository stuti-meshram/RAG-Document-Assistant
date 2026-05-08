"""
app/rag/chain.py
The core RAG chain: retrieve → prompt → generate → respond.
Uses LangChain's ChatOpenAI with a structured system prompt and
conversation memory injected at each turn.
"""

import logging
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.rag.memory import ConversationMemory
from app.rag.retriever import DocumentRetriever
from app.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── System prompt template ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert document assistant. Your role is to answer questions
accurately and concisely based ONLY on the provided document context.

Guidelines:
- Answer based strictly on the retrieved context below.
- If the answer isn't in the context, say "I couldn't find that in the uploaded documents."
- Quote or reference specific parts of documents when useful.
- Be concise but thorough. Use bullet points for lists.
- Maintain the conversation history for context-aware follow-up answers.

──────────────── DOCUMENT CONTEXT ────────────────
{context}
──────────────────────────────────────────────────

──────────────── CONVERSATION HISTORY ────────────
{history}
──────────────────────────────────────────────────
"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    retrieved_docs: list[Document]


class RAGChain:
    """
    Orchestrates the full RAG pipeline.

    Usage
    -----
    chain = RAGChain(retriever, memory)
    response = chain.ask("What are the payment terms?")
    print(response.answer)
    print(response.sources)
    """

    def __init__(self, retriever: DocumentRetriever, memory: ConversationMemory):
        self._retriever = retriever
        self._memory = memory
        self._llm = ChatOpenAI(
            model=settings.openai_chat_model,
            openai_api_key=settings.openai_api_key,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            streaming=False,
        )

    # ─── Public API ──────────────────────────────────────────────────────

    def ask(self, question: str) -> RAGResponse:
        """
        Run the full RAG pipeline for *question*:
          1. Retrieve relevant chunks
          2. Build the prompt with context + history
          3. Call the LLM
          4. Update memory
          5. Return structured response
        """
        # 1. Retrieve
        docs = self._retriever.retrieve(question)
        context = self._retriever.format_context(docs)
        sources = self._retriever.get_sources(docs)

        # 2. Build messages
        system_content = SYSTEM_PROMPT.format(
            context=context,
            history=self._memory.format_for_prompt(),
        )
        messages = [SystemMessage(content=system_content)]
        messages += self._memory.to_langchain_messages()

        # Add the new question from the user
        from langchain_core.messages import HumanMessage
        messages.append(HumanMessage(content=question))

        # 3. Generate
        logger.info("Calling LLM for question: '%s'", question[:80])
        try:
            ai_response = self._llm.invoke(messages)
            answer: str = ai_response.content  # type: ignore[assignment]
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            answer = "Sorry, I encountered an error while generating the answer. Please try again."

        # 4. Update memory
        self._memory.add("user", question)
        self._memory.add("assistant", answer)

        return RAGResponse(answer=answer, sources=sources, retrieved_docs=docs)

    def ask_streaming(self, question: str):
        """
        Generator that yields answer tokens one-by-one for streaming UIs.
        Also updates memory once the full response is assembled.
        """
        docs = self._retriever.retrieve(question)
        context = self._retriever.format_context(docs)
        sources = self._retriever.get_sources(docs)

        system_content = SYSTEM_PROMPT.format(
            context=context,
            history=self._memory.format_for_prompt(),
        )

        from langchain_core.messages import HumanMessage
        messages = [SystemMessage(content=system_content)]
        messages += self._memory.to_langchain_messages()
        messages.append(HumanMessage(content=question))

        full_answer = ""
        stream_llm = ChatOpenAI(
            model=settings.openai_chat_model,
            openai_api_key=settings.openai_api_key,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            streaming=True,
        )
        for chunk in stream_llm.stream(messages):
            token: str = chunk.content or ""  # type: ignore[assignment]
            full_answer += token
            yield token, sources

        self._memory.add("user", question)
        self._memory.add("assistant", full_answer)
