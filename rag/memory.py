"""
app/rag/memory.py
Manages conversation history for the RAG chain.
Keeps a rolling window of the last N message pairs.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from app.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ConversationMemory:
    """
    Rolling-window conversation buffer.

    >>> mem = ConversationMemory()
    >>> mem.add("user", "What is GDPR?")
    >>> mem.add("assistant", "GDPR stands for…")
    >>> mem.to_langchain_messages()
    [HumanMessage(…), AIMessage(…)]
    """

    def __init__(self, max_messages: int | None = None):
        self._max = max_messages or settings.max_history_messages
        self._messages: list[Message] = []

    # ─── Public API ──────────────────────────────────────────────────────

    def add(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))
        # Trim oldest pairs to stay within the window
        while len(self._messages) > self._max * 2:
            self._messages.pop(0)

    def to_langchain_messages(self) -> list:
        """Convert to LangChain HumanMessage / AIMessage objects."""
        from langchain_core.messages import AIMessage, HumanMessage

        lc_msgs = []
        for msg in self._messages:
            if msg.role == "user":
                lc_msgs.append(HumanMessage(content=msg.content))
            else:
                lc_msgs.append(AIMessage(content=msg.content))
        return lc_msgs

    def to_dict_list(self) -> list[dict]:
        """Return messages as plain dicts (for Streamlit rendering)."""
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def clear(self) -> None:
        self._messages.clear()
        logger.info("Conversation history cleared.")

    @property
    def is_empty(self) -> bool:
        return len(self._messages) == 0

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def format_for_prompt(self) -> str:
        """Render history as a plain-text block for injection into prompts."""
        if self.is_empty:
            return "No prior conversation."
        lines = []
        for msg in self._messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)
