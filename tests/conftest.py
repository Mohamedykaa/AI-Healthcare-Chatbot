"""
Shared pytest fixtures for the AI Healthcare Chatbot test suite.
================================================================

Provides reusable test objects that avoid importing production modules
(which trigger ChromaDB/Ollama initialization at module level).
"""

import os
import re
import pytest


# ============================================================
# FAKE DOCUMENT (stands in for langchain_core.documents.Document)
# ============================================================

class FakeDocument:
    """Minimal stand-in for langchain_core.documents.Document.

    Mirrors only the two attributes used by format_sources():
      - page_content: str
      - metadata: dict
    """

    def __init__(self, page_content: str = "", metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        src = self.metadata.get("source", "<no source>")
        return f"FakeDocument(source={src!r})"


@pytest.fixture
def make_doc():
    """Factory fixture — returns a FakeDocument builder."""
    def _make(source: str = "data/default.txt", content: str = ""):
        return FakeDocument(page_content=content, metadata={"source": source})
    return _make


@pytest.fixture
def make_doc_no_source():
    """Factory fixture — returns a FakeDocument with no 'source' key."""
    def _make(content: str = ""):
        return FakeDocument(page_content=content, metadata={})
    return _make


# ============================================================
# FAKE MESSAGE (stands in for HumanMessage / AIMessage)
# ============================================================

class FakeMessage:
    """Minimal stand-in for LangChain message objects."""

    def __init__(self, content: str, role: str = "human"):
        self.content = content
        self._role = role

    @property
    def is_human(self):
        return self._role == "human"


@pytest.fixture
def build_chat_history():
    """Factory fixture — builds a list of alternating Human/AI messages.

    Usage:
        history = build_chat_history(3)  # 3 exchanges = 6 messages
    """
    def _build(num_exchanges: int):
        history = []
        for i in range(num_exchanges):
            history.append(FakeMessage(f"User question {i+1}", role="human"))
            history.append(FakeMessage(f"AI answer {i+1}", role="ai"))
        return history
    return _build
