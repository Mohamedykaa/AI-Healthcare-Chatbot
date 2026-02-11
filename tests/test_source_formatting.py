"""
Source Formatting Tests
========================

Tests for format_sources() — the function that produces the
"📚 References:" citation line at the bottom of each response.

Uses FakeDocument from conftest.py to avoid LangChain imports.
"""

import os
import pytest


# ============================================================
# Reproduce format_sources from app.py (lines 352-371)
# ============================================================

def format_sources(context_documents) -> str:
    if not context_documents:
        return ""
    sources = set()
    for doc in context_documents:
        source = doc.metadata.get("source", "Medical Knowledge Base")
        source_name = os.path.basename(source)
        sources.add(source_name)
    if not sources:
        return ""
    return "\n\n---\n**📚 References:** " + ", ".join(sorted(sources))


# ===========================================================================
# EMPTY / NONE INPUTS
# ===========================================================================

class TestFormatSourcesEmpty:
    """format_sources must handle empty or absent input gracefully."""

    def test_empty_list(self):
        assert format_sources([]) == ""

    def test_empty_tuple(self):
        assert format_sources(()) == ""


# ===========================================================================
# SINGLE SOURCE
# ===========================================================================

class TestFormatSourcesSingle:
    """Single-document formatting."""

    def test_extracts_basename(self, make_doc):
        docs = [make_doc("data/medical_knowledge_medquad.txt")]
        result = format_sources(docs)
        assert "medical_knowledge_medquad.txt" in result

    def test_includes_reference_emoji(self, make_doc):
        docs = [make_doc("data/file.txt")]
        result = format_sources(docs)
        assert "📚 References:" in result

    def test_starts_with_separator(self, make_doc):
        docs = [make_doc("data/file.txt")]
        result = format_sources(docs)
        assert result.startswith("\n\n---\n")


# ===========================================================================
# MULTIPLE SOURCES
# ===========================================================================

class TestFormatSourcesMultiple:
    """Multiple documents with sorting and deduplication."""

    def test_sorted_alphabetically(self, make_doc):
        docs = [
            make_doc("data/public_health.txt"),
            make_doc("data/anatomy.txt"),
            make_doc("data/medquad.txt"),
        ]
        result = format_sources(docs)
        # Extract just the reference portion after "References:** "
        refs = result.split("References:** ")[1]
        names = [n.strip() for n in refs.split(",")]
        assert names == sorted(names)

    def test_duplicates_deduplicated(self, make_doc):
        docs = [
            make_doc("data/medquad.txt"),
            make_doc("data/medquad.txt"),
            make_doc("data/medquad.txt"),
        ]
        result = format_sources(docs)
        assert result.count("medquad.txt") == 1

    def test_different_paths_same_basename_deduplicated(self, make_doc):
        docs = [
            make_doc("data/v1/file.txt"),
            make_doc("data/v2/file.txt"),
        ]
        result = format_sources(docs)
        # Both resolve to "file.txt" basename — should appear once
        assert result.count("file.txt") == 1


# ===========================================================================
# MISSING / MALFORMED METADATA
# ===========================================================================

class TestFormatSourcesMetadata:
    """Graceful handling of missing or unusual metadata."""

    def test_missing_source_key_uses_default(self, make_doc_no_source):
        docs = [make_doc_no_source()]
        result = format_sources(docs)
        assert "Medical Knowledge Base" in result

    def test_empty_source_string(self, make_doc):
        docs = [make_doc("")]
        result = format_sources(docs)
        # os.path.basename("") returns "" — but it's still in the set
        assert isinstance(result, str)

    def test_source_with_nested_path(self, make_doc):
        docs = [make_doc("/usr/local/data/deep/nested/knowledge.txt")]
        result = format_sources(docs)
        assert "knowledge.txt" in result
        assert "/usr/local" not in result
