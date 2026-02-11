"""
Safety Filter Tests
====================

Tests for the ingestion pipeline's content safety filter:
- contains_forbidden_content()  — single-field keyword check
- is_safe_entry()               — multi-field aggregation

These filters prevent dosage, prescription, and treatment content
from entering the RAG knowledge base.
"""

import pytest
from typing import Optional


# ============================================================
# Reproduce pure functions from ingest_data.py (lines 95-118)
# ============================================================

FORBIDDEN_KEYWORDS = [
    "dosage", "dose", "mg", "ml",
    "tablet", "capsule", "injection",
    "prescribe", "prescription",
    "treatment plan", "therapy", "surgery",
    "emergency",
    "diagnose", "diagnosis",
    "medication", "drug",
]


def contains_forbidden_content(text: Optional[str]) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in text_lower:
            return True
    return False


def is_safe_entry(*fields: Optional[str]) -> bool:
    for field in fields:
        if contains_forbidden_content(field):
            return False
    return True


# ===========================================================================
# CONTAINS_FORBIDDEN_CONTENT — Keyword detection
# ===========================================================================

class TestContainsForbiddenContent:
    """Every FORBIDDEN_KEYWORD must be detected."""

    @pytest.mark.parametrize("keyword", FORBIDDEN_KEYWORDS)
    def test_bare_keyword_detected(self, keyword):
        assert contains_forbidden_content(keyword) is True

    @pytest.mark.parametrize("keyword", FORBIDDEN_KEYWORDS)
    def test_keyword_in_sentence(self, keyword):
        text = f"The recommended {keyword} for this condition is important."
        assert contains_forbidden_content(text) is True

    @pytest.mark.parametrize("keyword", FORBIDDEN_KEYWORDS)
    def test_keyword_uppercase_detected(self, keyword):
        assert contains_forbidden_content(keyword.upper()) is True

    @pytest.mark.parametrize("keyword", FORBIDDEN_KEYWORDS)
    def test_keyword_mixed_case_detected(self, keyword):
        mixed = keyword[0].upper() + keyword[1:]
        assert contains_forbidden_content(mixed) is True


class TestContainsForbiddenContentNegative:
    """Medical terms that are NOT forbidden."""

    @pytest.mark.parametrize("text", [
        "symptom",
        "fever",
        "cough",
        "headache",
        "blood pressure",
        "heart rate",
        "inflammation",
        "What is hypertension?",
        "Describe the common cold.",
        "Risk factors for diabetes include obesity.",
    ])
    def test_safe_medical_terms(self, text):
        assert contains_forbidden_content(text) is False


class TestContainsForbiddenContentEdge:
    """Boundary and null inputs."""

    def test_none_returns_false(self):
        assert contains_forbidden_content(None) is False

    def test_empty_string_returns_false(self):
        assert contains_forbidden_content("") is False

    def test_whitespace_only_returns_false(self):
        assert contains_forbidden_content("   ") is False

    def test_multiword_keyword_treatment_plan(self):
        assert contains_forbidden_content("Follow this treatment plan carefully") is True

    def test_partial_keyword_not_standalone(self):
        # "dos" is NOT "dose" or "dosage" — should be safe
        assert contains_forbidden_content("dos") is False

    def test_keyword_embedded_in_longer_word(self):
        # "drug" appears in "drug" but also in "drugstore"
        assert contains_forbidden_content("drugstore") is True  # substring match


# ===========================================================================
# IS_SAFE_ENTRY — Multi-field aggregation
# ===========================================================================

class TestIsSafeEntry:
    """All fields must be clean for the entry to be safe."""

    def test_all_safe_fields(self):
        assert is_safe_entry("What is a fever?", "A fever is elevated temperature.") is True

    def test_one_forbidden_field(self):
        assert is_safe_entry("What is the dosage?", "A fever is elevated temperature.") is False

    def test_second_field_forbidden(self):
        assert is_safe_entry("Safe question", "Take one tablet daily.") is False

    def test_no_fields(self):
        assert is_safe_entry() is True

    def test_single_safe_field(self):
        assert is_safe_entry("healthy eating habits") is True

    def test_single_forbidden_field(self):
        assert is_safe_entry("recommended medication") is False

    def test_none_field_treated_as_safe(self):
        assert is_safe_entry(None) is True

    def test_mixed_none_and_safe(self):
        assert is_safe_entry("safe text", None, "also safe") is True

    def test_mixed_none_and_forbidden(self):
        assert is_safe_entry("safe text", None, "take this drug") is False

    def test_three_safe_fields(self):
        assert is_safe_entry("field1", "field2", "field3") is True

    def test_all_forbidden_fields(self):
        assert is_safe_entry("dosage info", "tablet form", "injection site") is False

    def test_last_field_forbidden(self):
        assert is_safe_entry("safe", "safe", "safe", "contains diagnosis") is False
