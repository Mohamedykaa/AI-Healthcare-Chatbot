"""
Emergency Detection Tests
===========================

Tests for risk assessment and emergency response.
Uses backend.risk (dependency-free module shared with backend.core).
"""

import pytest
from src.core.risk import (
    CRITICAL_KEYWORDS,
    check_for_emergency,
    get_emergency_response,
)


class TestEmergencyPositiveDetection:
    @pytest.mark.parametrize("keyword", CRITICAL_KEYWORDS)
    def test_bare_keyword_triggers(self, keyword):
        assert check_for_emergency(keyword) is True


class TestEmergencyNegativeDetection:
    @pytest.mark.parametrize("text", [
        "I have a headache",
        "My stomach hurts",
        "I feel tired",
        "I have a runny nose and cold",
        "What causes diabetes?",
        "I have a rash on my arm",
        "My back is sore",
        "I feel nauseous",
        "I have a sore throat",
        "What is hypertension?",
    ])
    def test_common_symptom_does_not_trigger(self, text):
        assert check_for_emergency(text) is False


class TestEmergencyEdgeCases:
    def test_empty_string(self):
        assert check_for_emergency("") is False

    def test_whitespace_only(self):
        assert check_for_emergency("   \n\t  ") is False

    def test_keyword_at_start_of_sentence(self):
        assert check_for_emergency("Stroke symptoms started suddenly") is True

    def test_substring_stroke_in_stroked_is_not_emergency(self):
        assert check_for_emergency("I stroked the cat") is False


class TestEmergencyResponseContent:
    @pytest.fixture(autouse=True)
    def _response(self):
        self.response = get_emergency_response()

    def test_contains_alert_header(self):
        assert "EMERGENCY ALERT" in self.response

    def test_contains_911(self):
        assert "911" in self.response

    def test_contains_disclaimer(self):
        assert "I am an AI" in self.response

    def test_contains_action_items(self):
        assert "Call Emergency Services" in self.response
        assert "Emergency Room" in self.response

    def test_is_nonempty_string(self):
        assert isinstance(self.response, str)
        assert len(self.response) > 100
