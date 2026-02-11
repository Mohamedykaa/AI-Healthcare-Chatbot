"""
Emergency Detection Tests
===========================

Tests for check_for_emergency() and get_emergency_response().
"""

import re
import pytest


CRITICAL_KEYWORDS = [
    "heart attack", "stroke", "severe bleeding", "loss of consciousness",
    "unconscious", "seizure", "severe head injury", "poisoning", "overdose",
    "suicidal", "suicide", "self harm", "severe allergic reaction",
    "anaphylaxis", "choking", "heatstroke", "sunstroke",
]

ASSOCIATED_RED_FLAGS = ["shortness of breath", "cold sweating"]
CORE_SYMPTOM_KEYWORDS = ["chest pain", "cannot breathe", "cant breathe", "difficulty breathing", "fainting"]
SEVERE_MODIFIERS = ["severe", "crushing", "worst", "sudden"]
LOW_RISK_MODIFIERS = ["mild", "localized", "only when pressing", "brief"]

EMERGENCY_SCORE_THRESHOLD = 6
URGENT_SCORE_THRESHOLD = 3


def normalize_input(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered)


def contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _match_category_phrases(normalized_input: str, phrases: list[str], used_phrases: set[str]) -> set[str]:
    matched = set()
    for phrase in phrases:
        if phrase in used_phrases:
            continue
        if contains_phrase(normalized_input, phrase):
            matched.add(phrase)
    return matched


def calculate_risk_score(user_input: str) -> int:
    normalized = normalize_input(user_input)
    used_phrases: set[str] = set()
    score = 0

    red_flag_matches = _match_category_phrases(normalized, ASSOCIATED_RED_FLAGS, used_phrases)
    used_phrases.update(red_flag_matches)
    score += 3 * len(red_flag_matches)

    core_matches = _match_category_phrases(normalized, CORE_SYMPTOM_KEYWORDS, used_phrases)
    used_phrases.update(core_matches)
    score += 3 * len(core_matches)

    severe_matches = _match_category_phrases(normalized, SEVERE_MODIFIERS, used_phrases)
    used_phrases.update(severe_matches)
    score += 2 * len(severe_matches)

    low_risk_matches = _match_category_phrases(normalized, LOW_RISK_MODIFIERS, used_phrases)
    score -= 2 * len(low_risk_matches)

    return score


def assess_risk_level(user_input: str) -> str:
    normalized = normalize_input(user_input)
    for keyword in CRITICAL_KEYWORDS:
        if contains_phrase(normalized, keyword):
            return "EMERGENCY"

    score = calculate_risk_score(user_input)
    if score >= EMERGENCY_SCORE_THRESHOLD:
        return "EMERGENCY"
    if score >= URGENT_SCORE_THRESHOLD:
        return "URGENT"
    return "ROUTINE"


def check_for_emergency(user_input: str) -> bool:
    return assess_risk_level(user_input) == "EMERGENCY"


def get_emergency_response() -> str:
    return """🚨 **EMERGENCY ALERT** 🚨

Based on your description, this may be a medical emergency requiring immediate attention.

**IMPORTANT:**
- I cannot provide a diagnosis for emergency symptoms.
- Please seek immediate medical help.

**Recommended Actions:**
1. **Call Emergency Services** (911, 999, or your local emergency number)
2. **Go to the nearest Emergency Room** immediately
3. **Do not delay** - time is critical in medical emergencies

If you're with someone experiencing these symptoms:
- Stay calm and keep them comfortable
- Do not give them food or water unless instructed by medical personnel
- Be ready to describe the symptoms to emergency responders

**Remember:** I am an AI and cannot replace emergency medical care. Your safety is the priority.

---
*This is an automated safety response. Please seek professional medical attention immediately.*
"""


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
