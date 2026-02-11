"""Tests for deterministic triage risk scoring behavior."""

import re


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


def test_hard_stop_suicidal_is_emergency():
    assert assess_risk_level("I feel suicidal") == "EMERGENCY"


def test_hard_stop_loss_of_consciousness_is_emergency():
    assert assess_risk_level("there was loss of consciousness") == "EMERGENCY"


def test_heatstroke_is_explicit_hard_stop():
    assert assess_risk_level("heatstroke symptoms are severe") == "EMERGENCY"


def test_sunstroke_is_explicit_hard_stop():
    assert assess_risk_level("possible sunstroke after sun exposure") == "EMERGENCY"


def test_crushing_chest_pain_with_red_flags_is_emergency():
    text = "crushing chest pain with cold sweating and shortness of breath"
    assert assess_risk_level(text) == "EMERGENCY"


def test_mild_pressing_chest_pain_is_routine():
    assert assess_risk_level("mild chest pain only when pressing") == "ROUTINE"


def test_headache_is_routine():
    assert assess_risk_level("I have a headache") == "ROUTINE"


def test_boundary_aware_matching_avoids_partial_words():
    assert assess_risk_level("I stroked the cat") == "ROUTINE"


def test_normalization_handles_case_and_punctuation():
    assert assess_risk_level("Severe Chest Pain!!!") == "URGENT"


def test_overlap_does_not_duplicate_same_phrase_points():
    text = "shortness of breath shortness of breath"
    assert calculate_risk_score(text) == 3


def test_compatibility_helper_matches_emergency_assessment():
    for text in ["I feel suicidal", "crushing chest pain", "mild headache"]:
        assert check_for_emergency(text) == (assess_risk_level(text) == "EMERGENCY")


def test_mixed_case_complex_transition_to_emergency():
    text = "I had Mild Chest Pain yesterday, but now I have SHORTNESS OF BREATH and cold sweating!!!"
    assert assess_risk_level(text) == "EMERGENCY"
