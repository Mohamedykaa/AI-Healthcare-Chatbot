"""Tests for deterministic triage risk scoring behavior."""

from src.core.risk import (
    assess_risk_level,
    calculate_risk_score,
    check_for_emergency,
)


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


def test_arabic_chest_pain_with_red_flags_is_emergency():
    text = "لدي ألم في الصدر مع ضيق تنفس وتعرق بارد"
    assert assess_risk_level(text) == "EMERGENCY"


def test_arabic_mild_pressing_chest_pain_is_routine():
    text = "لدي ألم في الصدر خفيف وعند الضغط فقط"
    assert assess_risk_level(text) == "ROUTINE"
