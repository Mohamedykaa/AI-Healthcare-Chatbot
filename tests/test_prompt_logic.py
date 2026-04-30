"""Tests for prompt and language helper logic in src.core.logic."""

from src.core.logic import (
    build_system_prompt,
    get_triage_strategy,
    _detect_triage_phase,
    _check_sufficiency,
    contains_arabic,
    format_sources,
    get_preliminary_disclaimer,
    get_safe_fallback_answer,
    get_urgent_prefix,
    get_user_language,
)


def test_contains_arabic_detects_arabic_script():
    assert contains_arabic("مرحبا كيف حالك") is True


def test_contains_arabic_ignores_english_only_text():
    assert contains_arabic("hello how are you") is False


def test_get_user_language_returns_arabic_for_arabic_input():
    assert get_user_language("أشعر بألم في الصدر") == "ar"


def test_system_prompt_includes_context_guardrails():
    prompt = build_system_prompt("viral infections can cause fever")
    assert "reference evidence only" in prompt
    assert "Do not provide medication names, dosages, prescriptions, or treatment plans." in prompt


def test_initial_screening_screens_for_red_flags():
    """Phase INITIAL_SCREENING must instruct red flag screening."""
    strategy = get_triage_strategy("I have a headache", [], "ROUTINE")
    assert "red flag" in strategy.lower() or "loss of consciousness" in strategy.lower()
    # Must NOT be an assessment phase
    assert "— assessment" not in strategy.lower()


def test_characterization_identifies_missing_info():
    """Phase CHARACTERIZATION must mention what info is still missing."""
    from langchain_core.messages import HumanMessage, AIMessage
    history = [HumanMessage(content="I have a headache"), AIMessage(content="How long?")]
    strategy = get_triage_strategy("3 days", history, "ROUTINE")
    # Should mention what's still missing
    assert "missing" in strategy.lower() or "severity" in strategy.lower()


def test_differential_phase_when_sufficient_info():
    """After enough info is gathered (with red flags addressed), allow differential."""
    from langchain_core.messages import HumanMessage, AIMessage
    history = [
        HumanMessage(content="I've had headaches for 2 weeks"),
        AIMessage(content="How severe?"),
        HumanMessage(content="7/10, can't concentrate at work"),
        AIMessage(content="Any loss of consciousness or vision changes?"),
        HumanMessage(content="No loss of consciousness, no vision problems, but poor sleep and stress"),
        AIMessage(content="..."),
    ]
    strategy = get_triage_strategy("what do you think it could be?", history, "ROUTINE")
    assert "tier" in strategy.lower() or "common" in strategy.lower() or "ranked" in strategy.lower()


def test_urgent_gets_dedicated_path():
    """URGENT risk level must skip gradual pipeline and guard against ranked differential."""
    strategy = get_triage_strategy("severe chest pain", [], "URGENT")
    assert "same-day" in strategy.lower() or "urgent" in strategy.lower()
    assert "do not provide a ranked differential" in strategy.lower()


def test_sufficiency_check_detects_onset():
    from langchain_core.messages import HumanMessage, AIMessage
    history = [HumanMessage(content="headache for 3 days"), AIMessage(content="ok")]
    markers = _check_sufficiency(history)
    assert markers["onset_duration"] is True


def test_sufficiency_check_detects_severity():
    from langchain_core.messages import HumanMessage, AIMessage
    history = [HumanMessage(content="pain is 7/10 can't work"), AIMessage(content="ok")]
    markers = _check_sufficiency(history)
    assert markers["severity_impact"] is True


def test_sufficiency_includes_current_input():
    """_check_sufficiency must consider user_input, not just history."""
    from langchain_core.messages import HumanMessage, AIMessage
    history = [HumanMessage(content="I have a headache"), AIMessage(content="How long?")]
    # "3 days" is only in user_input, not in history
    markers = _check_sufficiency(history, user_input="for 3 days")
    assert markers["onset_duration"] is True


def test_red_flag_mandatory_for_differential():
    """DIFFERENTIAL must not be reached without red-flag status addressed."""
    from langchain_core.messages import HumanMessage, AIMessage
    # Has onset, severity, context — but NO red flag screening
    history = [
        HumanMessage(content="headache for 2 weeks, severe, poor sleep"),
        AIMessage(content="ok"),
        HumanMessage(content="can't work, very stressed"),
        AIMessage(content="ok"),
    ]
    phase = _detect_triage_phase(history, "ROUTINE", "still hurting")
    assert phase != "DIFFERENTIAL"  # must be CHARACTERIZATION, not differential


def test_first_turn_differential_with_rich_input():
    """If the very first message contains enough detail, skip screening."""
    rich_input = (
        "I've had a severe headache for 3 days, 8/10, can't work, "
        "no loss of consciousness, poor sleep and lots of stress"
    )
    phase = _detect_triage_phase([], "ROUTINE", rich_input)
    assert phase == "DIFFERENTIAL"


# ============================================================
# Arabic phase progression regression tests
# ============================================================


def test_arabic_initial_screening_on_first_turn():
    """Arabic symptom input on first turn with insufficient info → INITIAL_SCREENING."""
    phase = _detect_triage_phase([], "ROUTINE", "عندي صداع ودوخة")
    assert phase == "INITIAL_SCREENING"


def test_arabic_sufficiency_detects_onset_duration():
    from langchain_core.messages import HumanMessage, AIMessage
    history = [HumanMessage(content="عندي صداع منذ أسبوع"), AIMessage(content="ok")]
    markers = _check_sufficiency(history)
    assert markers["onset_duration"] is True


def test_arabic_sufficiency_detects_severity():
    from langchain_core.messages import HumanMessage, AIMessage
    history = [HumanMessage(content="الألم شديد ولا أستطيع النوم"), AIMessage(content="ok")]
    markers = _check_sufficiency(history)
    assert markers["severity_impact"] is True


def test_arabic_sufficiency_detects_red_flag_denial():
    from langchain_core.messages import HumanMessage, AIMessage
    history = [HumanMessage(content="لا فقدان وعي ولا زغللة"), AIMessage(content="ok")]
    markers = _check_sufficiency(history)
    assert markers["red_flag_addressed"] is True


def test_arabic_sufficiency_detects_context():
    from langchain_core.messages import HumanMessage, AIMessage
    history = [HumanMessage(content="نوم سيء وتوتر شديد"), AIMessage(content="ok")]
    markers = _check_sufficiency(history)
    assert markers["context_item"] is True


def test_arabic_full_phase_progression_to_differential():
    """Full Arabic conversation should reach DIFFERENTIAL when all markers met."""
    from langchain_core.messages import HumanMessage, AIMessage
    history = [
        HumanMessage(content="عندي صداع شديد منذ أسبوعين"),
        AIMessage(content="كم شدته من 10؟"),
        HumanMessage(content="8/10 ولا أستطيع العمل"),
        AIMessage(content="هل في فقدان وعي؟"),
        HumanMessage(content="لا فقدان وعي ولا زغللة، بس نوم سيء وتوتر"),
        AIMessage(content="..."),
    ]
    phase = _detect_triage_phase(history, "ROUTINE", "إيه رأيك؟")
    assert phase == "DIFFERENTIAL"


def test_system_prompt_handles_missing_context():
    prompt = build_system_prompt("")
    assert "No clearly relevant passages were retrieved" in prompt


def test_safe_fallback_answer_uses_arabic_when_needed():
    answer = get_safe_fallback_answer("أشعر بتعب شديد")
    assert "الأعراض" in answer


def test_urgent_prefix_uses_matching_language():
    assert "URGENT ADVICE REQUIRED" in get_urgent_prefix("I have chest pain")
    assert "نصيحة عاجلة" in get_urgent_prefix("لدي ألم في الصدر")


def test_preliminary_disclaimer_uses_matching_language():
    assert "preliminary educational assessment" in get_preliminary_disclaimer("I have a headache")
    assert "تقييم تعليمي أولي" in get_preliminary_disclaimer("أعاني من صداع")


def test_format_sources_uses_default_label_for_blank_source(make_doc):
    result = format_sources([make_doc("")])
    assert "Medical Knowledge Base" in result
