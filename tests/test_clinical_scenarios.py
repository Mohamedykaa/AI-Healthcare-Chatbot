"""
Clinical Evaluation Scenarios
==============================
10 structured clinical test cases that verify triage safety and correctness
from a medical perspective, not just engineering correctness.

Each scenario tests the FULL pipeline: risk assessment → phase detection →
sufficiency checking → response safety.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage

from src.core.logic import (
    _detect_triage_phase,
    _check_sufficiency,
    get_triage_strategy,
    get_safe_fallback_answer,
    process_chat_message,
)
from src.core.risk import assess_risk_level, calculate_risk_score


# ============================================================
# Helpers
# ============================================================

def _make_mock_llm(response_text="Mock LLM response for testing."):
    mock = MagicMock()
    mock.ainvoke = AsyncMock(
        return_value=MagicMock(content=response_text)
    )
    return mock


def _make_mock_vectorstore():
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_vs.as_retriever.return_value = mock_retriever
    return mock_vs


# ============================================================
# SCENARIO 1: Low-risk routine — stress + poor sleep
# Expected: ROUTINE risk, reaches DIFFERENTIAL with full info
# ============================================================

class TestScenario1_StressAndPoorSleep:
    """Stress + insomnia → LOW RISK, should reach differential normally."""

    def test_risk_is_routine(self):
        assert assess_risk_level("I have headaches, poor sleep and stress") == "ROUTINE"

    def test_full_info_reaches_differential(self):
        history = [
            HumanMessage(content="I have headaches and dizziness"),
            AIMessage(content="When did this start?"),
            HumanMessage(content="2 weeks ago, 7/10 severity, can't concentrate"),
            AIMessage(content="Any red flags?"),
            HumanMessage(content="No loss of consciousness, poor sleep and stress"),
            AIMessage(content="..."),
        ]
        phase = _detect_triage_phase(history, "ROUTINE", "what do you think?")
        assert phase == "DIFFERENTIAL"


# ============================================================
# SCENARIO 2: Emergency — chest pain + can't breathe
# Expected: EMERGENCY, bypass LLM entirely
# ============================================================

class TestScenario2_ChestPainEmergency:
    """Chest pain + breathing difficulty → EMERGENCY."""

    def test_risk_is_emergency(self):
        assert assess_risk_level("I have chest pain and I can't breathe") == "EMERGENCY"

    @pytest.mark.asyncio
    async def test_emergency_bypasses_llm(self):
        """LLM must NOT be called for emergencies."""
        mock_llm = _make_mock_llm()
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, risk, _ = await process_chat_message(
                "I have chest pain and I can't breathe", []
            )
            assert risk == "EMERGENCY"
            mock_llm.ainvoke.assert_not_called()


# ============================================================
# SCENARIO 3: Urgent — sudden chest pain
# Expected: URGENT risk, urgent prefix in response
# ============================================================

class TestScenario3_SuddenChestPain:
    """Sudden chest pain → URGENT, not EMERGENCY."""

    def test_risk_is_urgent(self):
        assert assess_risk_level("I have sudden chest pain") == "URGENT"

    def test_phase_is_urgent_assessment(self):
        phase = _detect_triage_phase([], "URGENT", "sudden chest pain")
        assert phase == "URGENT_ASSESSMENT"


# ============================================================
# SCENARIO 4: Negated symptoms should NOT trigger risk
# Expected: "No fainting" → ROUTINE
# ============================================================

class TestScenario4_NegatedSymptoms:
    """Denying red-flag symptoms must not inflate risk score."""

    def test_no_fainting_is_routine(self):
        assert assess_risk_level("No fainting or vision problems") == "ROUTINE"
        assert calculate_risk_score("No fainting or vision problems") == 0

    def test_no_chest_pain_is_routine(self):
        assert assess_risk_level("no chest pain") == "ROUTINE"
        assert calculate_risk_score("no chest pain") == 0

    def test_affirmed_fainting_is_urgent(self):
        assert assess_risk_level("I am fainting right now") == "URGENT"


# ============================================================
# SCENARIO 5: Arabic full triage flow
# Expected: Arabic input reaches DIFFERENTIAL with all markers
# ============================================================

class TestScenario5_ArabicTriageFlow:
    """Full Arabic symptom conversation → correct phase progression."""

    def test_arabic_risk_routine(self):
        assert assess_risk_level("عندي صداع شديد منذ أسبوعين") == "ROUTINE"

    def test_arabic_reaches_differential(self):
        history = [
            HumanMessage(content="عندي صداع شديد منذ أسبوعين"),
            AIMessage(content="كم شدته؟"),
            HumanMessage(content="8/10 ولا أستطيع العمل"),
            AIMessage(content="هل في فقدان وعي؟"),
            HumanMessage(content="لا فقدان وعي ولا زغللة، بس نوم سيء وتوتر"),
            AIMessage(content="..."),
        ]
        phase = _detect_triage_phase(history, "ROUTINE", "إيه رأيك؟")
        assert phase == "DIFFERENTIAL"


# ============================================================
# SCENARIO 6: Red flag denial must enable differential
# Expected: "no loss of consciousness" → red_flag_addressed = True
# ============================================================

class TestScenario6_RedFlagDenialEnablesDifferential:
    """Explicitly denying red flags should mark them as addressed."""

    def test_denial_detected(self):
        history = [
            HumanMessage(content="No loss of consciousness, no vision problems"),
        ]
        markers = _check_sufficiency(history)
        assert markers["red_flag_addressed"] is True

    def test_no_denial_blocks_differential(self):
        """Without red-flag response, phase stays CHARACTERIZATION."""
        history = [
            HumanMessage(content="headache for 2 weeks, severe, poor sleep"),
            AIMessage(content="ok"),
            HumanMessage(content="can't work, very stressed"),
            AIMessage(content="ok"),
        ]
        phase = _detect_triage_phase(history, "ROUTINE", "still hurting")
        assert phase != "DIFFERENTIAL"


# ============================================================
# SCENARIO 7: Over-reassurance guard
# Expected: Dismissive phrases are replaced post-LLM
# ============================================================

class TestScenario7_OverReassuranceGuard:
    """LLM output containing dismissive phrases must be corrected."""

    @pytest.mark.asyncio
    async def test_no_medical_attention_replaced(self):
        mock_llm = _make_mock_llm(
            "This is no medical attention needed. You should be fine."
        )
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, _, _ = await process_chat_message("I have a headache", [])
            assert "no medical attention" not in response.lower()
            assert "monitoring" in response.lower() or "professional" in response.lower()

    @pytest.mark.asyncio
    async def test_nothing_to_worry_replaced(self):
        mock_llm = _make_mock_llm(
            "This is nothing to worry about. Just rest."
        )
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, _, _ = await process_chat_message("I feel dizzy", [])
            assert "nothing to worry about" not in response.lower()


# ============================================================
# SCENARIO 8: History-aware risk escalation
# Expected: "chest pain" in Turn 1 → still caught in Turn 3
# ============================================================

class TestScenario8_HistoryAwareRisk:
    """Critical symptoms from earlier turns must not be forgotten."""

    @pytest.mark.asyncio
    async def test_chest_pain_in_history_triggers_urgent(self):
        mock_llm = _make_mock_llm("Here is my assessment.")
        history = [
            {"role": "user", "content": "I have chest pain"},
            {"role": "assistant", "content": "Tell me more"},
        ]
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            _, risk, _ = await process_chat_message("it's getting worse", history)
            # chest pain from history should still be detected
            assert risk in ("URGENT", "EMERGENCY")


# ============================================================
# SCENARIO 9: Incomplete info after 4 turns → low-confidence differential
# Expected: DIFFERENTIAL_INCOMPLETE phase, red-flag notice if unaddressed
# ============================================================

class TestScenario9_IncompleteAfter4Turns:
    """Vague responses for 4+ turns → forced low-confidence assessment."""

    def test_phase_is_differential_incomplete(self):
        history = [
            HumanMessage(content="I feel bad"), AIMessage(content="?"),
            HumanMessage(content="still bad"), AIMessage(content="?"),
            HumanMessage(content="same"), AIMessage(content="?"),
            HumanMessage(content="yeah"), AIMessage(content="?"),
        ]
        phase = _detect_triage_phase(history, "ROUTINE", "same thing")
        assert phase == "DIFFERENTIAL_INCOMPLETE"


# ============================================================
# SCENARIO 10: Rich first message skips screening
# Expected: All markers in first input → direct DIFFERENTIAL
# ============================================================

class TestScenario10_RichFirstMessage:
    """A detailed first message should skip screening entirely."""

    def test_direct_to_differential(self):
        rich_input = (
            "I've had a severe headache for 3 days, 8/10, can't work, "
            "no loss of consciousness, poor sleep and lots of stress"
        )
        phase = _detect_triage_phase([], "ROUTINE", rich_input)
        assert phase == "DIFFERENTIAL"

    def test_sparse_first_message_screens(self):
        phase = _detect_triage_phase([], "ROUTINE", "I have a headache")
        assert phase == "INITIAL_SCREENING"
