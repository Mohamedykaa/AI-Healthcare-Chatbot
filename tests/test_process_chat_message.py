"""
Process Chat Message Integration Tests
=========================================

Tests for the main process_chat_message() pipeline with mocked
LLM and vectorstore — verifying prompt selection, error handling,
risk-level behaviour, and context flow without real model calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.logic import (
    process_chat_message,
    get_safe_fallback_answer,
)


# ============================================================
# Helpers
# ============================================================

def _make_fake_doc(content="Fever is a common symptom.", source="data/medical.txt"):
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source}
    return doc


def _make_mock_llm(response_text="Based on your symptoms, this could be a viral infection."):
    mock = AsyncMock()
    result = MagicMock()
    result.content = response_text
    mock.ainvoke.return_value = result
    return mock


def _make_mock_vectorstore(docs=None):
    if docs is None:
        docs = [_make_fake_doc()]
    mock = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    mock.as_retriever.return_value = retriever
    return mock


# ============================================================
# Emergency bypass
# ============================================================

class TestEmergencyBypass:
    """Emergency input must bypass LLM entirely."""

    @pytest.mark.asyncio
    async def test_emergency_input_bypasses_llm(self):
        with patch("src.core.logic._LLM", _make_mock_llm()) as mock_llm, \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, risk, sources = await process_chat_message(
                "I'm having a heart attack", []
            )
            assert risk == "EMERGENCY"
            assert "EMERGENCY ALERT" in response
            # LLM should never be called
            mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_emergency_arabic_input_bypasses_llm(self):
        with patch("src.core.logic._LLM", _make_mock_llm()) as mock_llm, \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, risk, sources = await process_chat_message(
                "أشعر بأفكار انتحارية", []
            )
            assert risk == "EMERGENCY"
            assert "تنبيه طارئ" in response
            mock_llm.ainvoke.assert_not_called()


# ============================================================
# Prompt selection
# ============================================================

class TestPromptSelection:
    """First turn vs follow-up must use correct prompt template."""

    @pytest.mark.asyncio
    async def test_first_turn_uses_initial_screening(self):
        mock_llm = _make_mock_llm()
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            await process_chat_message("I have a headache", [])
            messages = mock_llm.ainvoke.call_args[0][0]
            sys_msg = messages[0].content.lower()
            # Initial screening must mention red flags
            assert "red flag" in sys_msg or "loss of consciousness" in sys_msg

    @pytest.mark.asyncio
    async def test_follow_up_uses_characterization(self):
        mock_llm = _make_mock_llm()
        history = [
            {"role": "user", "content": "I have a headache"},
            {"role": "assistant", "content": "How long have you had it?"},
        ]
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            await process_chat_message("About 3 days", history)
            messages = mock_llm.ainvoke.call_args[0][0]
            assert len(messages) == 4  # System, History User, History Assistant, Latest User
            sys_msg = messages[0].content.lower()
            # Characterization must mention missing info
            assert "missing" in sys_msg or "severity" in sys_msg


# ============================================================
# LLM failure handling
# ============================================================

class TestLLMFailureHandling:
    """LLM failures must return safe fallback, not crash."""

    @pytest.mark.asyncio
    async def test_connection_error_returns_fallback(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = ConnectionError("Ollama down")
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, risk, sources = await process_chat_message("I have a fever", [])
            assert "fallback safe-mode" in response
            assert risk == "ROUTINE"

    @pytest.mark.asyncio
    async def test_generic_exception_returns_fallback(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = RuntimeError("Model crashed")
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, risk, sources = await process_chat_message("I have a cough", [])
            assert "fallback safe-mode" in response


# ============================================================
# Urgent prefix
# ============================================================

class TestUrgentPrefix:
    """URGENT risk level must prepend the urgent warning prefix."""

    @pytest.mark.asyncio
    async def test_urgent_input_adds_prefix(self):
        mock_llm = _make_mock_llm()
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, risk, sources = await process_chat_message(
                "I have severe chest pain", []
            )
            assert risk == "URGENT"
            assert "URGENT ADVICE REQUIRED" in response


# ============================================================
# Echo guardrail
# ============================================================

class TestEchoGuardrail:
    """Degenerate LLM responses must trigger safe fallback."""

    @pytest.mark.asyncio
    async def test_echo_prompt_triggers_fallback(self):
        mock_llm = _make_mock_llm("You are an educational medical symptom checker.")
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, risk, _ = await process_chat_message("I have a cold", [])
            assert "fallback safe-mode" in response

    @pytest.mark.asyncio
    async def test_short_response_triggers_fallback(self):
        mock_llm = _make_mock_llm("OK")
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, risk, _ = await process_chat_message("I feel dizzy", [])
            assert "fallback safe-mode" in response

    @pytest.mark.asyncio
    async def test_empty_response_triggers_fallback(self):
        mock_llm = _make_mock_llm("")
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, risk, _ = await process_chat_message("I feel weak", [])
            assert "fallback safe-mode" in response


# ============================================================
# Source formatting integration
# ============================================================

class TestSourcesInResponse:
    """Sources must be formatted and returned from the pipeline."""

    @pytest.mark.asyncio
    async def test_sources_returned_for_routine_query(self):
        mock_llm = _make_mock_llm()
        docs = [_make_fake_doc(source="data/medquad.txt")]
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore(docs)):
            _, _, sources = await process_chat_message("I have a headache", [])
            assert "medquad.txt" in sources

    @pytest.mark.asyncio
    async def test_no_sources_for_emergency(self):
        with patch("src.core.logic._LLM", _make_mock_llm()), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            _, _, sources = await process_chat_message("heart attack", [])
            assert sources == ""


# ============================================================
# Language guardrail
# ============================================================

class TestLanguageGuardrail:
    """'the patient' must be replaced with 'you' in the final response."""

    @pytest.mark.asyncio
    async def test_patient_replaced_with_you(self):
        mock_llm = _make_mock_llm(
            "The patient should rest. Also the patient needs fluids."
        )
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, _, _ = await process_chat_message("I feel tired", [])
            assert "the patient" not in response
            assert "You should rest" in response


# ============================================================
# History-aware retrieval
# ============================================================

class TestHistoryAwareRetrieval:
    """Retrieval query must include symptom context from history."""

    @pytest.mark.asyncio
    async def test_retrieval_uses_composite_query(self):
        """When history exists, retriever.invoke() must receive more than just user_input."""
        mock_llm = _make_mock_llm()
        mock_vs = _make_mock_vectorstore()
        history = [
            {"role": "user", "content": "I have headaches and dizziness"},
            {"role": "assistant", "content": "How long have you had these?"},
        ]
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", mock_vs):
            await process_chat_message("for 3 days", history)
            retriever = mock_vs.as_retriever.return_value
            query_used = retriever.invoke.call_args[0][0]
            # Query must contain symptom context, not just "for 3 days"
            assert "headache" in query_used.lower() or "dizziness" in query_used.lower()

    @pytest.mark.asyncio
    async def test_first_turn_uses_raw_input(self):
        """On first turn with no history, retriever should use the raw input."""
        mock_llm = _make_mock_llm()
        mock_vs = _make_mock_vectorstore()
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", mock_vs):
            await process_chat_message("I have a headache", [])
            retriever = mock_vs.as_retriever.return_value
            query_used = retriever.invoke.call_args[0][0]
            assert query_used == "I have a headache"

    @pytest.mark.asyncio
    async def test_late_turn_retains_original_symptom_anchor(self):
        """After many turns, retrieval query must still include the first symptom message."""
        mock_llm = _make_mock_llm()
        mock_vs = _make_mock_vectorstore()
        history = [
            {"role": "user", "content": "I have headaches and dizziness"},
            {"role": "assistant", "content": "How long?"},
            {"role": "user", "content": "about 2 weeks"},
            {"role": "assistant", "content": "How severe?"},
            {"role": "user", "content": "7 out of 10"},
            {"role": "assistant", "content": "Any red flags?"},
            {"role": "user", "content": "no fainting"},
            {"role": "assistant", "content": "Sleep and stress?"},
        ]
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", mock_vs):
            await process_chat_message("poor sleep", history)
            retriever = mock_vs.as_retriever.return_value
            query_used = retriever.invoke.call_args[0][0]
            # Original symptom anchor must still be present
            assert "headache" in query_used.lower()


# ============================================================
# Anti-premature-closure
# ============================================================

class TestAntiPrematureClosure:
    """Fallback must not suggest specific conditions."""

    def test_fallback_does_not_diagnose(self):
        answer = get_safe_fallback_answer("I have a headache")
        assert "viral" not in answer.lower()
        assert "respiratory" not in answer.lower()
        assert "infection" not in answer.lower()

    def test_fallback_asks_for_more_info(self):
        answer = get_safe_fallback_answer("I feel dizzy")
        assert "more information" in answer.lower() or "how long" in answer.lower()


# ============================================================
# Deterministic red-flag notice
# ============================================================

class TestRedFlagNotice:
    """When DIFFERENTIAL_INCOMPLETE fires without red-flag status,
    a deterministic notice must be injected regardless of LLM output."""

    @pytest.mark.asyncio
    async def test_red_flag_notice_injected_after_4_turns_no_red_flags(self):
        mock_llm = _make_mock_llm("Here is my assessment without asking about red flags.")
        history = [
            {"role": "user", "content": "I feel bad"},
            {"role": "assistant", "content": "Tell me more"},
            {"role": "user", "content": "still bad"},
            {"role": "assistant", "content": "How long?"},
            {"role": "user", "content": "a while"},
            {"role": "assistant", "content": "Anything else?"},
            {"role": "user", "content": "no"},
            {"role": "assistant", "content": "ok"},
        ]
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, _, _ = await process_chat_message("still the same", history)
            # The deterministic notice must be present regardless of LLM output
            assert "red-flag" in response.lower() or "red flag" in response.lower()
            assert "loss of consciousness" in response.lower()

    @pytest.mark.asyncio
    async def test_no_red_flag_notice_when_flags_addressed(self):
        mock_llm = _make_mock_llm("Here is a proper assessment.")
        history = [
            {"role": "user", "content": "headache for 2 weeks, severe"},
            {"role": "assistant", "content": "Any red flags?"},
            {"role": "user", "content": "no loss of consciousness, no vision problems"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "poor sleep and stress"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "can't work"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "still hurting"},
            {"role": "assistant", "content": "ok"},
        ]
        with patch("src.core.logic._LLM", mock_llm), \
             patch("src.core.logic._VECTORSTORE", _make_mock_vectorstore()):
            response, _, _ = await process_chat_message("what is it?", history)
            # Red-flag notice should NOT appear because red flags were addressed
            assert "\ud83d\udea9" not in response
