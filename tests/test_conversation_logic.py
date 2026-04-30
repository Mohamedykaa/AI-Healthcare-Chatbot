"""
Conversation Logic Tests
=========================

Tests for deterministic decision logic extracted from on_message():
- Turn count calculation
- Phase decision (first turn vs follow-up)
- Echo guardrail conditions
- Chat history truncation
- History window formatting

These test the LOGIC, not the LLM or Chainlit.
"""

import pytest


# ============================================================
# Reproduce deterministic logic from app.py
# ============================================================

def compute_turn_count(chat_history: list) -> int:
    """Turn count is number of complete exchanges (app.py line 504)."""
    return len(chat_history) // 2


def is_first_turn(chat_history: list) -> bool:
    """First turn when no prior exchanges exist (app.py line 507)."""
    return compute_turn_count(chat_history) == 0


def is_echo_detected(answer: str) -> bool:
    """Echo guardrail logic (app.py line 545).

    Returns True if the answer should be replaced with fallback.
    """
    if not answer:
        return True
    if len(answer.strip()) < 10:
        return True
    if answer.strip().startswith("You are"):
        return True
    if "You match symptoms" in answer:
        return True
    return False


def truncate_history(chat_history: list) -> list:
    """History truncation (app.py lines 571-572)."""
    if len(chat_history) > 16:
        return chat_history[-16:]
    return chat_history


def format_history_window(chat_history: list, human_cls=None, window: int = 8) -> str:
    """History text formatting (app.py lines 496-501).

    Formats the last `window` messages into a string.
    Uses duck-typing: checks isinstance for human_cls if provided.
    """
    history_text = ""
    if chat_history:
        for msg_item in chat_history[-window:]:
            if hasattr(msg_item, 'content'):
                if human_cls and isinstance(msg_item, human_cls):
                    role = "User"
                else:
                    role = "Assistant"
                history_text += f"{role}: {msg_item.content}\n"
    return history_text


# ===========================================================================
# TURN COUNT
# ===========================================================================

class TestTurnCount:
    """Turn count = len(history) // 2."""

    @pytest.mark.parametrize("history_len, expected", [
        (0, 0),   # no messages
        (1, 0),   # one message (incomplete exchange)
        (2, 1),   # one full exchange
        (3, 1),   # one full + one pending
        (4, 2),   # two full exchanges
        (8, 4),   # four full exchanges
        (16, 8),  # eight full exchanges (max window)
    ])
    def test_turn_count(self, history_len, expected):
        history = ["msg"] * history_len
        assert compute_turn_count(history) == expected


# ===========================================================================
# PHASE DECISION
# ===========================================================================

class TestTriagePhaseDetection:
    """Phase detection imported from production — no logic duplication."""

    def test_empty_history_routine_is_initial_screening(self):
        from src.core.logic import _detect_triage_phase
        assert _detect_triage_phase([], "ROUTINE") == "INITIAL_SCREENING"

    def test_one_exchange_insufficient_is_characterization(self):
        from src.core.logic import _detect_triage_phase
        from langchain_core.messages import HumanMessage, AIMessage
        history = [HumanMessage(content="headache"), AIMessage(content="How long?")]
        assert _detect_triage_phase(history, "ROUTINE") == "CHARACTERIZATION"

    def test_sufficient_info_early_allows_differential(self):
        from src.core.logic import _detect_triage_phase
        from langchain_core.messages import HumanMessage, AIMessage
        history = [
            HumanMessage(content="headache for 2 weeks, 8/10 severity"),
            AIMessage(content="Any red flags?"),
            HumanMessage(content="No loss of consciousness, poor sleep and stress"),
            AIMessage(content="..."),
        ]
        assert _detect_triage_phase(history, "ROUTINE") == "DIFFERENTIAL"

    def test_insufficient_info_late_stays_characterization(self):
        from src.core.logic import _detect_triage_phase
        from langchain_core.messages import HumanMessage, AIMessage
        history = [
            HumanMessage(content="I feel bad"),
            AIMessage(content="Can you tell me more?"),
            HumanMessage(content="just bad"),
            AIMessage(content="How long?"),
        ]
        assert _detect_triage_phase(history, "ROUTINE") == "CHARACTERIZATION"

    def test_four_turns_insufficient_forces_differential_incomplete(self):
        from src.core.logic import _detect_triage_phase
        from langchain_core.messages import HumanMessage, AIMessage
        history = [
            HumanMessage(content="bad"), AIMessage(content="?"),
            HumanMessage(content="bad"), AIMessage(content="?"),
            HumanMessage(content="bad"), AIMessage(content="?"),
            HumanMessage(content="bad"), AIMessage(content="?"),
        ]
        assert _detect_triage_phase(history, "ROUTINE") == "DIFFERENTIAL_INCOMPLETE"

    def test_urgent_always_overrides_phase(self):
        from src.core.logic import _detect_triage_phase
        assert _detect_triage_phase([], "URGENT") == "URGENT_ASSESSMENT"

    def test_urgent_overrides_even_with_history(self):
        from src.core.logic import _detect_triage_phase
        from langchain_core.messages import HumanMessage, AIMessage
        history = [HumanMessage(content="q"), AIMessage(content="a")] * 5
        assert _detect_triage_phase(history, "URGENT") == "URGENT_ASSESSMENT"

    def test_first_turn_rich_input_skips_screening(self):
        """A first message with all markers met should go straight to DIFFERENTIAL."""
        from src.core.logic import _detect_triage_phase
        rich_input = (
            "Severe headache for 2 weeks, 8/10, can't work, "
            "no loss of consciousness, poor sleep and stress"
        )
        assert _detect_triage_phase([], "ROUTINE", rich_input) == "DIFFERENTIAL"


# ===========================================================================
# ECHO GUARDRAIL
# ===========================================================================

class TestEchoGuardrail:
    """Echo detection must catch degenerate LLM outputs."""

    # --- Must trigger (echo detected) ---

    def test_empty_string_triggers(self):
        assert is_echo_detected("") is True

    def test_none_triggers(self):
        assert is_echo_detected(None) is True

    def test_short_answer_triggers(self):
        assert is_echo_detected("OK") is True

    def test_very_short_answer_triggers(self):
        assert is_echo_detected("   Hi    ") is True  # strip → 2 chars

    def test_nine_char_answer_triggers(self):
        assert is_echo_detected("123456789") is True  # exactly 9 < 10

    def test_starts_with_you_are_triggers(self):
        assert is_echo_detected("You are an educational medical symptom checker.") is True

    def test_starts_with_you_are_whitespace(self):
        assert is_echo_detected("   You are repeating the prompt") is True  # strip first

    def test_contains_you_match_symptoms(self):
        assert is_echo_detected("Some text You match symptoms here") is True

    # --- Must NOT trigger (valid answers) ---

    def test_valid_medical_response(self):
        answer = "Based on your description, these symptoms are commonly associated with viral infections."
        assert is_echo_detected(answer) is False

    def test_exactly_10_chars(self):
        assert is_echo_detected("1234567890") is False  # exactly 10, not < 10

    def test_long_valid_answer(self):
        answer = "I understand you're experiencing headaches. " * 5
        assert is_echo_detected(answer) is False

    def test_contains_you_are_not_at_start(self):
        answer = "I think You are experiencing a common cold."
        assert is_echo_detected(answer) is False  # "You are" not at start


# ===========================================================================
# HISTORY TRUNCATION
# ===========================================================================

class TestHistoryTruncation:
    """Chat history must be capped at 16 items."""

    def test_empty_history_unchanged(self):
        assert truncate_history([]) == []

    def test_under_limit_unchanged(self):
        history = list(range(10))
        assert truncate_history(history) == list(range(10))

    def test_exactly_16_unchanged(self):
        history = list(range(16))
        assert truncate_history(history) == list(range(16))

    def test_17_items_truncated_to_last_16(self):
        history = list(range(17))
        result = truncate_history(history)
        assert len(result) == 16
        assert result == list(range(1, 17))

    def test_20_items_truncated_to_last_16(self):
        history = list(range(20))
        result = truncate_history(history)
        assert len(result) == 16
        assert result[0] == 4
        assert result[-1] == 19

    def test_truncation_keeps_most_recent(self):
        """Most recent messages must be preserved, not oldest."""
        history = [f"msg_{i}" for i in range(30)]
        result = truncate_history(history)
        assert result[-1] == "msg_29"
        assert result[0] == "msg_14"


# ===========================================================================
# HISTORY WINDOW FORMATTING
# ===========================================================================

class TestHistoryWindowFormatting:
    """The last 8 messages are formatted into a text string."""

    def test_empty_history_returns_empty(self):
        assert format_history_window([]) == ""

    def test_formats_messages_with_role_labels(self, build_chat_history):
        history = build_chat_history(2)  # 4 messages
        result = format_history_window(history)
        assert "User question 1" in result or "Assistant" in result
        # All messages have content attribute → should all appear
        assert result.count("\n") == 4  # 4 messages, 4 newlines

    def test_window_limits_to_last_8(self, build_chat_history):
        history = build_chat_history(10)  # 20 messages
        result = format_history_window(history, window=8)
        # Should only contain last 8 messages (exchanges 7-10)
        lines = [l for l in result.strip().split("\n") if l]
        assert len(lines) == 8

    def test_objects_without_content_skipped(self):
        """Objects lacking .content attribute are silently skipped."""
        history = [42, "string", None]  # no .content
        result = format_history_window(history)
        assert result == ""
