# ============================================================
# 🧠 Pytest — DiagnosisAgent + FollowUpManager Integration Test
# ============================================================
from src.chatbot_system.diagnosis_agent import DiagnosisAgent
from src.chatbot_system.followup_manager import FollowUpManager
import json
import pytest


@pytest.mark.integration
def test_diagnosis_and_followup_integration():
    """Test the interaction between DiagnosisAgent and FollowUpManager."""

    print("\n============================================================")
    print("🚀 Running DiagnosisAgent + FollowUpManager Integration Test")
    print("============================================================")

    # ------------------------------------------------------------
    # 1️⃣ Initialize system components
    # ------------------------------------------------------------
    manager = FollowUpManager()
    agent = DiagnosisAgent()

    assert manager is not None
    assert agent is not None
    print("✅ Components initialized successfully.")

    # ------------------------------------------------------------
    # 2️⃣ Simulate user input
    # ------------------------------------------------------------
    user_input = "I have fever, sore throat, and a slight cough."
    print(f"\n💬 User Input: {user_input}")

    # ------------------------------------------------------------
    # 3️⃣ Perform initial disease prediction
    # ------------------------------------------------------------
    result = agent.predict(user_input, followup_manager=manager, top_k=3)
    assert "predictions" in result
    assert len(result["predictions"]) > 0
    print("✅ Initial prediction completed.")

    # ------------------------------------------------------------
    # 4️⃣ Validate follow-up question generation
    # ------------------------------------------------------------
    has_questions = any(
        bool(pred["follow_up_questions"]) for pred in result["predictions"]
    )
    assert has_questions, "Expected at least one follow-up question."
    print("✅ Follow-up questions generated successfully.")

    # ------------------------------------------------------------
    # 5️⃣ Pop the next question and record an answer
    # ------------------------------------------------------------
    q = manager.get_next_question()
    assert q is not None, "No pending follow-up questions were found."

    print(f"\n🎯 Next Question: {q['text']}")
    manager.record_answer(q["id"], "yes")

    answers = manager.get_all_answers()
    assert q["id"] in answers
    print(f"✅ Answer recorded for question: {q['id']}")

    # ------------------------------------------------------------
    # 6️⃣ Verify disease boosts are stored
    # ------------------------------------------------------------
    boosts = manager.get_disease_boosts()
    assert isinstance(boosts, dict)
    print("✅ Disease boosts verified.")

    # ------------------------------------------------------------
    # 7️⃣ Test save and restore functionality
    # ------------------------------------------------------------
    saved_state = manager.to_json()
    new_manager = FollowUpManager()
    new_manager.from_json(saved_state)

    assert new_manager.get_all_answers() == manager.get_all_answers()
    assert new_manager.get_disease_boosts() == manager.get_disease_boosts()
    print("✅ Save/restore functionality verified.")

    print("\n============================================================")
    print("✅ Integration Test Completed Successfully")
    print("============================================================")
