# ============================================================
# 🧠 Pytest — FollowUpManager Integration Test (Enhanced)
# ============================================================
import json
import pytest
from src.chatbot_system.followup_manager import FollowUpManager


@pytest.mark.integration
def test_followup_manager_integration():
    """✅ Test the FollowUpManager end-to-end functionality with strong assertions."""

    print("\n============================================================")
    print("🔍 Testing FollowUpManager (disease-scoped follow-ups)")
    print("============================================================")

    # ------------------------------------------------------------
    # 1️⃣ Create FollowUpManager instance
    # ------------------------------------------------------------
    manager = FollowUpManager(negative_boost_multiplier=-0.5)
    assert manager is not None, "❌ FollowUpManager failed to initialize."
    print("✅ FollowUpManager initialized successfully.")

    # ------------------------------------------------------------
    # 2️⃣ Add multiple questions from different diseases
    # ------------------------------------------------------------
    
    # ✅ FIX: Assign disease *inside* each question dict
    # This correctly tests the multi-disease logic.
    questions = [
        {
            "id": "flu_q1",
            "question": "Do you have fever?",
            "severity": 5,
            "boosts": [{"name": "flu", "value": 0.3}],
            "disease": "flu" # <-- Added disease
        },
        {
            "id": "flu_q2",
            "question": "Do you have a sore throat?",
            "severity": 4,
            "boosts": [{"name": "flu", "value": 0.2}],
            "disease": "flu" # <-- Added disease
        },
        {
            "id": "covid_q1",
            "question": "Do you have loss of smell?",
            "severity": 5, # Same as flu_q1
            "boosts": [{"name": "covid-19", "value": 0.4}],
            "disease": "covid-19" # <-- Added disease
        },
    ]

    # ✅ FIX: Remove disease_scope, as it's now in the questions
    added_count = manager.add_questions(questions) 
    
    # ✅ STRENGTHENED: Check exact count
    assert added_count == 3, "❌ Failed to add all 3 questions."
    print("✅ Added 3 follow-up questions for 'flu' and 'covid-19'.")

    # ------------------------------------------------------------
    # 3️⃣ Check pending queue
    # ------------------------------------------------------------
    queue = getattr(manager, "pending_questions", [])
    assert isinstance(queue, list), "❌ Queue structure is invalid."
    
    # ✅ STRENGTHENED: Check exact count
    assert len(queue) == 3, "❌ Queue length is not 3."
    print(f"📋 Pending questions count: {len(queue)}")

    # ------------------------------------------------------------
    # 4️⃣ Pop next question and record answer
    # ------------------------------------------------------------
    q = manager.get_next_question()
    assert q is not None, "❌ No question retrieved from queue."
    
    # ✅ STRENGTHENED: Check prioritization logic.
    # 'flu_q1' (seq 0) and 'covid_q1' (seq 2) both have severity 5.
    # 'flu_q1' has boost_total 0.3, 'covid_q1' has 0.4.
    # The sort key is (severity, boost_total, -seq).
    # (5, 0.4, -2) vs (5, 0.3, -0)
    # 'covid_q1' should be first.
    assert q['id'] == "covid_q1", f"❌ Wrong question popped! Expected 'covid_q1' based on severity/boost, got {q['id']}"
    print(f"\n🎯 Next Question (Correctly prioritized): {q['question']}")

    manager.record_answer(q["id"], "yes")
    answers = manager.get_all_answers()
    assert q["id"] in answers, "❌ Answer not recorded correctly."
    print(f"✅ Recorded 'yes' answer for: {q['id']}")

    # ------------------------------------------------------------
    # 5️⃣ Verify disease boosts
    # ------------------------------------------------------------
    boosts = manager.get_disease_boosts()
    assert isinstance(boosts, dict), "❌ Boosts should be a dictionary."
    
    # ✅ STRENGTHENED: Check the *value* of the boost.
    assert "covid-19" in boosts, "❌ 'covid-19' boost not recorded."
    assert boosts["covid-19"] == pytest.approx(0.4), "❌ 'covid-19' boost value is incorrect."
    assert "flu" not in boosts, "❌ 'flu' boost should not be present yet."
    print("⚡ Disease Boosts:", json.dumps(boosts, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------
    # 6️⃣ Save and restore state
    # ------------------------------------------------------------
    json_state = manager.to_json()
    assert isinstance(json_state, str) and len(json_state) > 0, "❌ Saved state is empty."
    print(f"💾 Saved state length: {len(json_state)}")

    new_manager = FollowUpManager()
    new_manager.from_json(json_state)
    print("♻️ State reloaded successfully.")

    # Compare restored data
    assert new_manager.get_all_answers() == manager.get_all_answers(), "❌ Answers mismatch after restore."
    assert new_manager.get_disease_boosts() == manager.get_disease_boosts(), "❌ Boosts mismatch after restore."
    
    # ✅ STRENGTHENED: Check queue length after restore
    new_queue = getattr(new_manager, "pending_questions", [])
    assert len(new_queue) == 2, "❌ Pending queue mismatch after restore."
    print(f"✅ State restoration verified successfully (pending: {len(new_queue)}, answers: 1).")

    print("\n============================================================")
    print("✅ FollowUpManager Integration Test Completed Successfully")
    print("============================================================")


# Allow manual run for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) # Run with pytest