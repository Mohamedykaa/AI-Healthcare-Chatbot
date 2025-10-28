# ============================================================
# 🧠 Pytest — Chatbot System Integration Test (Automated)
# ============================================================
import sys
from pathlib import Path
import json

# Ensure repo root is on path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.chatbot_system.diagnosis_agent import DiagnosisAgent
from src.chatbot_system.followup_manager import FollowUpManager
from src.chatbot_system.recommendation_agent import RecommendationAgent


def test_chatbot_full_integration():
    """
    ✅ Automated integration test for:
        - DiagnosisAgent
        - FollowUpManager
        - RecommendationAgent
    """

    # ------------------------------------------------------------
    # 1️⃣ Initialize all components
    # ------------------------------------------------------------
    diag_agent = DiagnosisAgent()
    rec_agent = RecommendationAgent()
    followup = FollowUpManager()

    assert diag_agent is not None, "DiagnosisAgent not initialized."
    assert rec_agent is not None, "RecommendationAgent not initialized."
    assert followup is not None, "FollowUpManager not initialized."

    # ------------------------------------------------------------
    # 2️⃣ Simulate user input
    # ------------------------------------------------------------
    user_input = "I have a sore throat and a slight fever."
    result = diag_agent.predict(user_input, followup)

    assert isinstance(result, dict), "DiagnosisAgent.predict() must return a dict."
    assert "predictions" in result, "Missing 'predictions' key in result."
    assert len(result["predictions"]) > 0, "No predictions returned."

    # ------------------------------------------------------------
    # 3️⃣ Validate prediction structure
    # ------------------------------------------------------------
    for pred in result["predictions"]:
        assert "disease" in pred, "Prediction missing 'disease' key."
        assert "probability" in pred, "Prediction missing 'probability' key."
        assert isinstance(pred["probability"], (float, int))

    # ------------------------------------------------------------
    # 4️⃣ Handle follow-up questions
    # ------------------------------------------------------------
    pending = getattr(followup, "queue", getattr(followup, "pending_questions", []))
    assert isinstance(pending, list), "Follow-up questions list not found."

    # Simulate yes/no answers for first two questions (if exist)
    for q in pending[:2]:
        followup.record_answer(q["id"], "yes")

    # ------------------------------------------------------------
    # 5️⃣ Re-run diagnosis with updated answers
    # ------------------------------------------------------------
    updated = diag_agent.predict(user_input, followup)
    assert "predictions" in updated, "Updated predictions missing."
    assert len(updated["predictions"]) > 0, "No updated predictions returned."

    # ------------------------------------------------------------
    # 6️⃣ Validate recommendation agent
    # ------------------------------------------------------------
    top_disease = updated["predictions"][0]["disease"]
    details = rec_agent.get_details(top_disease)

    assert isinstance(details, dict), "Recommendation details must be a dict."
    assert "description" in details, "Missing disease description."
    assert "precautions" in details, "Missing precautions list."

    # ------------------------------------------------------------
    # 7️⃣ Test save & restore of follow-up manager
    # ------------------------------------------------------------
    saved_state = followup.to_json()
    restored = FollowUpManager()
    restored.from_json(saved_state)

    assert followup.get_all_answers() == restored.get_all_answers(), "Answers mismatch after restore."
    assert followup.get_disease_boosts() == restored.get_disease_boosts(), "Disease boosts mismatch after restore."

    # ------------------------------------------------------------
    # ✅ Print summary
    # ------------------------------------------------------------
    print("\n--- Chatbot Integration Test Summary ---")
    print(f"User input: {user_input}")
    print("Top predicted diseases:")
    for p in updated["predictions"]:
        print(f"  - {p['disease']}: {p['probability']:.2%}")
    print(f"\nTop disease details: {top_disease}")
    print(json.dumps(details, indent=2))
    print("\n✅ Chatbot integration test passed successfully.")


# Optional: Allow manual run for debugging
if __name__ == "__main__":
    test_chatbot_full_integration()
