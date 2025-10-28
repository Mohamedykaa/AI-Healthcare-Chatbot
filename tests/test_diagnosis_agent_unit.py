# ============================================================
# 🔬 Pytest — DiagnosisAgent Unit Tests (Logic)
# ============================================================
import pytest
# Ensure TextCleaner can be imported if DiagnosisAgent needs it during init mock patching
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Attempt import for context, though it might be mocked
try:
    from src.utils.text_cleaner import TextCleaner
except ImportError:
    TextCleaner = None # Define as None if unavailable

from src.chatbot_system.diagnosis_agent import DiagnosisAgent

# 'mocker' is a fixture provided by the 'pytest-mock' plugin
@pytest.fixture
def agent(mocker):
    """
    Fixture to create a DiagnosisAgent instance *without* loading
    any real models or files from disk. We "patch" the loaders
    to do nothing, making tests fast and isolated.
    """
    # Patch the methods that load from disk
    mocker.patch('src.chatbot_system.diagnosis_agent.DiagnosisAgent.load_model', return_value=None)
    mocker.patch('src.chatbot_system.diagnosis_agent.DiagnosisAgent.load_knowledge_base', return_value=None)
    mocker.patch('src.chatbot_system.diagnosis_agent.DiagnosisAgent.load_fallback_dataset', return_value=None)

    # Mock TextCleaner import *if* it failed, to allow agent instantiation
    if TextCleaner is None:
         mocker.patch('src.chatbot_system.diagnosis_agent.TextCleaner', None)
         print("Mocking TextCleaner as None for Agent instantiation")

    # Mock the TextCleaner *instance* creation within the agent's __init__
    mock_cleaner_instance = mocker.MagicMock()
    # Define side effect for cleaner: just return input lowercase/stripped for simplicity in unit tests
    mock_cleaner_instance.transform.side_effect = lambda x: str(x).lower().strip()
    mocker.patch('src.chatbot_system.diagnosis_agent.TextCleaner', return_value=mock_cleaner_instance)


    # Now we can create an instance safely
    agent_instance = DiagnosisAgent()

    # Mock internal components
    agent_instance.label_encoder = mocker.MagicMock()
    # Provide example classes for tests that might need them
    agent_instance.label_encoder.classes_ = ['flu', 'cold', 'migraine', 'allergy'] # Use lowercase
    agent_instance.pipeline = mocker.MagicMock()
    # Ensure the mocked cleaner instance is set correctly
    agent_instance.text_cleaner_instance = mock_cleaner_instance

    return agent_instance

def test_combine_scores_logic(agent):
    """
    Unit test for the _combine_scores method.
    NOTE: This test WILL FAIL until the boost logic in
    DiagnosisAgent._combine_scores is corrected.
    """
    print("\n🔬 Testing _combine_scores logic...")

    ml_scores = {'flu': 0.8, 'cold': 0.2}
    rule_scores = {'cold': 0.5, 'migraine': 0.3}
    csv_scores = {'flu': 0.1}
    boosts = {'flu': 0.1}

    combined = agent._combine_scores(ml_scores, rule_scores, csv_scores, boosts)

    # Expected calculation for 'flu': base=0.49, final=0.59
    assert combined.get('flu', 0.0) == pytest.approx(0.59), "Calculation for 'flu' with boost failed."
    # Expected calculation for 'cold': base=0.27, final=0.27
    assert combined.get('cold', 0.0) == pytest.approx(0.27), "Calculation for 'cold' failed."
    # Expected calculation for 'migraine': base=0.09, final=0.09
    assert combined.get('migraine', 0.0) == pytest.approx(0.09), "Calculation for 'migraine' failed."

    assert all(k == k.lower() for k in combined.keys()), "Combined score keys should be lowercase."

    print("⚠️  NOTE: If this test fails, check DiagnosisAgent._combine_scores boost logic.")
    # print("✅ _combine_scores logic is correct.") # Keep commented

def test_combine_scores_capping_at_one(agent):
    """
    Tests that the combined score is capped at 1.0.
    """
    print("\n🔬 Testing _combine_scores capping at 1.0...")
    ml_scores = {'flu': 1.0}
    rule_scores = {'flu': 1.0}
    csv_scores = {'flu': 0.0} # Provide empty or zero scores for missing sources
    boosts = {'flu': 0.5}

    # ✅ FIX: Use the correct argument name 'followup_boosts' and pass csv_scores
    combined = agent._combine_scores(ml_scores, rule_scores, csv_scores=csv_scores, followup_boosts=boosts)

    # Expected: base=0.9, final=1.4 -> capped at 1.0
    assert combined.get('flu', 0.0) == 1.0, "Score should be capped at 1.0."
    print("✅ Score capping at 1.0 works.")


def test_predict_with_mocked_scorers(agent, mocker):
    """
    Tests the main predict() function logic (sorting, formatting, follow-ups)
    by mocking the internal scoring methods.
    """
    print("\n🔬 Testing predict() function logic with mocked scorers...")

    # 1. Setup: Define mock outputs (use lowercase keys consistently)
    mocker.patch.object(agent, '_ml_scores', return_value={'flu': 0.9, 'cold': 0.1})
    mocker.patch.object(agent, '_rule_match_scores', return_value={'cold': 0.7, 'allergy': 0.5})
    mocker.patch.object(agent, '_csv_fallback_scores', return_value={}) # Empty CSV scores

    # Mock the follow-up retrieval
    mock_followups = [{"id": "q_flu_1", "text": "Do you have a runny nose?", "severity": 2}]
    mocker.patch.object(agent, '_get_followups_for_disease', return_value=mock_followups)

    # Mock the FollowUpManager instance passed to predict
    mock_manager = mocker.MagicMock()
    mock_manager.get_disease_boosts.return_value = {} # No boosts initially

    # 2. Act: Call the predict function
    result = agent.predict("I have fever and cough", followup_manager=mock_manager, top_k=3)
    preds = result.get("predictions", [])

    # 3. Assert: Check the results

    # Expected scores after _combine_scores (assuming boost logic is eventually fixed):
    # flu:     0.54
    # cold:    0.27
    # allergy: 0.15

    assert len(preds) == 3, f"Expected 3 predictions, got {len(preds)}"

    # ✅ FIX: Check order and values expecting lowercase disease names
    assert preds[0]["disease"] == "flu", f"Expected first disease 'flu', got {preds[0]['disease']}"
    assert preds[0]["probability"] == pytest.approx(0.54)

    assert preds[1]["disease"] == "cold", f"Expected second disease 'cold', got {preds[1]['disease']}"
    assert preds[1]["probability"] == pytest.approx(0.27)

    assert preds[2]["disease"] == "allergy", f"Expected third disease 'allergy', got {preds[2]['disease']}"
    assert preds[2]["probability"] == pytest.approx(0.15)

    # Check that follow-ups were retrieved and passed to the manager
    agent._get_followups_for_disease.assert_called()
    mock_manager.add_questions.assert_called()

    # Check that follow-ups are included in the output for the corresponding disease
    assert "follow_up_questions" in preds[0]
    assert len(preds[0]["follow_up_questions"]) > 0
    assert preds[0]["follow_up_questions"][0]["id"] == "q_flu_1"
    assert "follow_up_questions" in preds[1]
    assert "follow_up_questions" in preds[2]

    print("✅ predict() function logic (sorting, formatting, follow-up handling) seems correct.")