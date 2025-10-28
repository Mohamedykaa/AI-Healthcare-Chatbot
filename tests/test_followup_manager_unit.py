# ============================================================
# 🔬 Pytest — FollowUpManager Unit Tests (Logic)
# ============================================================
import pytest
from src.chatbot_system.followup_manager import FollowUpManager

@pytest.fixture
def manager():
    """Provides a clean, empty FollowUpManager instance for each test."""
    return FollowUpManager(negative_boost_multiplier=-0.5) # Use a clear negative value for testing

# ------------------------------------------------------------
# Test Answer Normalization
# ------------------------------------------------------------

@pytest.mark.parametrize("input_answer, expected_output", [
    ("yes", "yes"),
    ("y", "yes"),
    ("نعم", "yes"),
    ("ايوه", "yes"),
    ("NO", "no"),
    ("n", "no"),
    ("لا", "no"),
    ("maybe", "partial_yes"),
    ("a bit", "partial_yes"),
    ("ممكن", "partial_yes"),
    ("i don't know", "partial_yes"), # Fallback
])
def test_normalize_answer(manager, input_answer, expected_output):
    """Tests the _normalize_answer method with various inputs."""
    assert manager._normalize_answer(input_answer) == expected_output

# ------------------------------------------------------------
# Test Queue Prioritization Logic
# ------------------------------------------------------------

def test_reorder_queue_logic(manager):
    """
    Tests that _reorder_queue correctly sorts questions based on:
    1. Severity (DESC)
    2. Boost Total (DESC) (Calculated from 'boosts' list)
    3. Sequence/Age (ASC - older first)
    """
    print("\n🔬 Testing queue prioritization (_reorder_queue)...")
    
    questions = [
        # ✅ FIX: Provide the 'boosts' list, not 'boost_total',
        # as add_questions recalculates it.
        {
            "id": "q1_low_sev", "question": "q1", "severity": 1, 
            "boosts": [{"name": "flu", "value": 0.1}] # boost_total = 0.1
        }, # Seq 0
        {
            "id": "q2_high_sev", "question": "q2", "severity": 5,
             "boosts": [{"name": "flu", "value": 0.1}] # boost_total = 0.1
        }, # Seq 1
        {
            "id": "q3_mid_sev", "question": "q3", "severity": 3,
             "boosts": [{"name": "flu", "value": 0.5}] # boost_total = 0.5
        }, # Seq 2
        {
            "id": "q4_high_sev_high_boost", "question": "q4", "severity": 5,
             "boosts": [{"name": "flu", "value": 0.5}] # boost_total = 0.5
        }, # Seq 3
    ]
    
    # Add questions one by one (seq 0, 1, 2, 3)
    for q in questions:
        manager.add_questions([q], reorder=False) # Add without reordering
    
    # Manually trigger the reorder
    manager._reorder_queue()
    
    # Get the ordered list of IDs from the pending questions
    ordered_ids = [q['id'] for q in manager.pending_questions]
    
    # Expected order:
    # 1. q4_high_sev_high_boost (Severity 5, Boost 0.5, Seq 3)
    # 2. q2_high_sev (Severity 5, Boost 0.1, Seq 1)
    # 3. q3_mid_sev (Severity 3, Boost 0.5, Seq 2)
    # 4. q1_low_sev (Severity 1, Boost 0.1, Seq 0)
    expected_order = ["q4_high_sev_high_boost", "q2_high_sev", "q3_mid_sev", "q1_low_sev"]
    
    assert ordered_ids == expected_order
    print("✅ Queue prioritization logic is correct.")

# ------------------------------------------------------------
# Test Answer Recording & Boost Calculation
# ------------------------------------------------------------

@pytest.fixture
def manager_with_question(manager):
    """Provides a manager with one question already added."""
    question = {
        "id": "flu_q1",
        "question": "Do you have fever?",
        "severity": 5,
        "boosts": [
            {"name": "flu", "value": 0.3},
            {"name": "common cold", "value": 0.1}
        ],
        "disease": "flu"
    }
    manager.add_questions([question])
    return manager

def test_record_answer_yes(manager_with_question):
    """Tests that a 'yes' answer applies 100% of the positive boost."""
    print("\n🔬 Testing 'yes' answer boost...")
    manager_with_question.record_answer("flu_q1", "yes")
    boosts = manager_with_question.get_disease_boosts()
    
    assert boosts.get("flu") == pytest.approx(0.3)
    assert boosts.get("common cold") == pytest.approx(0.1)
    print("✅ 'yes' answer applied boosts correctly.")

def test_record_answer_no(manager_with_question):
    """Tests that a 'no' answer applies the negative_boost_multiplier."""
    print("\n🔬 Testing 'no' answer boost...")
    # Manager was initialized with negative_boost_multiplier=-0.5
    manager_with_question.record_answer("flu_q1", "no")
    boosts = manager_with_question.get_disease_boosts()
    
    # flu boost = 0.3 * -0.5 = -0.15
    # cold boost = 0.1 * -0.5 = -0.05
    assert boosts.get("flu") == pytest.approx(-0.15)
    assert boosts.get("common cold") == pytest.approx(-0.05)
    print("✅ 'no' answer applied negative boosts correctly.")

def test_record_answer_partial(manager_with_question):
    """Tests that a 'partial_yes' (maybe) answer applies 50% of the boost."""
    print("\n🔬 Testing 'partial_yes' answer boost...")
    manager_with_question.record_answer("flu_q1", "maybe")
    boosts = manager_with_question.get_disease_boosts()
    
    # flu boost = 0.3 * 0.5 = 0.15
    # cold boost = 0.1 * 0.5 = 0.05
    assert boosts.get("flu") == pytest.approx(0.15)
    assert boosts.get("common cold") == pytest.approx(0.05)
    print("✅ 'partial_yes' answer applied 50% boosts correctly.")