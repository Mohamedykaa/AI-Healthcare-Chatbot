# tests/test_recommendation_agent.py
import pytest
import pandas as pd
from src.chatbot_system.recommendation_agent import RecommendationAgent
from difflib import get_close_matches # Import for fuzzy match testing logic if needed, although the agent uses it internally

# Fixture remains the same as the previous correct version
@pytest.fixture
def agent(mocker):
    """
    Provides a RecommendationAgent instance with mocked DataFrames
    in the expected WIDE format for precautions.
    """
    # --- Mock Precautions DataFrame (WIDE Format) ---
    prec_data_wide = {
        'disease': ['Flu', 'Common Cold', 'COVID-19', 'flu'],
        'precaution_1': ['Rest', 'Wash hands', 'Isolate', 'Drink fluids'],
        'precaution_2': ['Drink fluids', None, 'Wear mask', 'Rest'],
        'precaution_3': [None, None, None, None]
    }
    mock_prec_df_wide = pd.DataFrame(prec_data_wide)

    # --- Mock Descriptions DataFrame (Long Format - Correct) ---
    desc_data = {
        'disease': ['Flu', 'Common Cold', 'COVID-19', 'Diabetes'], # Ensure 'Diabetes' is here for fuzzy test
        'description': ['Viral illness affecting respiratory system.',
                        'Mild viral infection.',
                        'Viral disease caused by SARS-CoV-2.',
                        'Metabolic disease involving blood sugar.'] # Added specific description
    }
    mock_desc_df = pd.DataFrame(desc_data)

    # Mock pd.read_csv calls using side_effect
    def mock_read_csv(*args, **kwargs):
        file_path = str(args[0])
        if 'precaution' in file_path:
            return mock_prec_df_wide.copy()
        elif 'Description' in file_path:
            return mock_desc_df.copy()
        else:
            raise FileNotFoundError(f"Mock: File not found {file_path}")

    mocker.patch('pandas.read_csv', side_effect=mock_read_csv)
    mocker.patch('builtins.print') # Suppress print statements during test

    # Initialize agent with fuzzy matching enabled (default cutoff is usually fine)
    agent_instance = RecommendationAgent() # Can pass fuzzy_match_cutoff=0.7 if needed

    # Basic verification
    assert 'flu' in agent_instance.precaution_map
    assert 'diabetes' in agent_instance.description_map # Verify diabetes loaded

    return agent_instance

# --- Test Precaution Retrieval ---
def test_get_precautions_found(agent):
    precautions = agent._get_precautions("flu")
    assert precautions == ['Drink fluids', 'Rest']

def test_get_precautions_found_case_variation(agent):
    precautions = agent.get_precautions("Common Cold")
    assert precautions == ['Wash hands']

def test_get_precautions_not_found(agent):
    precautions = agent.get_precautions("Unknown Disease")
    assert precautions == []

# --- Test Description Retrieval ---
def test_get_description_found(agent):
    description = agent._get_description("flu")
    assert description == 'Viral illness affecting respiratory system.'

def test_get_description_not_found(agent):
    # Use public method which standardizes name first
    description = agent.get_details("Unknown Disease")['description']
    assert description == "No detailed description is available for this condition." # Check default message

# --- Test Get Details (Combined Output) ---
def test_get_details_found(agent):
    details = agent.get_details("Flu")
    assert isinstance(details, dict)
    assert details["precautions"] == ['Drink fluids', 'Rest']
    assert details["description"] == 'Viral illness affecting respiratory system.'
    assert "tests" in details
    assert "advice" in details

def test_get_details_not_found(agent):
    details = agent.get_details("Unknown Disease XYZ") # Use a clearly unknown name
    assert isinstance(details, dict)
    assert "No specific precautions found" in details["precautions"][0]
    assert "No detailed description is available" in details["description"]

# --- Test Alias ---
def test_get_recommendations_alias(agent):
    details = agent.get_details("Flu")
    recommendations = agent.get_recommendations("Flu")
    assert details == recommendations

# --- Test Internal Map Key Normalization ---
def test_internal_maps_normalization(agent):
    """Ensures internal maps use lowercase standardized disease keys."""
    # print("\n🔬 Testing internal map key normalization...") # Keep commented out unless debugging
    for key in agent.precaution_map.keys():
        assert key == agent._standardize_name(key), f"Bad key in precaution_map: '{key}'"
    # print("  ✅ Keys in precaution_map look standardized.")
    for key in agent.description_map.keys():
        assert key == agent._standardize_name(key), f"Bad key in description_map: '{key}'"
    # print("  ✅ Keys in description_map look standardized.")


# --- ✅ Added Test for Fuzzy Matching Fallback ---
def test_get_details_fuzzy_match(agent):
    """Tests that fuzzy matching works for typos."""
    print("\n🔬 Testing fuzzy match fallback...")
    # "diabates" is a typo for "diabetes" which exists in the description mock data
    # but not in the precaution mock data.
    details_typo = agent.get_details("diabates")

    assert isinstance(details_typo, dict)
    # Description should be found via fuzzy match
    assert details_typo["description"] == 'Metabolic disease involving blood sugar.', \
        "Fuzzy match failed to find correct description for 'diabates' -> 'diabetes'"
    # Precautions should NOT be found (as 'diabetes' isn't in mock precaution data)
    # and should return the default message.
    assert "No specific precautions found" in details_typo["precautions"][0], \
        "Precautions should not be found via fuzzy match as 'diabetes' not in prec_map"
    print("  ✅ Fuzzy match test completed.")