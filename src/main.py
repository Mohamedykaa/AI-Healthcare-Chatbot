# main.py (Enhanced Version)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
import json
from typing import List, Dict, Tuple
import os
import sys

# --- Add project root to path to allow imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.text_cleaner import TextCleaner


# --- App Initialization ---
app = FastAPI(
    title="Hybrid Disease Prediction API",
    description="Combines an NLP model with a rule-based engine to provide dynamic, interactive pre-diagnoses.",
    version="3.0.0"
)

# --- Load Model & Encoder ---
try:
    # This line makes the TextCleaner class available to joblib
    import __main__
    __main__.TextCleaner = TextCleaner

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, "models", "optimized_nlp_pipeline.joblib")
    encoder_path = os.path.join(base_dir, "models", "nlp_label_encoder.joblib")

    model_pipeline = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print("✅ API: Model and encoder loaded successfully!")

except Exception as e:
    model_pipeline = None
    label_encoder = None
    print(f"❌ API Error: Could not load model files: {e}")

# --- Knowledge Base Loading and Pre-processing ---
def preprocess_kb_for_fast_lookup(kb_rules: List[Dict]) -> Dict:
    """ Pre-processes the KB list into a dictionary for O(1) lookups. """
    processed_kb = {}
    for i, rule in enumerate(kb_rules):
        # Create unique IDs for follow-up questions for a simpler API
        for j, follow_up in enumerate(rule.get("follow_ups", [])):
            follow_up["id"] = f"q_{i}_{j}"
        
        for condition in rule.get("conditions", []):
            disease_name = condition["name"].lower()
            if disease_name not in processed_kb:
                processed_kb[disease_name] = []
            processed_kb[disease_name].append(rule)
    return processed_kb

knowledge_base = {}
raw_kb_rules = []
try:
    kb_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "english_knowledge_base.json")
    with open(kb_path, "r") as f:
        kb_json = json.load(f)
        raw_kb_rules = kb_json.get("rules", [])
        knowledge_base = preprocess_kb_for_fast_lookup(raw_kb_rules)
        print("✅ Knowledge base loaded and pre-processed successfully!")
except Exception as e:
    print(f"⚠️ Could not load or process knowledge base: {e}")

# --- Pydantic Schemas ---
class SymptomRequest(BaseModel):
    text: str
    top_k: int = 3
    follow_up_answers: Dict[str, str] = {} # e.g., {"q_0_0": "yes", "q_1_0": "no"}

# --- Helper Functions for Prediction Logic ---
def find_rule_based_predictions(text: str) -> List[Dict]:
    """Finds conditions from the KB where symptoms match the input text."""
    text_lower = text.lower()
    rule_predictions = []
    for rule in raw_kb_rules:
        matched_symptoms = [s for s in rule["symptoms"] if s.lower() in text_lower]
        match_score = len(matched_symptoms) / len(rule["symptoms"]) if rule["symptoms"] else 0

        if match_score >= 0.5:  # Consider rule if 50% or more symptoms match
            for condition in rule["conditions"]:
                final_score = condition["score"] * match_score
                rule_predictions.append({"disease": condition["name"], "probability": final_score})
    return rule_predictions

def apply_kb_rules(disease_name: str, base_prob: float, answers: Dict) -> Tuple[float, List[Dict]]:
    """ Helper function to find KB rules and apply boosts from follow-up questions. """
    disease_key = disease_name.lower()
    final_prob = base_prob
    questions = []
    
    rules = knowledge_base.get(disease_key, [])
    
    if rules:
        rule = rules[0]
        for cond in rule.get("conditions", []):
            if cond["name"].lower() == disease_key:
                # Use the higher of the ML prob or the rule's base score
                final_prob = max(base_prob, cond.get("score", base_prob))
                break
        
        boost_total = 0.0
        for follow_up in rule.get("follow_ups", []):
            q_id = follow_up["id"]
            question_text = follow_up["question"]
            questions.append({"id": q_id, "text": question_text})
            
            user_answer = answers.get(q_id, "").lower()
            if user_answer in ["yes", "y"]:
                boost_total += float(follow_up.get("boost_value", 0.0))
        
        final_prob += boost_total

    return min(final_prob, 1.0), questions

# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "Hybrid Disease Prediction API is running."}

@app.post("/predict")
def predict(request: SymptomRequest):
    if not model_pipeline or not label_encoder:
        raise HTTPException(status_code=503, detail="Model not loaded properly.")

    try:
        # 1. Get ML model predictions for all classes
        text_input = [request.text]
        model_probs = model_pipeline.predict_proba(text_input)[0]
        ml_predictions = {label_encoder.classes_[i]: prob for i, prob in enumerate(model_probs)}

        # 2. Get rule-based predictions from knowledge base
        rule_predictions_list = find_rule_based_predictions(request.text)
        rule_predictions = {p["disease"]: p["probability"] for p in rule_predictions_list}

        # 3. Combine ML and Rule-based predictions into a hybrid score
        combined_scores = ml_predictions.copy()
        for disease, rule_prob in rule_predictions.items():
            ml_prob = combined_scores.get(disease, 0.0)
            # Give higher weight to rule-based predictions
            combined_scores[disease] = (rule_prob * 0.7) + (ml_prob * 0.3)

        # 4. Sort all diseases by the new hybrid score
        sorted_diseases = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        top_k_hybrid = sorted_diseases[:request.top_k]
        
        # 5. Build the final response, applying follow-up boosts to the top results
        final_predictions = []
        for disease_name, hybrid_prob in top_k_hybrid:
            # Apply boosts on top of the already improved hybrid probability
            boosted_prob, extra_questions = apply_kb_rules(disease_name, hybrid_prob, request.follow_up_answers)
            
            final_predictions.append({
                "disease": disease_name,
                "probability": round(boosted_prob, 4),
                "follow_up_questions": extra_questions,
            })

        # 6. Re-sort in case boosts changed the order and return
        final_predictions = sorted(final_predictions, key=lambda x: x["probability"], reverse=True)
        return {"predictions": final_predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

