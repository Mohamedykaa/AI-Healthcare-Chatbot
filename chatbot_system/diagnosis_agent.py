import joblib
import os
import traceback
import numpy as np
import sys
import json
from typing import List, Dict

# Add the project root to the path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.text_cleaner import TextCleaner


class DiagnosisAgent:
    """
    Handles disease prediction using a hybrid of a trained ML model
    and a rule-based knowledge base.
    """

    def __init__(self, model_path=None, encoder_path=None, kb_path=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "optimized_nlp_pipeline.joblib")
        if encoder_path is None:
            encoder_path = os.path.join(base_dir, "models", "nlp_label_encoder.joblib")
        # --- UPDATE: Use the more detailed english_knowledge_base.json for rules ---
        if kb_path is None:
            kb_path = os.path.join(base_dir, "english_knowledge_base.json")
            
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.kb_path = kb_path
        
        self.model = None
        self.label_encoder = None
        self.hybrid_rules = []
        
        self.load_model()
        self.load_knowledge_base()

    def load_model(self):
        """ Loads the trained model and encoder safely. """
        try:
            import __main__
            __main__.TextCleaner = TextCleaner

            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            if os.path.exists(self.encoder_path):
                self.label_encoder = joblib.load(self.encoder_path)
            else:
                raise FileNotFoundError(f"Label encoder not found: {self.encoder_path}")

            print("✅ Diagnosis model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            traceback.print_exc()

    def load_knowledge_base(self):
        """ Loads the raw knowledge base rules from the correct file. """
        try:
            with open(self.kb_path, "r") as f:
                self.hybrid_rules = json.load(f).get("rules", [])
            print("✅ Diagnosis Agent: Hybrid Knowledge base loaded.")
        except Exception as e:
            print(f"⚠️ Diagnosis Agent: Could not load knowledge base: {e}")


    def predict_top_diseases(self, text_input, top_k=3):
        """
        Predicts top diseases using the definitive HYBRID approach.
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Please check model path.")

        try:
            # 1. Get ML model predictions for all classes
            probabilities = self.model.predict_proba([text_input])[0]
            ml_predictions = {self.label_encoder.classes_[i]: prob for i, prob in enumerate(probabilities)}

            # 2. Get rule-based predictions from the hybrid knowledge base
            text_lower = text_input.lower()
            rule_predictions_list = []
            for rule in self.hybrid_rules:
                matched_symptoms = [s for s in rule["symptoms"] if s.lower() in text_lower]
                match_score = len(matched_symptoms) / len(rule["symptoms"]) if rule["symptoms"] else 0

                if match_score >= 0.5:
                    for condition in rule["conditions"]:
                        final_score = condition["score"] * match_score
                        rule_predictions_list.append({"disease": condition["name"], "probability": final_score})
            
            rule_predictions = {p["disease"]: p["probability"] for p in rule_predictions_list}

            # --- UPDATE: Use a more powerful MAX-based combination logic ---
            combined_scores = ml_predictions.copy()
            for disease, rule_prob in rule_predictions.items():
                ml_prob = combined_scores.get(disease, 0.0)
                # The final score is the HIGHER of the rule's score or the model's score.
                # This ensures strong rules always override weak model predictions.
                combined_scores[disease] = max(rule_prob, ml_prob)

            # 4. Sort and get top K results from the definitive combined scores
            sorted_diseases = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
            top_k_hybrid = sorted_diseases[:top_k]
            
            results = [{"disease": disease, "probability": prob} for disease, prob in top_k_hybrid]
            return results

        except Exception as e:
            print(f"⚠️ Prediction failed: {e}")
            traceback.print_exc()
            return []

