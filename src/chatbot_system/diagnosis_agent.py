import joblib
import os
import traceback
import sys
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import Counter
import re # Needed for fallback CSV processing

# Ensure project root is in the system path for util import
sys.path.append(str(Path(__file__).resolve().parents[2]))

# --- CRITICAL IMPORT: TextCleaner ---
try:
    from src.utils.text_cleaner import TextCleaner
    print("✅ DiagnosisAgent: TextCleaner imported successfully.")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: DiagnosisAgent cannot import TextCleaner: {e}")
    print("   Ensure src/utils/text_cleaner.py exists and the environment is correct.")
    # Option 1: Raise error immediately
    raise ImportError("TextCleaner is essential for DiagnosisAgent and could not be imported.") from e
    # Option 2: Set TextCleaner to None and handle it later (less safe)
    # TextCleaner = None
    # print("   DiagnosisAgent will operate without proper text cleaning!")


class DiagnosisAgent:
    """
    🚀 DiagnosisAgent v3.4 (Refactored & Corrected)
    --------------------------------
    - Combines ML, Knowledge Base (KB), and CSV fallback (using wide CSV).
    - Integrates with FollowUpManager for adaptive questioning & boosting.
    - Requires TextCleaner for consistent preprocessing.
    - Improved rule and CSV symptom matching logic.
    - Removed redundant methods and old hacks.
    """

    def __init__(self, model_path=None, encoder_path=None, kb_path=None, csv_fallback_path=None):
        base = Path(__file__).resolve().parents[2]

        # Default paths relative to project root
        self.model_path = model_path or str(base / "models" / "optimized_nlp_pipeline.joblib")
        self.encoder_path = encoder_path or str(base / "models" / "nlp_label_encoder.joblib")
        self.kb_path = kb_path or str(base / "data" / "english_knowledge_base.json")
        # ✅ Use the wide CSV as fallback source
        self.csv_fallback_path = csv_fallback_path or str(base / "data" / "DiseaseAndSymptoms.csv")

        # Components
        self.pipeline: Optional[Any] = None
        self.label_encoder: Optional[Any] = None
        self.raw_kb_rules: List[Dict[str, Any]] = []
        self.kb_lookup: Dict[str, List[Dict]] = {}
        self.fallback_data_wide: Optional[pd.DataFrame] = None # Store the wide CSV

        # Configuration
        self.rule_match_threshold = 0.25 # Min overlap ratio for a rule to be considered
        self.key_symptom_boost = 0.10 # Small boost if user mentions phrases from follow-up questions
        self.max_questions_per_disease = 4 # Max follow-ups to return per disease

        # ✅ Instantiate TextCleaner (raises error if import failed)
        if TextCleaner is None:
             # This path is taken only if Option 2 was chosen above
             print("❌ ERROR: TextCleaner could not be imported. Agent cannot function.")
             raise RuntimeError("TextCleaner import failed.")
        self.text_cleaner_instance = TextCleaner()
        print("✅ DiagnosisAgent: TextCleaner instance created.")

        # Load all components
        self.load_model()
        self.load_knowledge_base()
        self.load_fallback_dataset() # Loads the wide CSV

    # ====================================================
    # LOADERS
    # ====================================================

    def load_model(self):
        """Loads the ML pipeline and label encoder."""
        try:
            # ✅ Removed the __main__.TextCleaner hack - no longer needed with corrected training
            if os.path.exists(self.model_path):
                self.pipeline = joblib.load(self.model_path)
                print(f"✅ DiagnosisAgent: Model pipeline loaded from {self.model_path}.")
            else:
                print(f"⚠️ Warning: Model pipeline not found at {self.model_path}. ML predictions disabled.")
                self.pipeline = None

            if os.path.exists(self.encoder_path):
                self.label_encoder = joblib.load(self.encoder_path)
                print(f"✅ DiagnosisAgent: Label encoder loaded from {self.encoder_path}.")
            else:
                print(f"⚠️ Warning: Label encoder not found at {self.encoder_path}. ML predictions disabled.")
                self.label_encoder = None

        except Exception as e:
            print(f"❌ Error loading model/encoder: {e}")
            traceback.print_exc()
            self.pipeline, self.label_encoder = None, None

    def load_knowledge_base(self):
        """Loads and indexes the JSON knowledge base."""
        try:
            if os.path.exists(self.kb_path):
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    kb = json.load(f)
                    # Ensure rules use lowercase standardized symptoms
                    self.raw_kb_rules = kb.get("rules", [])
                    for rule in self.raw_kb_rules:
                        if 'symptoms' in rule:
                            rule['symptoms'] = [str(s).strip().lower().replace('_',' ') for s in rule['symptoms']]

            else:
                print(f"⚠️ Warning: Knowledge base file not found at {self.kb_path}. Rule-based matching disabled.")
                self.raw_kb_rules = []

            # Index KB by lowercase condition name for fast lookup
            self.kb_lookup = {}
            for rule in self.raw_kb_rules:
                for cond in rule.get("conditions", []):
                    # Name should already be lowercase from the generator script
                    name = cond.get("name", "").strip()
                    if name:
                        self.kb_lookup.setdefault(name, []).append(rule)

            print(f"✅ Knowledge base loaded. {len(self.kb_lookup)} diseases indexed.")
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            traceback.print_exc()
            self.raw_kb_rules = []
            self.kb_lookup = {}

    def load_fallback_dataset(self):
        """Loads the WIDE fallback CSV dataset."""
        try:
            if os.path.exists(self.csv_fallback_path):
                self.fallback_data_wide = pd.read_csv(self.csv_fallback_path)
                # Clean column names immediately
                self.fallback_data_wide.columns = [str(c).strip().lower().rstrip('_') for c in self.fallback_data_wide.columns]
                # Ensure 'disease' column exists
                if 'disease' not in self.fallback_data_wide.columns:
                     print(f"❌ Error: Fallback CSV {self.csv_fallback_path} is missing 'disease' column after cleaning.")
                     self.fallback_data_wide = None
                else:
                     print(f"✅ Fallback dataset (wide) loaded with {len(self.fallback_data_wide)} entries.")
            else:
                print(f"⚠️ Warning: Fallback dataset not found at {self.csv_fallback_path}. CSV fallback disabled.")
                self.fallback_data_wide = None
        except Exception as e:
            print(f"❌ Error loading fallback dataset: {e}")
            traceback.print_exc()
            self.fallback_data_wide = None

    # ====================================================
    # TEXT CLEANING (Uses TextCleaner instance)
    # ====================================================

    def _clean_text(self, text: Any) -> str:
        """Cleans text using the TextCleaner instance."""
        if not text:
            return ""
        try:
            # ✅ Use the instantiated cleaner
            return self.text_cleaner_instance.transform(text)
        except Exception as e:
            print(f"⚠️ Error during text cleaning: {e}. Returning raw text.")
            # Fallback to basic string conversion if transform fails
            return str(text).lower().strip()

    # ====================================================
    # SCORING METHODS
    # ====================================================

    def _ml_scores(self, cleaned_text: str) -> Dict[str, float]:
        """Return ML-based probabilities."""
        # Check if both pipeline and encoder are loaded
        if self.pipeline is None or self.label_encoder is None:
            return {}
        if not hasattr(self.pipeline, "predict_proba"):
             print("⚠️ Warning: Loaded pipeline object does not have 'predict_proba' method.")
             return {}

        try:
            probs = self.pipeline.predict_proba([cleaned_text])[0]
            # Ensure classes are strings and lowercase
            classes = [str(cls).lower() for cls in self.label_encoder.classes_]
            return {classes[i]: float(probs[i]) for i in range(len(classes))}
        except Exception as e:
            print(f"❌ Error during ML prediction: {e}")
            traceback.print_exc()
            return {}

    def _rule_match_scores(self, cleaned_text: str) -> Dict[str, float]:
        """Match user input against KB rules using improved logic."""
        if not self.raw_kb_rules:
             return {}

        # User's input symptoms as a set of individual words
        text_tokens = set(cleaned_text.split())
        scores = {}

        for rule in self.raw_kb_rules:
            # Symptoms from the rule (already lowercased during loading)
            rule_symptoms = rule.get("symptoms", [])
            if not rule_symptoms: continue

            matched_symptom_count = 0
            # ✅ IMPROVED MATCHING: Check multi-word symptoms
            for symptom in rule_symptoms:
                symptom_words = set(symptom.split())
                # Check if ALL words of the symptom are present in the user's input tokens
                if symptom_words.issubset(text_tokens):
                    matched_symptom_count += 1

            if matched_symptom_count == 0: continue # Skip if no symptoms matched

            # Calculate match ratio based on matched symptoms
            ratio = matched_symptom_count / len(rule_symptoms)

            # Check for key follow-up phrases (simple substring match is okay here)
            key_hits = sum(1 for f in rule.get("follow_ups", []) if f.get("question", "").lower() in cleaned_text)
            # Find the base score for conditions in this rule
            cond_score = max([float(c.get("score", 0.5)) for c in rule.get("conditions", [])], default=0.5)

            # Combine scores
            weighted_score = (cond_score * ratio) + (self.key_symptom_boost * key_hits)

            # If score meets threshold, add/update score for associated diseases
            if weighted_score >= self.rule_match_threshold:
                for cond in rule.get("conditions", []):
                    # Disease name should already be lowercase from KB generation
                    name = cond.get("name", "").strip()
                    if name:
                        # Keep the highest score found for this disease from any matching rule
                        scores[name] = max(scores.get(name, 0.0), weighted_score)
        return scores

    def _csv_fallback_scores(self, cleaned_text: str) -> Dict[str, float]:
        """Use simple overlap matching from the WIDE CSV."""
        if self.fallback_data_wide is None:
            return {}

        user_symptoms_set = set(cleaned_text.split())
        results = {}

        # Identify symptom columns in the fallback data (e.g., symptom_1, symptom_2, ...)
        symptom_cols_fallback = [col for col in self.fallback_data_wide.columns if re.match(r'^symptom_\d+$', col)]
        if not symptom_cols_fallback:
             print("⚠️ Warning: No 'symptom_NUMBER' columns found in fallback CSV. Fallback scoring may fail.")
             return {}

        for _, row in self.fallback_data_wide.iterrows():
            disease = row.get("disease", "").strip() # Already lowercased during load
            if not disease: continue

            # ✅ Extract and standardize symptoms from the WIDE row
            row_symptoms = set()
            for col in symptom_cols_fallback:
                symptom_val = row.get(col)
                if pd.notna(symptom_val) and str(symptom_val).strip():
                    # Standardize symptom from CSV like other places
                    standardized_s = str(symptom_val).strip().lower().replace('_', ' ')
                    if standardized_s:
                        row_symptoms.add(standardized_s)

            if not row_symptoms: continue

            # Calculate overlap score
            overlap = len(user_symptoms_set.intersection(row_symptoms))
            if overlap > 0:
                # Score is the ratio of matched symptoms to total symptoms for that disease in the CSV row
                score = overlap / len(row_symptoms)
                # Keep the highest score found for this disease
                results[disease] = max(results.get(disease, 0.0), score)

        return results


    # ====================================================
    # COMBINATION LOGIC
    # ====================================================

    def _combine_scores(self, ml_scores, rule_scores, csv_scores, followup_boosts):
        """Merge all score sources, applying boosts."""
        merged = {}
        # Ensure all disease names are consistently lowercase
        all_diseases = set(ml_scores.keys()) | set(rule_scores.keys()) | set(csv_scores.keys()) | set(followup_boosts.keys())

        for disease_lc in all_diseases:
            # Get scores, defaulting to 0.0
            m = ml_scores.get(disease_lc, 0.0)
            r = rule_scores.get(disease_lc, 0.0)
            c = csv_scores.get(disease_lc, 0.0)

            # Weighted average - adjust weights as needed
            # Giving ML highest weight, then rules, then CSV fallback
            base_score = (0.6 * m) + (0.3 * r) + (0.1 * c)

            # Apply boosts from follow-up answers (keys in followup_boosts are already lowercase)
            boost_val = float(followup_boosts.get(disease_lc, 0.0))

            # Calculate final score, capped between 0.0 and 1.0
            final_score = min(max(base_score + boost_val, 0.0), 1.0)

            merged[disease_lc] = round(final_score, 4) # Store with 4 decimal places

        return merged

    # ====================================================
    # FOLLOW-UP RETRIEVAL
    # ====================================================

    def _get_followups_for_disease(self, disease_name_lc: str) -> List[Dict]:
        """Retrieve and prioritize follow-up questions for a disease."""
        # disease_name_lc is already lowercase
        rules = self.kb_lookup.get(disease_name_lc, [])
        questions = []
        seen_questions_text = set() # For efficient de-duplication

        # Collect all unique questions from relevant rules
        for rule in rules:
            for followup in rule.get("follow_ups", []):
                q_text = followup.get("question", "").strip()
                q_id = followup.get("id")
                if not q_text or not q_id: continue

                # Use lowercase, punctuation-free text for de-duplication check
                q_text_key = re.sub(r'[^\w\s]', '', q_text.lower()).strip()

                if q_text_key not in seen_questions_text:
                    seen_questions_text.add(q_text_key)
                    questions.append({
                        "id": q_id,
                        "text": q_text, # Keep original text for display
                        "severity": followup.get("severity", 1) # Default severity 1
                    })

        # Prioritize questions: Higher severity first, then potentially by frequency if needed
        # (Frequency calculation removed for simplicity, severity is primary)
        sorted_q = sorted(questions, key=lambda q: q["severity"], reverse=True)

        return sorted_q[:self.max_questions_per_disease] # Return top N questions

    # ====================================================
    # MAIN PREDICTION METHOD
    # ====================================================

    def predict(self, text_input: str, followup_manager: Optional[Any] = None, top_k: int = 3) -> Dict[str, Any]:
        """
        Performs diagnosis by combining ML, Rules, and CSV scores,
        applying follow-up boosts, and retrieving relevant follow-up questions.
        """
        try:
            # 1. Clean the input text
            cleaned_text = self._clean_text(text_input)
            if not cleaned_text:
                 print("⚠️ Warning: Input text was empty after cleaning.")
                 return {"predictions": []}

            # 2. Get scores from all sources
            ml = self._ml_scores(cleaned_text)
            rules = self._rule_match_scores(cleaned_text)
            csv = self._csv_fallback_scores(cleaned_text)

            # 3. Get boosts from FollowUpManager (expects lowercase keys)
            boosts = {}
            if followup_manager and hasattr(followup_manager, 'get_disease_boosts'):
                boosts = followup_manager.get_disease_boosts()

            # 4. Combine scores and apply boosts
            combined_scores = self._combine_scores(ml, rules, csv, boosts)

            # 5. Get top K predictions with score > 0
            # Ensure keys are consistently lowercase before sorting
            top_predictions_raw = sorted(
                 ((disease.lower(), score) for disease, score in combined_scores.items() if score > 0),
                 key=lambda item: item[1],
                 reverse=True
            )[:top_k]


            # 6. Format results and retrieve follow-up questions
            final_predictions = []
            for disease_lc, prob in top_predictions_raw:
                # Retrieve follow-ups for the lowercase disease name
                follow_up_questions = self._get_followups_for_disease(disease_lc)

                # Add questions to the FollowUpManager if provided
                if followup_manager and hasattr(followup_manager, 'add_questions') and follow_up_questions:
                    # Pass questions with scope so manager knows which disease they belong to
                    followup_manager.add_questions(follow_up_questions, disease_scope=disease_lc)

                final_predictions.append({
                    "disease": disease_lc, # Return lowercase disease name
                    "probability": round(prob, 3),
                    # Optional: include source mix for debugging/analysis
                    # "source_mix": {
                    #     "ml": ml.get(disease_lc, 0),
                    #     "rules": rules.get(disease_lc, 0),
                    #     "csv": csv.get(disease_lc, 0)
                    # },
                    # Return the questions intended for this disease prediction stage
                    "follow_up_questions": [
                         {"id": q["id"], "text": q["text"], "severity": q["severity"]}
                         for q in follow_up_questions
                    ]
                })

            return {"predictions": final_predictions}

        except Exception as e:
            print(f"❌ Error during DiagnosisAgent.predict: {e}")
            traceback.print_exc()
            return {"predictions": []} # Return empty list on failure

    # Note: Removed predict_top_diseases as predict now returns the desired structure.
    # Callers should use agent.predict(...) and access the 'predictions' key.

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing DiagnosisAgent ---")
    agent = DiagnosisAgent()

    # Simple test case
    test_symptom = "i have high fever and persistent headache with body aches"
    print(f"\nInput: {test_symptom}")

    # --- Test 1: Initial prediction (no FollowUpManager) ---
    print("\n--- Test 1: Initial Prediction ---")
    initial_result = agent.predict(test_symptom, top_k=3)
    print("Result:")
    print(json.dumps(initial_result, indent=2))

    # --- Test 2: Prediction with FollowUpManager ---
    print("\n--- Test 2: Prediction with FollowUpManager ---")
    # Need FollowUpManager for this test
    try:
        from chatbot_system.followup_manager import FollowUpManager
        fm = FollowUpManager()
        # Simulate answering 'yes' to a potential flu question (ID needs to match generated KB)
        # Example: Find a question ID from the KB JSON first
        # fm.record_answer("q_flu_example_symptom", "yes") # Replace with a real ID
        print("Simulating interaction with FollowUpManager...")
        result_with_fm = agent.predict(test_symptom, followup_manager=fm, top_k=3)
        print("Result (with FollowUpManager):")
        print(json.dumps(result_with_fm, indent=2))
        print("\nFollowUpManager state after prediction:")
        print(f"  Pending questions: {len(fm.pending_questions)}")
        print(f"  Disease boosts: {fm.get_disease_boosts()}")

    except ImportError:
        print("\nCould not import FollowUpManager, skipping Test 2.")
    except Exception as e:
        print(f"\nError during Test 2: {e}")

    print("\n--- DiagnosisAgent Test Complete ---")