# D:\disease_prediction_project\src\main.py

# --- Path modification ---
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✅ Added project root to sys.path: {project_root}")


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, json, re
from typing import List, Dict, Tuple
import traceback

# --- Try importing TextCleaner from its new location ---
try:
    # ✅ This import should now match what the joblib file expects
    from src.utils.text_cleaner import TextCleaner
    print("✅ TextCleaner imported successfully from src.utils.text_cleaner.")

except ImportError as import_err:
    print(f"❌ Critical Error: Could not import TextCleaner from src.utils.text_cleaner: {import_err}")
    TextCleaner = None
    print("⚠️ TextCleaner could not be imported. Model loading will likely fail.")


# --- Initialize FastAPI App ---
app = FastAPI(
    title="Hybrid Disease Prediction API (v3.4)",
    description="Uses merged dataset + hybrid knowledge-based engine for improved medical predictions.",
    version="3.4.0",
)

# --- Load Model & Encoder ---
model_pipeline = None
label_encoder = None
if TextCleaner is not None:
    try:
        base_dir = project_root
        model_path = os.path.join(base_dir, "models", "optimized_nlp_pipeline.joblib")
        encoder_path = os.path.join(base_dir, "models", "nlp_label_encoder.joblib")

        # ✅ --- HACK REMOVED ---
        # We no longer need the 'sys.modules' or '__main__' hacks
        # because the new model file (created by the updated train_model.py)
        # already knows the correct path: 'src.utils.text_cleaner'

        print(f"Attempting to load model from: {model_path}")
        model_pipeline = joblib.load(model_path) # <<< This should now work
        print(f"Attempting to load encoder from: {encoder_path}")
        label_encoder = joblib.load(encoder_path)
        print("✅ Model and encoder loaded successfully (v3.4 - Cleaned).")

    except FileNotFoundError as fnf_err:
        print(f"❌ Error loading model files: File not found - {fnf_err}")
    except ModuleNotFoundError as mod_err:
        # If this error still appears, it means you haven't retrained the model
        print(f"❌ Error loading model files: ModuleNotFoundError - {mod_err}")
        print(f"❌ MAKE SURE YOU HAVE RUN THE UPDATED 'train_model.py' SCRIPT FIRST.")
        print(traceback.format_exc())
    except Exception as e:
        print(f"❌ Error loading model files: {type(e).__name__} - {e}")
        print(traceback.format_exc())
else:
    print("❌ Model loading skipped because TextCleaner could not be imported.")


# --- Load Knowledge Base ---
def preprocess_kb(kb_rules: List[Dict]) -> Dict:
    kb_dict = {}
    for i, rule in enumerate(kb_rules):
        disease_name_for_id = "unknown_disease"
        if rule.get("conditions"):
            disease_name_for_id = rule["conditions"][0].get("name", f"rule{i}").lower().replace(" ", "_")
        for j, follow_up in enumerate(rule.get("follow_ups", [])):
            q_text_slug = re.sub(r'\W+', '_', follow_up.get("question", "")[:20]).lower().strip('_')
            follow_up["id"] = follow_up.get("id") or f"q_{disease_name_for_id}_{q_text_slug}_{j}"
        for condition in rule.get("conditions", []):
            disease_name = condition.get("name", "").lower()
            if disease_name: kb_dict.setdefault(disease_name, []).append(rule)
    return kb_dict

knowledge_base, raw_kb_rules = {}, []
try:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    kb_path = os.path.join(base_dir, "data", "english_knowledge_base.json")
    print(f"Attempting to load KB from: {kb_path}")
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_json = json.load(f)
        raw_kb_rules = kb_json.get("rules", [])
        knowledge_base = preprocess_kb(raw_kb_rules)
        print(f"✅ Knowledge base loaded successfully. {len(knowledge_base)} diseases indexed.")
except FileNotFoundError:
    print(f"⚠️ KB file not found at {kb_path}. Rules limited.")
except Exception as e:
    print(f"⚠️ Could not load/preprocess KB: {e}")
    print(traceback.format_exc())

# --- Input Schema ---
class SymptomRequest(BaseModel):
    text: str
    top_k: int = 3
    follow_up_answers: Dict[str, str] = {}

# --- Helper Functions ---
def find_rule_based_predictions(text: str) -> List[Dict]:
    if not raw_kb_rules: return []
    text_lower = text.lower()
    predictions = []
    symptoms_in_text = set()
    try:
        if TextCleaner:
            cleaned_text = TextCleaner().transform(text_lower)
            symptoms_in_text = set(cleaned_text.split())
        else: symptoms_in_text = set(re.findall(r'\b\w+\b', text_lower))
    except Exception: symptoms_in_text = set(re.findall(r'\b\w+\b', text_lower))
    
    # (Rest of the function is correct)
    for rule in raw_kb_rules:
        rule_symptoms = set(s.lower() for s in rule.get("symptoms", []))
        if not rule_symptoms: continue
        matched_symptoms = rule_symptoms.intersection(symptoms_in_text)
        if matched_symptoms:
            match_ratio = len(matched_symptoms) / len(rule_symptoms)
            if match_ratio >= 0.4:
                for cond in rule.get("conditions", []):
                    disease_name = cond.get("name")
                    if disease_name:
                        score = cond.get("score", 0.5) * match_ratio
                        predictions.append({"disease": disease_name, "probability": score})
    aggregated_preds = {}
    for p in predictions:
        d = p["disease"]
        aggregated_preds[d] = max(aggregated_preds.get(d, 0.0), p["probability"])
    return [{"disease": d, "probability": p} for d, p in aggregated_preds.items()]

def apply_kb_rules( disease: str, base_prob: float, answers: Dict[str, str]) -> Tuple[float, List[Dict]]:
    disease_key = disease.lower()
    prob = base_prob
    questions_to_return = []
    boost = 0.0
    NEGATIVE_BOOST_MULTIPLIER = -0.25
    rules_for_disease = knowledge_base.get(disease_key, [])
    if not rules_for_disease: return prob, []
    processed_qids = set()
    
    # (Rest of the function is correct)
    for rule in rules_for_disease:
        for follow_up in rule.get("follow_ups", []):
            qid = follow_up.get("id")
            if not qid: continue
            if qid not in processed_qids:
                questions_to_return.append(follow_up)
                processed_qids.add(qid)
            ans = answers.get(qid, "").lower()
            if not ans: continue
            for boost_item in follow_up.get("boosts", []):
                if boost_item.get("name", "").lower() == disease_key:
                    try:
                        boost_value = float(boost_item.get("value", 0.0))
                        multiplier = 0.5
                        if ans in ["yes", "y"]: multiplier = 1.0
                        elif ans in ["no", "n"]: multiplier = NEGATIVE_BOOST_MULTIPLIER
                        boost += boost_value * multiplier
                        break
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid boost value for qid {qid}: {boost_item}")
    final_prob = min(max(prob + boost, 0.0), 1.0)
    return final_prob, questions_to_return

# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "Hybrid Disease Prediction API (v3.4) is running."}

@app.post("/predict")
def predict_v1(req: SymptomRequest):
    print("Received request on legacy /predict endpoint.")
    return predict_logic(req, version="v1")

@app.post("/predict_v2")
def predict_v2(req: SymptomRequest):
    print(f"Received request on /predict_v2: text='{req.text}', answers={req.follow_up_answers}")
    return predict_logic(req, version="v2")

# Main prediction logic
def predict_logic(req: SymptomRequest, version="v2"):
    if not model_pipeline or not label_encoder:
        print("❌ Prediction failed: Model or encoder not available.")
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")
    try:
        text_input = [req.text]
        model_probs = model_pipeline.predict_proba(text_input)[0]
        ml_predictions = { label_encoder.classes_[i]: prob for i, prob in enumerate(model_probs) }
        print(f"ℹ️ ML Predictions (raw): {dict(list(sorted(ml_predictions.items(), key=lambda item: item[1], reverse=True))[:5])}")
        
        rule_preds_list = find_rule_based_predictions(req.text)
        rule_preds = {p["disease"]: p["probability"] for p in rule_preds_list}
        print(f"ℹ️ Rule Predictions: {rule_preds}")
        
        # (Rest of the logic is correct)
        combined_scores = ml_predictions.copy()
        for disease, rule_prob in rule_preds.items():
            ml_prob = combined_scores.get(disease, 0.0)
            combined_scores[disease] = min((ml_prob * 0.7) + (rule_prob * 0.3), 1.0)
        
        sorted_diseases = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_results_unboosted = [(d, p) for d, p in sorted_diseases if p > 0.0][: req.top_k]
        print(f"ℹ️ Top {req.top_k} Combined (Pre-Boost): {top_results_unboosted}")
        
        final_predictions = []
        all_follow_up_questions = {}
        for disease, probability in top_results_unboosted:
            boosted_prob, questions_for_disease = apply_kb_rules(disease, probability, req.follow_up_answers)
            final_predictions.append({ "disease": disease, "probability": round(boosted_prob, 4)})
            for q in questions_for_disease:
                q_id = q.get("id")
                if q_id and q_id not in all_follow_up_questions:
                    all_follow_up_questions[q_id] = q
        
        final_predictions = sorted(final_predictions, key=lambda x: x["probability"], reverse=True)
        print(f"ℹ️ Final Top {req.top_k} (Post-Boost): {final_predictions}")
        
        if final_predictions:
            final_predictions[0]["follow_up_questions"] = list(all_follow_up_questions.values())
            for i in range(1, len(final_predictions)):
                final_predictions[i]["follow_up_questions"] = []
        
        return {
            "predictions": final_predictions,
            "model_version": "v3.4",
            "input_text": req.text,
        }
    except Exception as e:
        print(f"❌ Error during prediction logic: {type(e).__name__} - {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")