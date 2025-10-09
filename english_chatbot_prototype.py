# english_chatbot_prototype.py
import json
from typing import List, Dict

def load_knowledge_base(filepath: str) -> Dict:
    """Loads the knowledge base from a JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Knowledge base file not found at {filepath}")
        return {}

def find_best_matching_rule(user_symptoms: List[str], kb: Dict) -> Dict:
    """Finds the best matching rule based on symptom overlap."""
    best_rule = None
    max_score = -1
    
    user_symptoms_set = set(s.lower().strip() for s in user_symptoms)
    
    for rule in kb.get("rules", []):
        rule_symptoms_set = set(rule.get("symptoms", []))
        intersection = user_symptoms_set.intersection(rule_symptoms_set)
        if not intersection:
            continue
            
        score = len(intersection) / len(rule_symptoms_set)
        if score > max_score:
            max_score = score
            best_rule = rule
            
    return best_rule

def get_initial_diagnosis(rule: Dict) -> List[Dict]:
    if not rule:
        return []
    return sorted(rule.get("conditions", []), key=lambda x: x["score"], reverse=True)

def handle_follow_ups(rule: Dict, initial_diagnosis: List[Dict]) -> List[Dict]:
    if not rule or not rule.get("follow_ups"):
        return initial_diagnosis

    print("\nTo help refine the diagnosis, please answer the following questions (yes/no):")
    diagnosis_map = {cond["name"]: cond for cond in initial_diagnosis}
    
    for follow_up in rule.get("follow_ups", []):
        answer = input(f"> {follow_up['question']} ")
        if answer.lower().strip() in ["yes", "y"]:
            condition_to_boost = follow_up.get("if_yes_boosts")
            boost_value = follow_up.get("boost_value", 0.0)
            if condition_to_boost in diagnosis_map:
                diagnosis_map[condition_to_boost]["score"] += boost_value
            else:
                initial_diagnosis.append({"name": condition_to_boost, "score": boost_value})

    return sorted(initial_diagnosis, key=lambda x: x["score"], reverse=True)

def format_final_response(diagnosis: List[Dict]):
    if not diagnosis:
        return "\nI'm sorry, I couldn’t determine a possible condition based on your symptoms. Please consult a doctor."
    response = "\n✅ Refined Diagnosis (sorted by likelihood):\n"
    for cond in diagnosis:
        likelihood = f"{cond['score']*100:.0f}%"
        response += f"- {cond['name']} (Likelihood: {likelihood})\n"
    response += "\n⚠️ Disclaimer: This is a pre-diagnosis and not a substitute for professional medical advice. Always consult a doctor."
    return response

def main():
    knowledge_base = load_knowledge_base("english_knowledge_base.json")
    if not knowledge_base:
        return
    print("🩺 Welcome to the Enhanced Healthcare Chatbot!")
    print("Please describe your symptoms, separated by commas (e.g., headache, nausea).")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter your symptoms: ")
        if user_input.lower() == "exit":
            print("Goodbye! Stay healthy. 💙")
            break
            
        symptoms_list = [s.strip() for s in user_input.split(",")]
        best_rule = find_best_matching_rule(symptoms_list, knowledge_base)
        
        if not best_rule:
            print("\nI'm sorry, I don't have enough information about these symptoms. Please consult a doctor.")
            continue

        initial_diagnosis = get_initial_diagnosis(best_rule)
        refined_diagnosis = handle_follow_ups(best_rule, initial_diagnosis)
        print(format_final_response(refined_diagnosis))
        print("-" * 40)

if __name__ == "__main__":
    main()
