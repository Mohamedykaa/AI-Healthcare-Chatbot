import json
from typing import Dict

class RecommendationAgent:
    """
    This agent provides recommendations (tests, advice, specialist, and risk level)
    for a given disease by looking it up in its knowledge base.
    """

    def __init__(self, knowledge_path="chatbot_system/knowledge_base.json"):
        try:
            with open(knowledge_path, "r", encoding="utf-8") as f:
                self.knowledge = json.load(f)
            print("✅ Knowledge base loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            self.knowledge = {}

    def get_recommendations(self, disease_name: str) -> Dict:
        """
        Returns a dictionary with tests, advice, specialist, and risk level for a given disease.
        """
        disease_key = disease_name.lower()

        if disease_key in self.knowledge:
            data = self.knowledge[disease_key]
            data["risk_level"] = data.get("risk_level", "unknown")
            data["specialist"] = data.get("specialist", "General Practitioner")
            return data

        return {
            "tests": ["N/A"],
            "advice": "No specific advice found in the knowledge base. Please consult a doctor.",
            "specialist": "General Practitioner",
            "risk_level": "unknown"
        }
