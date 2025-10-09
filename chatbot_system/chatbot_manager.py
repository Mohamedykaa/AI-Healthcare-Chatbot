# chatbot_system/chatbot_manager.py

from chatbot_system.symptom_agent import SymptomAgent
from chatbot_system.diagnosis_agent import DiagnosisAgent
from chatbot_system.recommendation_agent import RecommendationAgent

class ChatbotManager:
    """Main orchestrator managing the flow between agents."""

    def __init__(self):
        print("Initializing chatbot agents...")
        self.symptom_agent = SymptomAgent()
        self.diagnosis_agent = DiagnosisAgent()
        self.recommendation_agent = RecommendationAgent()
        print("-" * 50)

    def start_chat(self):
        print("👋 Welcome to the Advanced AI Healthcare Chatbot!")
        print("Describe your symptoms in a full sentence (type 'exit' to quit).")
        print("-" * 50)

        while True:
            user_input = input("\nYou: ")
            if user_input.lower().strip() == "exit":
                print("\n🩺 Goodbye! Stay healthy 💙")
                break

            if not user_input.strip():
                print("Bot: Please describe your symptoms first.")
                continue

            # 1️⃣ Collect symptoms
            symptom_text = self.symptom_agent.collect_symptoms(user_input)

            # 2️⃣ Diagnose using the AI model
            predictions = self.diagnosis_agent.predict_top_diseases(symptom_text)

            print("\n🤖 Bot: Based on your symptoms, here are possible conditions:")
            for pred in predictions:
                disease = pred['disease']
                prob = pred['probability']
                print(f"\n- {disease} (Likelihood: {prob:.2%})")

                # 3️⃣ Get recommendations
                rec = self.recommendation_agent.get_recommendations(disease)
                tests = ", ".join(rec['tests'])
                advice = rec['advice']

                print(f"  🧪 Recommended tests: {tests}")
                print(f"  💡 Advice: {advice}")

            print("\n" + "-" * 50)

if __name__ == "__main__":
    manager = ChatbotManager()
    manager.start_chat()
