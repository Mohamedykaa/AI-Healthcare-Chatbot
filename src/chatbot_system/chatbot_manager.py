from chatbot_system.symptom_agent import SymptomAgent
from chatbot_system.diagnosis_agent import DiagnosisAgent
from chatbot_system.recommendation_agent import RecommendationAgent
from chatbot_system.followup_manager import FollowUpManager
import json


class ChatbotManager:
    """Main orchestrator managing the flow between all AI agents."""

    def __init__(self):
        print("Initializing chatbot agents...")
        self.symptom_agent = SymptomAgent()
        self.diagnosis_agent = DiagnosisAgent()
        self.recommendation_agent = RecommendationAgent()
        self.followup_manager = FollowUpManager()

        # The knowledge base (optional)
        self.knowledge_base = getattr(self.diagnosis_agent, "kb_lookup", {})

        # ✅ Simplified: No longer loads or generates follow-up questions here
        print("✅ Chatbot agents initialized successfully.")
        print("-" * 50)

    def start_chat(self):
        print("👋 Welcome to the Advanced AI Healthcare Chatbot!")
        print("Please describe your symptoms in a full sentence.")
        print("Type 'exit' anytime to quit.")
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
            self.symptom_agent.collect_symptoms(user_input)
            symptoms_text = self.symptom_agent.get_symptom_text()

            if not self.symptom_agent.collected_symptoms:
                print("Bot: I couldn't recognize specific symptoms yet. Can you describe them differently or add more details?")
                print("-" * 50)
                continue

            # 2️⃣ Predict diseases
            predictions = self.diagnosis_agent.predict_top_diseases(
                symptoms_text,
                followup_manager=self.followup_manager
            )

            print("\n🤖 Bot: Based on your symptoms, here are possible conditions:")
            if not predictions:
                print("I couldn't identify specific conditions based on that. Could you provide more details?")
                print("-" * 50)
                continue

            # 3️⃣ Ask follow-up questions per disease
            for pred in predictions:
                disease = pred["disease"]
                prob = pred["probability"]
                print(f"\n- {disease} (Initial Likelihood: {prob:.2%})")

                if self.followup_manager.has_followup_questions(disease):
                    print(f"\n🩻 Let's confirm some details about {disease}...")

                    while self.followup_manager.has_followup_questions(disease):
                        question_obj = self.followup_manager.get_next_question_for_disease(disease)
                        if not question_obj:
                            break

                        q_text = question_obj["text"]
                        q_id = question_obj["id"]

                        answer = ""
                        while answer not in ["yes", "no"]:
                            raw_answer = input(f"🤖 {q_text} (yes/no): ").strip()
                            answer = self.followup_manager._normalize_answer(raw_answer)
                            if answer not in ["yes", "no"]:
                                print("Please answer with 'yes' or 'no' (or similar phrases).")

                        self.followup_manager.record_answer(q_id, raw_answer)

                    score = self.followup_manager.get_followup_score(disease)
                    print(f"✅ Updated confidence score for {disease} based on answers: {score:.2%}")

            # 4️⃣ Re-evaluate predictions after follow-up
            print("\n🧠 Re-evaluating based on your answers...")
            updated_predictions = self.diagnosis_agent.predict_top_diseases(
                symptoms_text,
                followup_manager=self.followup_manager
            )

            print("\n🤖 Bot: Updated possible conditions:")
            if not updated_predictions:
                print("After follow-up, I still need more details to narrow down the conditions.")
                print("-" * 50)
                continue

            for pred in updated_predictions:
                disease = pred["disease"]
                prob = pred["probability"]
                print(f"\n- {disease} (Final Likelihood: {prob:.2%})")

                rec = self.recommendation_agent.get_details(disease)
                precautions = ", ".join(rec.get("precautions", ["Consult a doctor for advice."]))
                description = rec.get("description", "No detailed description available.")

                print(f"    📋 Description: {description}")
                print(f"    🛡️ Precautions: {precautions}")

            print("\n" + "-" * 50)

    # --- Functions for testing ---
    def process_user_input(self, user_input: str):
        if hasattr(self, "chat"):
            return self.chat(user_input)
        elif hasattr(self, "handle_input"):
            return self.handle_input(user_input)
        else:
            print(f"[ChatbotManager.process_user_input] Simulating response for: {user_input}")
            self.symptom_agent.collect_symptoms(user_input)
            symptoms_text = self.symptom_agent.get_symptom_text()
            if not self.symptom_agent.collected_symptoms:
                return "I couldn't recognize symptoms in that input."

            predictions = self.diagnosis_agent.predict_top_diseases(symptoms_text, followup_manager=self.followup_manager)
            if predictions:
                top_disease = predictions[0]['disease']
                return f"Based on '{symptoms_text}', I suspect {top_disease}. Any other symptoms?"
            return "I received your input. How else can I help?"

    def get_session_summary(self):
        fm = self.followup_manager
        answers_dict = getattr(fm, "user_answers", {})
        pending_list = getattr(fm, "pending_questions", [])
        boosts_dict = getattr(fm, "disease_boosts", {})

        summary = {
            "total_questions_asked": len(answers_dict),
            "pending_questions": len(pending_list),
            "boosted_diseases": list(boosts_dict.keys()),
        }
        print(f"[ChatbotManager.get_session_summary] {summary}")
        return summary


if __name__ == "__main__":
    manager = ChatbotManager()
    manager.start_chat()
