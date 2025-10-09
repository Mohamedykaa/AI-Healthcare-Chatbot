# chatbot_system/symptom_agent.py

class SymptomAgent:
    """
    This agent is responsible for collecting and summarizing
    the user's symptom descriptions. It prepares the text for
    the DiagnosisAgent.
    """
    def __init__(self):
        self.raw_user_input = ""

    def collect_symptoms(self, user_input: str) -> str:
        """
        Takes raw user input as a full sentence or keywords.
        Stores it for later processing by the DiagnosisAgent.
        """
        self.raw_user_input = user_input.strip()
        return self.raw_user_input

    def get_symptom_text(self) -> str:
        """Returns the collected symptom text."""
        if not self.raw_user_input:
            return "No symptoms recorded yet."
        return self.raw_user_input
