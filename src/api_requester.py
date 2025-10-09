import requests
from typing import Dict

class APIRequester:
    """
    A helper class to communicate with the FastAPI backend.
    """

    def __init__(self, api_url="http://127.0.0.1:8000/predict"):
        self.api_url = api_url

    def send_request(self, symptoms_text: str) -> Dict:
        """
        Sends the user's symptom text to the FastAPI endpoint.
        """
        try:
            payload = {"text": symptoms_text}
            response = requests.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"API connection failed: {e}"}
