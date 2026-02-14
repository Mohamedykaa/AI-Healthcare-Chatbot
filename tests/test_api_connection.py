import pytest
from fastapi.testclient import TestClient
from api import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("api.process_chat_message")
def test_chat_endpoint_success(mock_process):
    # Mock successful response
    mock_process.return_value = ("Test response", "ROUTINE", "")
    
    payload = {
        "message": "Hello",
        "history": []
    }
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Test response"
    assert data["risk_level"] == "ROUTINE"

@patch("api.process_chat_message")
def test_chat_endpoint_error(mock_process):
    # Mock exception
    mock_process.side_effect = Exception("Internal Error")
    
    payload = {
        "message": "Crash me",
        "history": []
    }
    # FastAPI handles exceptions, usually returning 500 if unhandled or caught by exception handler
    response = client.post("/chat", json=payload)
    assert response.status_code == 500
    assert "Internal Error" in response.json()["detail"]

def test_chat_emergency_Mock():
    # Test emergency flow directly if possible, or mock logic
    pass
