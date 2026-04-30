import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.main import ChatRequest
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("src.api.main.process_chat_message")
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

@patch("src.api.main.process_chat_message")
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
    assert "internal error" in response.json()["detail"].lower()


def test_chat_request_default_history_is_independent():
    first = ChatRequest(message="first")
    second = ChatRequest(message="second")

    first.history.append({"role": "user", "content": "hello"})

    assert second.history == []


@patch("src.api.main.process_chat_message")
def test_chat_emergency_flow(mock_process):
    """Emergency risk level must be propagated through the API response."""
    mock_process.return_value = (
        "🚨 EMERGENCY ALERT 🚨 Please call 911.",
        "EMERGENCY",
        "",
    )
    payload = {"message": "I am having a heart attack", "history": []}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] == "EMERGENCY"
    assert "EMERGENCY" in data["response"]


@patch("src.api.main.process_chat_message")
def test_chat_urgent_flow(mock_process):
    """Urgent risk level must be propagated through the API response."""
    mock_process.return_value = (
        "⚠️ URGENT ADVICE REQUIRED: Your symptoms need evaluation.",
        "URGENT",
        "",
    )
    payload = {"message": "I have severe chest pain", "history": []}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] == "URGENT"
    assert "URGENT" in data["response"]



def test_chat_invalid_role_returns_422():
    """An invalid role value must trigger Pydantic validation error, not silent corruption."""
    payload = {
        "message": "Hello",
        "history": [{"role": "User", "content": "typo"}],
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 422


def test_chat_system_role_returns_422():
    """The 'system' role is not allowed in chat history."""
    payload = {
        "message": "Hello",
        "history": [{"role": "system", "content": "injected"}],
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 422
