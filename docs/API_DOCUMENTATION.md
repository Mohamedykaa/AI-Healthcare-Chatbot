# API Documentation

## REST API (`src/api/main.py`)

A FastAPI server exposing the medical chatbot as a REST API.
This is the **headless** interface — use it for programmatic access, mobile integrations (e.g. Flutter), or when you don't need the Chainlit UI.

### Start the server

```bash
pip install -r requirements_api.txt    # if not already installed
python run_api.py
# OR
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

The API will be available at `http://localhost:8001`.

> **Note:** Chainlit runs on port 8000, the API runs on port 8001. Both share the same backend logic in `src/core/logic.py`.

---

## Endpoints

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

---

### `POST /chat`

Send a user message through the RAG pipeline and receive the chatbot's response.

**Request body:**
```json
{
  "message": "I have a headache and feel dizzy",
  "history": [
    { "role": "user", "content": "I have been feeling tired" },
    { "role": "assistant", "content": "I understand you're feeling tired..." }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | `string` | Yes | The user's current message |
| `history` | `list[ChatMsg]` | No | Previous conversation turns (default: `[]`) |

Each `ChatMsg` has:
| Field | Type | Description |
|-------|------|-------------|
| `role` | `"user"` \| `"assistant"` | One of exactly `"user"` or `"assistant"` (validated; other values return 422) |
| `content` | `string` | The message content |

**Response:**
```json
{
  "response": "Based on your symptoms, this pattern could be consistent with...",
  "risk_level": "ROUTINE",
  "sources": "\n\n---\n**📚 References:** medical_knowledge_medquad.txt"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `response` | `string` | The chatbot's answer (may include urgent prefix for URGENT risk) |
| `risk_level` | `string` | `"ROUTINE"`, `"URGENT"`, or `"EMERGENCY"` |
| `sources` | `string` | Source citations (empty string if no sources retrieved) |

**Risk level behavior:**
- `ROUTINE` — Normal RAG pipeline response
- `URGENT` — Response is prefixed with an urgent warning banner
- `EMERGENCY` — LLM is **not called**; a deterministic safety response is returned immediately

**Error response (500):**
```json
{
  "detail": "An internal error occurred. Please try again."
}
```

> **Note:** For security, the API returns a generic error message. Detailed errors are logged server-side only.

---

## Usage Examples

### Python
```python
import requests

# First message
response = requests.post("http://localhost:8001/chat", json={
    "message": "I have been feeling very tired with frequent headaches",
    "history": []
})
data = response.json()
print(data["response"])
print(data["risk_level"])

# Follow-up (with history)
response2 = requests.post("http://localhost:8001/chat", json={
    "message": "Yes, I also feel cold all the time",
    "history": [
        {"role": "user", "content": "I have been feeling very tired with frequent headaches"},
        {"role": "assistant", "content": data["response"]}
    ]
})
print(response2.json()["response"])
```

### cURL
```bash
# Simple message
curl -X POST "http://localhost:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the symptoms of diabetes?", "history": []}'

# Health check
curl http://localhost:8001/health
```

---

## CORS

CORS is enabled with `allow_origins=["*"]` for development. In production, restrict this to specific origins.

---

## Architecture

The API shares the same backend as the Chainlit UI:

```
run_api.py → src/api/main.py → src/core/logic.py → (risk.py, llm.py, vectorstore.py)
```

Both entry points call `process_chat_message()` from `src/core/logic.py`, ensuring identical behavior regardless of interface.
