from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from contextlib import asynccontextmanager
from src.core.logic import initialize_components, process_chat_message

# ============================================================
# API MODELS
# ============================================================

class ChatMsg(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMsg] = []

class ChatResponse(BaseModel):
    response: str
    risk_level: str
    sources: str

# ============================================================
# APP LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and DB on startup
    try:
        initialize_components()
    except Exception as e:
        print(f"Startup Error: {e}")
        # Re-raise to prevent the app from starting in a broken state
        raise e
    yield
    # Cleanup if needed

app = FastAPI(title="Medical Chatbot API", lifespan=lifespan)

# ============================================================
# PROCESS REQUEST
# ============================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert Pydantic models to dicts/objects expected by core
        # core.process_chat_message expects a list of objects or dicts
        # We pass the list of dicts directly
        formatted_history = [msg.model_dump() for msg in request.history]
        
        response_text, risk_level, sources_text = await process_chat_message(
            request.message, 
            formatted_history
        )
        
        return ChatResponse(
            response=response_text,
            risk_level=risk_level,
            sources=sources_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
