import logging
from contextlib import asynccontextmanager
from typing import List, Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.logic import initialize_components, process_chat_message

# Load environment variables (e.g. CHROMA_PERSIST_DIR override)
load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================
# API MODELS
# ============================================================


class ChatMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMsg] = Field(default_factory=list)


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
        logger.error("Startup Error: %s", e)
        # Re-raise to prevent the app from starting in a broken state
        raise
    yield
    # Cleanup if needed


app = FastAPI(title="Medical Chatbot API", lifespan=lifespan)

# ============================================================
# CORS - Allow Flutter / mobile apps to connect
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# PROCESS REQUEST
# ============================================================


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert Pydantic models to dicts/objects expected by core.
        formatted_history = [msg.model_dump() for msg in request.history]

        response_text, risk_level, sources_text = await process_chat_message(
            request.message,
            formatted_history,
        )

        return ChatResponse(
            response=response_text,
            risk_level=risk_level,
            sources=sources_text,
        )
    except Exception as e:
        logger.error("Chat endpoint error: %s", e)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        )


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
