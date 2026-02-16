from langchain_community.chat_models import ChatOllama
from src.core.config import LLM_MODEL

def get_llm() -> ChatOllama:
    try:
        return ChatOllama(model=LLM_MODEL, temperature=0.3, num_predict=2048)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChatOllama: {e}")
