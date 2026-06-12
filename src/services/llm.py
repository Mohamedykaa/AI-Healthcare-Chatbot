import logging

from langchain_community.chat_models import ChatOllama

from src.core.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TIMEOUT,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

class FallbackLLM:
    """Wrapper that tries primary LLM and falls back to secondary LLM on failure."""
    def __init__(self, primary_llm, fallback_llm):
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm

    async def ainvoke(self, messages, **kwargs):
        import asyncio
        logger.info("Using Gemini provider")
        try:
            # Force a maximum wait time so Langchain's exponential backoff doesn't hang the app
            return await asyncio.wait_for(self.primary_llm.ainvoke(messages, **kwargs), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Gemini timeout (likely quota backoff). Switching to Ollama.")
            try:
                result = await self.fallback_llm.ainvoke(messages, **kwargs)
                logger.info("Fallback provider activated successfully.")
                return result
            except Exception as fallback_e:
                logger.error("Both Gemini and Ollama unavailable.")
                raise RuntimeError("Both Gemini and Ollama unavailable.") from fallback_e
        except Exception as e:
            error_str = str(e).lower()
            if any(term in error_str for term in ["quota", "resourceexhausted", "rate limit", "429", "timeout", "serviceunavailable", "service unavailable"]):
                logger.warning("Gemini quota exceeded. Switching to Ollama.")
            else:
                logger.warning("Gemini unavailable. Switching to Ollama.")
                
            try:
                result = await self.fallback_llm.ainvoke(messages, **kwargs)
                logger.info("Fallback provider activated successfully.")
                return result
            except Exception as fallback_e:
                logger.error("Both Gemini and Ollama unavailable.")
                raise RuntimeError("Both Gemini and Ollama unavailable.") from fallback_e

    def invoke(self, messages, **kwargs):
        logger.info("Using Gemini provider")
        try:
            return self.primary_llm.invoke(messages, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if any(term in error_str for term in ["quota", "resourceexhausted", "rate limit", "429", "timeout", "serviceunavailable", "service unavailable"]):
                logger.warning("Gemini quota exceeded. Switching to Ollama.")
            else:
                logger.warning("Gemini unavailable. Switching to Ollama.")
                
            try:
                result = self.fallback_llm.invoke(messages, **kwargs)
                logger.info("Fallback provider activated successfully.")
                return result
            except Exception as fallback_e:
                logger.error("Both Gemini and Ollama unavailable.")
                raise RuntimeError("Both Gemini and Ollama unavailable.") from fallback_e


class LLMFactory:
    @staticmethod
    def create():
        """Create and return the configured LLM instance based on LLM_PROVIDER."""
        ollama_llm = None
        try:
            ollama_llm = ChatOllama(model=LLM_MODEL, temperature=0.3, num_predict=2048)
        except Exception as exc:
            logger.error("Failed to initialise ChatOllama fallback: %s", exc)

        if LLM_PROVIDER == "gemini":
            if not GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY is not set. Switching to Ollama.")
                if ollama_llm:
                    return ollama_llm
                raise RuntimeError("GEMINI_API_KEY is required and Ollama fallback failed.")
            
            from langchain_google_genai import ChatGoogleGenerativeAI
            try:
                gemini_llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.3,
                    max_retries=0,  # Set to 0 so exceptions are immediately caught by our fallback wrapper
                    timeout=LLM_TIMEOUT,
                )
                logger.info("ChatGoogleGenerativeAI initialised (model=%s)", GEMINI_MODEL)
                
                # Wrap it with our fallback logic
                if ollama_llm:
                    return FallbackLLM(primary_llm=gemini_llm, fallback_llm=ollama_llm)
                return gemini_llm
            except Exception as exc:
                logger.error("Failed to initialise ChatGoogleGenerativeAI: %s", exc)
                if ollama_llm:
                    logger.warning("Gemini unavailable during init. Switching to Ollama.")
                    return ollama_llm
                raise RuntimeError(f"Failed to initialize ChatGoogleGenerativeAI: {exc}") from exc

        # Default fallback: Ollama
        if ollama_llm:
            logger.info("ChatOllama initialised (model=%s)", LLM_MODEL)
            return ollama_llm
            
        raise RuntimeError("Failed to initialize any LLM provider.")


def get_llm():
    """Backward compatible function to return the configured LLM instance."""
    return LLMFactory.create()
