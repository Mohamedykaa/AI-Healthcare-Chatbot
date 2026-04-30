import logging

from langchain_community.chat_models import ChatOllama

from src.core.config import LLM_MODEL, LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def get_llm() -> ChatOllama:
    """Initialise and return the ChatOllama instance.

    Raises RuntimeError on failure.
    """
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0.3, num_predict=2048)
        logger.info("ChatOllama initialised (model=%s)", LLM_MODEL)
        return llm
    except Exception as exc:
        logger.error("Failed to initialise ChatOllama: %s", exc)
        raise RuntimeError(f"Failed to initialize ChatOllama: {exc}") from exc
