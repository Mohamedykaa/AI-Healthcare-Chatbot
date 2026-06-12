import asyncio
import logging
import sys
from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded
from src.services.llm import LLMFactory

# Setup logging to capture output
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s: %(message)s')
logger = logging.getLogger("src.services.llm")
logger.setLevel(logging.INFO)

async def run_tests():
    print("==================================================")
    print("TEST 1: Gemini Available")
    print("==================================================")
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.ainvoke", new_callable=AsyncMock) as mock_gemini:
        mock_gemini.return_value = HumanMessage(content="[Gemini] I am working fine.")
        llm = LLMFactory.create()
        res = await llm.ainvoke([HumanMessage(content="Hello")])
        print(f"Response: {res.content}\n")

    print("==================================================")
    print("TEST 2: Invalid Gemini API Key")
    print("==================================================")
    with patch("src.services.llm.GEMINI_API_KEY", ""):
        try:
            llm_no_key = LLMFactory.create()
            print("Successfully fell back to Ollama during init.")
        except Exception as e:
            print("Failed:", e)
    print()

    print("==================================================")
    print("TEST 3: Quota Exceeded")
    print("==================================================")
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.ainvoke", new_callable=AsyncMock) as mock_gemini:
        with patch("langchain_community.chat_models.ChatOllama.ainvoke", new_callable=AsyncMock) as mock_ollama:
            mock_gemini.side_effect = ResourceExhausted("429 Quota exceeded for metric: generate_content_free_tier_requests")
            mock_ollama.return_value = HumanMessage(content="[Ollama] I am your fallback.")
            
            llm = LLMFactory.create()
            res = await llm.ainvoke([HumanMessage(content="Hello")])
            print(f"Response: {res.content}\n")

    print("==================================================")
    print("TEST 4: Network Timeout")
    print("==================================================")
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.ainvoke", new_callable=AsyncMock) as mock_gemini:
        with patch("langchain_community.chat_models.ChatOllama.ainvoke", new_callable=AsyncMock) as mock_ollama:
            mock_gemini.side_effect = DeadlineExceeded("Timeout while contacting API")
            mock_ollama.return_value = HumanMessage(content="[Ollama] I am your fallback.")
            
            llm = LLMFactory.create()
            res = await llm.ainvoke([HumanMessage(content="Hello")])
            print(f"Response: {res.content}\n")

    print("==================================================")
    print("TEST 5: Arabic Query (with fallback)")
    print("==================================================")
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.ainvoke", new_callable=AsyncMock) as mock_gemini:
        with patch("langchain_community.chat_models.ChatOllama.ainvoke", new_callable=AsyncMock) as mock_ollama:
            mock_gemini.side_effect = ResourceExhausted("429 Quota exceeded")
            mock_ollama.return_value = HumanMessage(content="[Ollama] مرحبا! أنا أعمل كبديل.")
            
            llm = LLMFactory.create()
            res = await llm.ainvoke([HumanMessage(content="أعاني من صداع ودوخة")])
            print(f"Response: {res.content}\n")

if __name__ == "__main__":
    asyncio.run(run_tests())
