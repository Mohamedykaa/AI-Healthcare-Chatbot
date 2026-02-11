
import asyncio
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.infrastructure.ai.llm_provider import BioMistralProvider
from app.domain.entities import ChatSession, Message, MessageRole
from app.core.config import settings

async def benchmark():
    print("--- LLM Benchmark Start ---")
    
    # Force Sync Mode to simulate local behavior if needed, but Provider handles logic
    print(f"FORCE_SYNC_MODE: {settings.FORCE_SYNC_MODE}")
    print(f"Model Path: {settings.MODEL_PATH_BIOMISTRAL}")
    
    # 1. Load Model
    start_load = time.time()
    print("Initializing Provider (Loading Model)...")
    provider = BioMistralProvider()
    end_load = time.time()
    print(f"Model Load Time: {end_load - start_load:.2f} seconds")
    
    if not provider._model:
        print("❌ Model failed to load.")
        return

    # 2. Extract Symptoms Benchmark
    text = "I have a severe headache and high fever for 3 days."
    print(f"\nBenchmarking analyze_symptoms with text: '{text}'")
    start_symptom = time.time()
    symptoms = await provider.analyze_symptoms(text)
    end_symptom = time.time()
    print(f"Result: {symptoms}")
    print(f"Extraction Time: {end_symptom - start_symptom:.2f} seconds")

    # 3. Generation Benchmark
    print("\nBenchmarking generate_response...")
    import uuid
    session = ChatSession(
        id=uuid.uuid4(),
        patient_id="test_patient",
        messages=[
            Message(role=MessageRole.USER, content="I have a headache and fever. What should I do?")
        ]
    )
    
    start_gen = time.time()
    response = await provider.generate_response(session)
    end_gen = time.time()
    print(f"Response: {response.content}")
    print(f"Generation Time: {end_gen - start_gen:.2f} seconds")
    
    print("--- Benchmark Complete ---")

if __name__ == "__main__":
    asyncio.run(benchmark())
