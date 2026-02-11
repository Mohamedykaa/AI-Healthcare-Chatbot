"""
Diagnostic script to test if the Ollama model can generate any output.
This bypasses the full RAG chain to isolate the LLM issue.
"""
import asyncio
from langchain_community.chat_models import ChatOllama

async def test_ollama():
    print("=" * 60)
    print("OLLAMA MODEL DIAGNOSTIC TEST")
    print("=" * 60)
    
    LLM_MODEL = "cniongolo/biomistral"
    
    try:
        print(f"\n[1] Connecting to Ollama with model: {LLM_MODEL}...")
        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.1,
            num_predict=100,  # Very short response
        )
        print("    ✓ Connection established.")
        
        print("\n[2] Sending simple test prompt...")
        prompt = "Say hello and tell me what you are."
        
        response = await llm.ainvoke(prompt)
        
        print("\n[3] Response received:")
        print("-" * 40)
        print(response.content if hasattr(response, 'content') else str(response))
        print("-" * 40)
        
        if response.content:
            print("\n✅ SUCCESS: Model is generating output!")
        else:
            print("\n❌ FAILURE: Model returned empty response!")
            
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        print("\nThis suggests the Ollama model is not working correctly.")
        print("Recommendation: Try a different model like 'mistral' or 'llama3'.")

if __name__ == "__main__":
    asyncio.run(test_ollama())
