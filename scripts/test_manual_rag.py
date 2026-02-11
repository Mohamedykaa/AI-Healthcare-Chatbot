"""
Diagnostic script to test manual RAG approach (bypassing create_stuff_documents_chain).
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Config
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "cniongolo/biomistral"

async def test_manual_rag():
    print("=" * 60)
    print("MANUAL RAG DIAGNOSTIC TEST")
    print("=" * 60)
    
    try:
        # 1. Load embeddings
        print("\n[1] Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 2. Load vectorstore
        print("\n[2] Loading vectorstore...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        
        # 3. Create LLM
        print("\n[3] Creating LLM...")
        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.1,
            num_predict=500,
        )
        
        # 4. Retrieve documents manually
        user_input = "I have a fever and a sore throat."
        print(f"\n[4] Retrieving docs for: '{user_input}'...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(user_input)
        
        context_text = "\n\n".join([doc.page_content for doc in docs])
        print(f"    Retrieved {len(docs)} documents.")
        print(f"    Context preview: {context_text[:200]}...")
        
        # 5. Build prompt manually
        print("\n[5] Building prompt manually...")
        prompt = f"""You are a medical assistant.

Based on the following medical context, analyze the patient's symptoms and ask follow-up questions.

Context:
{context_text[:1000]}

Patient Question: {user_input}

Your analysis:"""
        
        print(f"    Prompt length: {len(prompt)} chars")
        
        # 6. Call LLM directly
        print("\n[6] Calling LLM directly...")
        response = await llm.ainvoke(prompt)
        
        print("\n[7] Response:")
        print("-" * 40)
        answer = response.content if hasattr(response, 'content') else str(response)
        print(answer if answer else "EMPTY")
        print("-" * 40)
        
        if answer:
            print("\n✅ SUCCESS: Manual RAG is working!")
        else:
            print("\n❌ FAILURE: Still empty!")
            
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_manual_rag())
