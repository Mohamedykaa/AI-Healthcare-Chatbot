"""
Diagnostic script to test the RAG chain specifically.
This tests if context is being correctly injected.
"""
import asyncio
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Config
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "cniongolo/biomistral"

SYSTEM_PROMPT = """You are a medical assistant."""

USER_INSTRUCTIONS = """
Instructions: Analyze the symptoms and ask follow-up questions.

Context:
{context}

User Question:
{input}
"""

async def test_rag_chain():
    print("=" * 60)
    print("RAG CHAIN DIAGNOSTIC TEST")
    print("=" * 60)
    
    try:
        # 1. Load embeddings
        print("\n[1] Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("    ✓ Embeddings loaded.")
        
        # 2. Load vectorstore
        print("\n[2] Loading vectorstore...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        print(f"    ✓ Vectorstore loaded. Collection count: {vectorstore._collection.count()}")
        
        # 3. Create LLM
        print("\n[3] Creating LLM...")
        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.1,
            num_predict=500,
        )
        print("    ✓ LLM created.")
        
        # 4. Create simple RAG chain (no history)
        print("\n[4] Creating simple RAG chain...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_INSTRUCTIONS),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("    ✓ RAG chain created.")
        
        # 5. Test query
        print("\n[5] Testing with query: 'I have a fever and sore throat'...")
        response = await rag_chain.ainvoke({
            "input": "I have a fever and a sore throat."
        })
        
        print("\n[6] Response:")
        print("-" * 40)
        answer = response.get("answer", "NO ANSWER KEY")
        print(f"Answer: {answer[:500] if answer else 'EMPTY'}")
        print("-" * 40)
        
        # Debug: Show context
        print("\n[7] Retrieved Context Documents:")
        for i, doc in enumerate(response.get("context", [])):
            print(f"  Doc {i+1}: {doc.page_content[:100]}...")
        
        if answer:
            print("\n✅ SUCCESS: RAG chain is working!")
        else:
            print("\n❌ FAILURE: RAG chain returned empty answer!")
            
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rag_chain())
