import os
import re
import shutil
from typing import List, Optional
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
MEDICAL_KNOWLEDGE_FILES = [
    "data/medical_knowledge_medquad.txt",
    "data/medical_knowledge_medmcqa.txt",
    "data/medical_knowledge_public_health.txt",
]
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3:8b")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
RETRIEVER_K = int(os.environ.get("RETRIEVER_K", "6"))
RETRIEVER_SCORE_THRESHOLD = float(os.environ.get("RETRIEVER_SCORE_THRESHOLD", "0.3"))
CONTEXT_CHAR_LIMIT_PER_DOC = int(os.environ.get("CONTEXT_CHAR_LIMIT_PER_DOC", "400"))

# Risk/emergency logic lives in backend.risk (dependency-free, shared with tests)
from backend.risk import assess_risk_level, get_emergency_response

# ============================================================
# LOGIC
# ============================================================

def get_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def validate_chroma_db(vectorstore: Chroma) -> bool:
    try:
        results = vectorstore.similarity_search("health", k=1)
        collection = vectorstore._collection
        return collection.count() > 0
    except Exception as e:
        print(f"ChromaDB validation failed: {e}")
        return False

def normalize_text(text: str) -> str:
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def load_or_create_vectorstore(embedding_function: HuggingFaceEmbeddings) -> Chroma:
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Found existing ChromaDB at {CHROMA_PERSIST_DIR}")
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embedding_function
            )
            if validate_chroma_db(vectorstore):
                print("ChromaDB loaded and validated successfully")
                return vectorstore
            else:
                print("ChromaDB validation failed, recreating...")
                shutil.rmtree(CHROMA_PERSIST_DIR)
        except Exception as e:
            print(f"Error loading ChromaDB: {e}")
            if os.path.exists(CHROMA_PERSIST_DIR):
                shutil.rmtree(CHROMA_PERSIST_DIR)
    
    print("Loading medical knowledge from multiple sources...")
    all_documents = []
    for filepath in MEDICAL_KNOWLEDGE_FILES:
        if os.path.exists(filepath):
            print(f"   Loading: {filepath}")
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.page_content = normalize_text(doc.page_content)
                doc.metadata["source"] = filepath
            all_documents.extend(docs)
        else:
            print(f"   Not found (skip): {filepath}")
    
    if not all_documents:
        raise FileNotFoundError("No medical knowledge files found.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n---\n", "\n\n", "\n", ". ", " ", ""]
    )
    splits = text_splitter.split_documents(all_documents)
    
    BATCH_SIZE = 5000
    if len(splits) <= BATCH_SIZE:
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function, persist_directory=CHROMA_PERSIST_DIR)
    else:
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_function)
        for i in range(0, len(splits), BATCH_SIZE):
            batch = splits[i:i + BATCH_SIZE]
            vectorstore.add_documents(batch)
    
    vectorstore.persist()
    print(f"ChromaDB created and persisted at {CHROMA_PERSIST_DIR}")
    return vectorstore

def get_llm() -> ChatOllama:
    try:
        return ChatOllama(model=LLM_MODEL, temperature=0.3, num_predict=2048)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChatOllama: {e}")

def format_sources(context_documents: list) -> str:
    if not context_documents: return ""
    sources = set()
    for doc in context_documents:
        source = doc.metadata.get("source", "Medical Knowledge Base")
        sources.add(os.path.basename(source))
    if not sources: return ""
    return "\n\n---\n**📚 References:** " + ", ".join(sorted(sources))

# Global singletons
_EMBEDDING_FUNCTION = None
_VECTORSTORE = None
_LLM = None

def initialize_components(max_retries: int = 2) -> None:
    global _EMBEDDING_FUNCTION, _VECTORSTORE, _LLM
    if _VECTORSTORE is not None and _LLM is not None: return

    for attempt in range(1, max_retries + 1):
        try:
            _EMBEDDING_FUNCTION = get_embedding_function()
            _VECTORSTORE = load_or_create_vectorstore(_EMBEDDING_FUNCTION)
            _LLM = get_llm()
            print("Medical Chatbot initialization complete!")
            return
        except Exception as exc:
            print(f"Initialization attempt {attempt}/{max_retries} failed: {exc}")
    
    raise RuntimeError("Medical Chatbot initialization failed.")

async def process_chat_message(user_input: str, chat_history: list):
    """
    Main logic to process a chat message.
    Returns: (response_text, risk_level, sources_text)
    """
    risk_level = assess_risk_level(user_input)
    if risk_level == "EMERGENCY":
        return get_emergency_response(), risk_level, ""

    if not _LLM or not _VECTORSTORE:
        initialize_components()

    retriever = _VECTORSTORE.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": RETRIEVER_K, "score_threshold": RETRIEVER_SCORE_THRESHOLD}
    )
    docs = retriever.invoke(user_input)
    context_text = "\n".join(
        [doc.page_content[:CONTEXT_CHAR_LIMIT_PER_DOC] for doc in docs[:RETRIEVER_K]]
    )
    
    history_text = ""
    if chat_history:
        for msg_item in chat_history[-8:]:
            if isinstance(msg_item, dict): # For API
                role = "User" if msg_item.get("role") == "user" else "Assistant"
                content = msg_item.get("content", "")
            else: # For LangChain objects
                role = "User" if isinstance(msg_item, HumanMessage) else "Assistant"
                content = msg_item.content
            history_text += f"{role}: {content}\n"

    turn_count = len(chat_history) // 2
    if turn_count == 0:
        prompt = f"""You are an educational medical symptom checker.
You talk to users who describe how they feel.
You respond by explaining symptoms in simple medical terms.

You usually say what such symptoms are commonly related to, mention 2-3 possible conditions in a cautious way, and ask a few short questions to differentiate between them.

You do NOT name a single disease yet.

Medical context: {context_text[:800]}

User: {user_input}"""
    else:
        prompt = f"""You are an educational medical symptom checker.
You are continuing a conversation with a user about their symptoms.
You use the conversation history to narrow down the possibilities.

If you have enough information from the user's answers, you may say: "Based on your symptoms and answers, the most likely condition is..." and explain why it fits better than alternatives.

CRITICAL INSTRUCTIONS:
- Use medically cautious, uncertainty-aware language when evidence is limited.
- Avoid vague hedging like "I guess" or "maybe", but use precise language like "This pattern is consistent with..."
- If confidence is limited, state this clearly and suggest the safest next steps.
- Do NOT ask for feedback (e.g., "Does this sound right?") after giving your assessment.
- Do NOT say "This is just a guess".

If you still need more information to be sure, ask 1-2 focused follow-up questions.

{f"Conversation so far:{chr(10)}{history_text}" if history_text else ""}

User: {user_input}"""

    response = await _LLM.ainvoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)

    if not answer or len(answer.strip()) < 10 or answer.strip().startswith("You are") or "You match symptoms" in answer:
        answer = (
            "I'm sorry you're not feeling well. "
            "Based on general medical discussions, your symptoms are often related "
            "to common upper respiratory conditions or viral infections.\n\n"
            "(Response generated via fallback safe-mode due to model load)"
        )

    answer = answer.replace("the patient", "you").replace("The patient", "You")
    sources_text = format_sources(docs)
    
    urgent_prefix = ""
    if risk_level == "URGENT":
        urgent_prefix = "⚠️ **URGENT ADVICE REQUIRED:** Your symptoms may need same-day medical evaluation.\n\n"
    
    final_response = urgent_prefix + answer
    
    return final_response, risk_level, sources_text
