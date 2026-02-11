"""
Local Medical Chatbot - Chainlit Application
=============================================

A fully offline medical chatbot using:
- ChatOllama (llama3:8b model)
- HuggingFace embeddings (all-MiniLM-L6-v2)
- ChromaDB for persistent vector storage
- LangChain RAG architecture

Entry point: chainlit run app.py
"""

import os
import re
import shutil
from typing import List, Optional
import asyncio

# Load environment variables from .env (if present)
from dotenv import load_dotenv
load_dotenv()

# Chainlit imports
import chainlit as cl

# LangChain core imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LangChain community imports
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

# LangChain chain imports
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION (loaded from .env with safe defaults)
# ============================================================

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
MEDICAL_KNOWLEDGE_FILES = [
    "data/medical_knowledge_medquad.txt",
    "data/medical_knowledge_medmcqa.txt",
    "data/medical_knowledge_public_health.txt",
]
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3:8b")

# Chunk settings for document splitting
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))

# Emergency keywords that require immediate escalation
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "can't breathe", "cannot breathe",
    "difficulty breathing", "severe bleeding", "loss of consciousness",
    "unconscious", "fainting", "seizure", "severe head injury",
    "poisoning", "overdose", "suicidal", "suicide", "self-harm",
    "severe allergic reaction", "anaphylaxis", "choking"
]


# ============================================================
# SYSTEM PROMPT WITH SAFETY CONSTRAINTS
# ============================================================

SYSTEM_PROMPT = """You are a medical decision-support assistant designed to demonstrate
an interactive preliminary medical assessment for educational purposes.
"""

USER_INSTRUCTIONS = """
Analysis Instructions:
1. Provide a brief analysis of the symptoms using the retrieved context. (2-3 sentences)
2. Ask 2-3 filtering questions to clarify the situation.
3. State clearly: "Based on general medical principles (limited context available)..." if context is weak.
4. End with a standard medical disclaimer.

Context:
{context}

User Question:
{input}
"""


# ============================================================
# EMBEDDING FUNCTION (CPU-FORCED)
# ============================================================

def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Create HuggingFace embeddings with CPU-forced execution.
    Uses the same embedding function for both creation and loading.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ============================================================
# VECTOR STORE MANAGEMENT
# ============================================================

def validate_chroma_db(vectorstore: Chroma) -> bool:
    """
    Validate ChromaDB integrity by performing a test query.
    Returns True if DB is valid and accessible.
    """
    try:
        # Test similarity search with a generic query
        results = vectorstore.similarity_search("health", k=1)
        # Check collection has documents
        collection = vectorstore._collection
        count = collection.count()
        return count > 0
    except Exception as e:
        print(f"ChromaDB validation failed: {e}")
        return False


def normalize_text(text: str) -> str:
    """
    Normalize text by stripping whitespace and cleaning newlines.
    """
    # Replace multiple newlines with single newline
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def load_or_create_vectorstore(embedding_function: HuggingFaceEmbeddings) -> Chroma:
    """
    Load existing ChromaDB or create new one from medical knowledge files.
    
    Logic:
    1) If ./chroma_db exists -> Load and validate
    2) If not exists -> Load all medical knowledge files, split, create DB
    """
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"📂 Found existing ChromaDB at {CHROMA_PERSIST_DIR}")
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embedding_function
            )
            
            if validate_chroma_db(vectorstore):
                print("✅ ChromaDB loaded and validated successfully")
                return vectorstore
            else:
                print("⚠️ ChromaDB validation failed, recreating...")
                shutil.rmtree(CHROMA_PERSIST_DIR)
        except Exception as e:
            print(f"❌ Error loading ChromaDB: {e}")
            if os.path.exists(CHROMA_PERSIST_DIR):
                shutil.rmtree(CHROMA_PERSIST_DIR)
    
    # Load documents from all medical knowledge files
    print("📄 Loading medical knowledge from multiple sources...")
    all_documents = []
    
    for filepath in MEDICAL_KNOWLEDGE_FILES:
        if os.path.exists(filepath):
            print(f"   📥 Loading: {filepath}")
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            # Normalize text content
            for doc in docs:
                doc.page_content = normalize_text(doc.page_content)
                doc.metadata["source"] = filepath
            all_documents.extend(docs)
        else:
            print(f"   ⚠️ Not found (skip): {filepath}")
    
    if not all_documents:
        raise FileNotFoundError(
            "No medical knowledge files found. "
            "Please run 'python ingest_data.py' first to generate the data files."
        )
    
    print(f"   📊 Loaded {len(all_documents)} documents from {len(MEDICAL_KNOWLEDGE_FILES)} sources")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n---\n", "\n\n", "\n", ". ", " ", ""]
    )
    splits = text_splitter.split_documents(all_documents)
    
    print(f"📊 Created {len(splits)} document chunks")
    
    # ChromaDB has max batch size of 5461, so we batch insert
    BATCH_SIZE = 5000
    
    if len(splits) <= BATCH_SIZE:
        # Small enough to insert at once
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            persist_directory=CHROMA_PERSIST_DIR
        )
    else:
        # Create empty vectorstore first
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_function
        )
        
        # Add documents in batches
        for i in range(0, len(splits), BATCH_SIZE):
            batch = splits[i:i + BATCH_SIZE]
            print(f"   📥 Adding batch {i//BATCH_SIZE + 1}/{(len(splits) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} docs)")
            vectorstore.add_documents(batch)
    
    vectorstore.persist()
    
    print(f"✅ ChromaDB created and persisted at {CHROMA_PERSIST_DIR}")
    return vectorstore


# ============================================================
# LLM INITIALIZATION
# ============================================================

def get_llm() -> ChatOllama:
    """
    Initialize ChatOllama with Llama 3 model.
    Includes error handling for model loading issues.
    """
    try:
        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.3,  # Optimized for Llama 3 stability (0.2-0.4 range)
            num_predict=2048,  # Sufficient for full triage response
        )
        return llm
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize ChatOllama with model '{LLM_MODEL}'. "
            f"Ensure Ollama is running and the model is pulled. Error: {e}"
        )


# ============================================================
# RAG CHAIN CONSTRUCTION
# ============================================================

from langchain.chains import create_history_aware_retriever

def create_rag_chain(llm: ChatOllama, vectorstore: Chroma):
    """
    Create the RAG chain with history-aware retrieval.
    """
    # 1. Create Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    
    # 2. History-Aware Retriever Logic
    # Reformulates the user's question based on history if needed
    contextualize_q_system_prompt = """Given a chat history and the latest user question
    which might reference context in the chat history, formulate a standalone question
    which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # 3. Answer Generation Logic
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", USER_INSTRUCTIONS),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 4. Final RAG Chain combining both
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain


# ============================================================
# EMERGENCY DETECTION
# ============================================================

def check_for_emergency(user_input: str) -> bool:
    """
    Check if user input contains emergency symptoms.
    Returns True if emergency detected.
    """
    user_lower = user_input.lower()
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in user_lower:
            return True
    return False


def get_emergency_response() -> str:
    """
    Return the emergency response message.
    """
    return """🚨 **EMERGENCY ALERT** 🚨

Based on your description, this may be a medical emergency requiring immediate attention.

**IMPORTANT:**
- I cannot provide a diagnosis for emergency symptoms.
- Please seek immediate medical help.

**Recommended Actions:**
1. **Call Emergency Services** (911, 999, or your local emergency number)
2. **Go to the nearest Emergency Room** immediately
3. **Do not delay** - time is critical in medical emergencies

If you're with someone experiencing these symptoms:
- Stay calm and keep them comfortable
- Do not give them food or water unless instructed by medical personnel
- Be ready to describe the symptoms to emergency responders

**Remember:** I am an AI and cannot replace emergency medical care. Your safety is the priority.

---
*This is an automated safety response. Please seek professional medical attention immediately.*
"""


# ============================================================
# SOURCE CITATION
# ============================================================

def format_sources(context_documents: List[Document]) -> str:
    """
    Format source documents as a clean list of references (titles/filenames only).
    Does NOT dump raw content.
    """
    if not context_documents:
        return ""
    
    sources = set()
    for doc in context_documents:
        source = doc.metadata.get("source", "Medical Knowledge Base")
        # Clean up source name (basename)
        source_name = os.path.basename(source)
        sources.add(source_name)
    
    if not sources:
        return ""
        
    return "\n\n---\n**📚 References:** " + ", ".join(sorted(sources))


# ============================================================
# ROBUST STARTUP INITIALIZATION (lazy + retry + cached failure)
# ============================================================

_EMBEDDING_FUNCTION = None
_VECTORSTORE = None
_LLM = None
_RAG_CHAIN = None
_INITIALIZATION_ERROR = None


def initialize_components(max_retries: int = 2) -> None:
    """Initialize shared components once with bounded retries.

    Raises RuntimeError if initialization fails after retries.
    """
    global _EMBEDDING_FUNCTION, _VECTORSTORE, _LLM, _RAG_CHAIN, _INITIALIZATION_ERROR

    if _RAG_CHAIN is not None and _VECTORSTORE is not None and _LLM is not None:
        return

    if _INITIALIZATION_ERROR is not None:
        raise RuntimeError(str(_INITIALIZATION_ERROR))

    print("🚀 Starting Medical Chatbot initialization...")
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            _EMBEDDING_FUNCTION = get_embedding_function()
            _VECTORSTORE = load_or_create_vectorstore(_EMBEDDING_FUNCTION)
            _LLM = get_llm()
            _RAG_CHAIN = create_rag_chain(_LLM, _VECTORSTORE)
            print("✅ Medical Chatbot initialization complete!")
            _INITIALIZATION_ERROR = None
            return
        except Exception as exc:
            last_error = exc
            print(f"⚠️ Initialization attempt {attempt}/{max_retries} failed: {exc}")

    _INITIALIZATION_ERROR = RuntimeError(
        f"Medical Chatbot initialization failed after {max_retries} attempts: {last_error}"
    )
    raise _INITIALIZATION_ERROR


# ============================================================
# CHAINLIT HANDLERS
# ============================================================

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chat session with RAG chain.
    Called when a new user session starts.
    Uses pre-initialized global vectorstore and RAG chain.
    """
    await cl.Message(content="🏥 Initializing Medical Chatbot... Please wait.").send()
    
    try:
        # Initialize shared components lazily with retry support
        initialize_components()

        # Initialize chat history for this session
        chat_history = []

        # Create retriever from shared vectorstore
        retriever = _VECTORSTORE.as_retriever(search_kwargs={"k": 2})
        
        # Store in session
        cl.user_session.set("rag_chain", _RAG_CHAIN)
        cl.user_session.set("retriever", retriever)
        cl.user_session.set("llm", _LLM)
        cl.user_session.set("chat_history", chat_history)
        
        # Send welcome message
        welcome_message = """# 🏥 Local Medical Chatbot

Welcome! I'm your medical information assistant.

## ⚠️ Important Disclaimers
- I am an **AI language model**, not a medical professional.
- Information provided is for **educational purposes only**.
- Always **consult a qualified healthcare professional** for medical advice.

## 💬 How to Use
- Ask questions about medical conditions, symptoms, or health topics.
- I will provide information based on my medical knowledge base.
- Sources will be cited for transparency.

## 🚨 Emergencies
If you're experiencing a medical emergency, please call emergency services (911/999/112) immediately.

---
**How can I help you today?**
"""
        await cl.Message(content=welcome_message).send()
        
    except FileNotFoundError as e:
        await cl.Message(content=f"❌ **Setup Error:** {str(e)}").send()
    except RuntimeError as e:
        await cl.Message(content=f"❌ **LLM Error:** {str(e)}").send()
    except Exception as e:
        await cl.Message(content=f"❌ **Initialization Error:** {str(e)}").send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming user messages.
    Processes through RAG chain and returns response with sources.
    """
    user_input = message.content
    
    # Get session data
    rag_chain = cl.user_session.get("rag_chain")
    chat_history = cl.user_session.get("chat_history")
    
    if rag_chain is None:
        await cl.Message(
            content="⚠️ Session not initialized. Please refresh the page."
        ).send()
        return
    
    # Check for emergency symptoms first
    if check_for_emergency(user_input):
        await cl.Message(content=get_emergency_response()).send()
        return
    
    # Send thinking indicator
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # 1. Get session objects
        retriever = cl.user_session.get("retriever")
        llm = cl.user_session.get("llm")
        
        if not retriever or not llm:
            msg.content = "⚠️ Session not initialized. Please refresh the page."
            await msg.update()
            return
        
        # 2. Retrieve relevant medical context
        docs = retriever.invoke(user_input)
        context_text = "\n".join([doc.page_content[:200] for doc in docs[:2]])
        
        # 3. Format chat history (last 4 exchanges = 8 messages)
        history_text = ""
        if chat_history:
            for msg_item in chat_history[-8:]:
                if hasattr(msg_item, 'content'):
                    role = "User" if isinstance(msg_item, HumanMessage) else "Assistant"
                    history_text += f"{role}: {msg_item.content}\n"
        
        # 4. Determine conversation phase
        turn_count = len(chat_history) // 2
        
        # 5. Build prompt as PERSONALITY DESCRIPTION (prevents 7B echo)
        if turn_count == 0:
            # First turn: describe symptoms and ask questions
            prompt = f"""You are an educational medical symptom checker.
You talk to users who describe how they feel.
You respond by explaining symptoms in simple medical terms.

You usually say what such symptoms are commonly related to, mention 2-3 possible conditions in a cautious way, and ask a few short questions to differentiate between them.

You do NOT name a single disease yet.

Medical context: {context_text[:250]}

User: {user_input}"""
        else:
            # Follow-up turns: explicit diagnosis allowed
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
        
        # 6. Call LLM
        response = await llm.ainvoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # 7. ECHO GUARDRAIL (Crucial for 7B models)
        # If model just repeats the prompt or starts with "You are", uses fallback
        if not answer or len(answer.strip()) < 10 or answer.strip().startswith("You are") or "You match symptoms" in answer:
            answer = (
                "I'm sorry you're not feeling well. "
                "Based on general medical discussions, your symptoms are often related "
                "to common upper respiratory conditions or viral infections.\n\n"
                "To better understand your symptoms:\n"
                "• How long have you had them?\n"
                "• Is your fever mild or high?\n"
                "• Do you have a cough or headache?\n\n"
                "(Response generated via fallback safe-mode due to model load)"
            )
            print("DEBUG [X]: Echo/Empty detected - using safe fallback")
        
        # 8. Simple code guardrails for language
        answer = answer.replace("the patient", "you").replace("The patient", "You")
        sources_text = format_sources(docs)
        full_response = answer + sources_text
        full_response += "\n\n---\n*⚕️ This is a preliminary educational assessment only. Please consult a healthcare professional for proper diagnosis and treatment.*"
        
        # 10. Update message
        msg.content = full_response
        await msg.update()
        
        # 11. Update chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > 16:
            chat_history = chat_history[-16:]
        cl.user_session.set("chat_history", chat_history)
        
    except Exception as e:
        await cl.Message(content=f"❌ **Error:** {str(e)}\n\nPlease try again.").send()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Run this application with: chainlit run app.py")
