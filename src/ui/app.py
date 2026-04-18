"""
Local Medical Chatbot - Chainlit Application
=============================================

A fully offline medical chatbot using:
- ChatOllama (llama3:8b model)
- HuggingFace embeddings (all-MiniLM-L6-v2)
- ChromaDB for persistent vector storage
- LangChain RAG architecture

Entry point: chainlit run src/ui/app.py
"""


from dotenv import load_dotenv
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage

# Import core logic
from src.core.logic import (
    initialize_components,
    process_chat_message
)


# Load environment variables
load_dotenv()

# ============================================================
# CHAINLIT HANDLERS
# ============================================================

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chat session with RAG chain.
    Called when a new user session starts.
    """
    await cl.Message(content="🏥 Initializing Medical Chatbot... Please wait.").send()
    
    try:
        # Initialize shared components lazily
        initialize_components()

        # Initialize chat history for this session
        chat_history = []
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
    chat_history = cl.user_session.get("chat_history")
    
    if chat_history is None:
        await cl.Message(
            content="⚠️ Session not initialized. Please refresh the page."
        ).send()
        return
    
    # Send thinking indicator
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Process message using core backend logic
        response_text, risk_level, sources_text = await process_chat_message(user_input, chat_history)
        
        # Append disclaimer
        full_response = response_text + sources_text
        if risk_level != "EMERGENCY":
            full_response += "\n\n---\n*⚕️ This is a preliminary educational assessment only. Please consult a healthcare professional for proper diagnosis and treatment.*"
        
        # Update message
        msg.content = full_response
        await msg.update()
        
        # Update chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response_text))
        if len(chat_history) > 16:
            chat_history = chat_history[-16:]
        cl.user_session.set("chat_history", chat_history)
        
    except Exception as e:
        await cl.Message(content=f"❌ **Error:** {str(e)}\n\nPlease try again.").send()


if __name__ == "__main__":
    print("Run this application with: chainlit run src/ui/app.py")
