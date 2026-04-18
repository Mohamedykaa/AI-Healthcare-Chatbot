import os
from langchain_core.messages import HumanMessage
from src.core.config import (
    RETRIEVER_K,
    RETRIEVER_SCORE_THRESHOLD,
    CONTEXT_CHAR_LIMIT_PER_DOC
)
from src.core.risk import assess_risk_level, get_emergency_response
from src.services.llm import get_llm
from src.services.vectorstore import (
    get_embedding_function,
    load_or_create_vectorstore
)

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

def format_sources(context_documents: list) -> str:
    if not context_documents: return ""
    sources = set()
    for doc in context_documents:
        source = doc.metadata.get("source", "Medical Knowledge Base")
        sources.add(os.path.basename(source))
    if not sources: return ""
    return "\n\n---\n**📚 References:** " + ", ".join(sorted(sources))

async def process_chat_message(user_input: str, chat_history: list):
    """
    Main logic to process a chat message.
    Returns: (response_text, risk_level, sources_text)
    """
    risk_level = assess_risk_level(user_input)
    if risk_level == "EMERGENCY":
        return get_emergency_response(), risk_level, ""

    if _LLM is None or _VECTORSTORE is None:
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

Medical context: {context_text[:800]}

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
