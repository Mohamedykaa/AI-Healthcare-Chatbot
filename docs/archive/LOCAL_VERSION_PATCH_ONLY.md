# Local Version Patch Modifications

This document contains exact copy-paste code snippets for migrating the core RAG, Safety, and Prompt Architecture improvements to the original local Ollama codebase. 

All cloud, API, Gemini, and Fallback wrapper configurations have been stripped out.

---

## 1. `.chainlit/config.toml`

**Reasoning:** Silences the "Translation file not found" warning spam when processing Arabic inputs.
**Status:** RECOMMENDED

### Original
```toml
# Force a specific language for all users (e.g., "en-US", "he-IL", "fr-FR")
# If not set, the browser's language will be used
# language = "en-US"
```

### Modified
```toml
# Force a specific language for all users (e.g., "en-US", "he-IL", "fr-FR")
# If not set, the browser's language will be used
language = "en-US"
```

---

## 2. `src/core/config.py`

**Reasoning:** Updates the RAG architecture to pull more context but strict-filter it, applies hard context character limits to prevent Ollama Out-Of-Memory (OOM) errors, and upgrades the ChromaDB directory to bypass corrupted SQLite caches.
**Status:** REQUIRED

### Original
```python
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
# ...
RETRIEVER_K = int(os.environ.get("RETRIEVER_K", "5"))
```

### Modified
```python
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db_v2")
# ...
RETRIEVER_K = int(os.environ.get("RETRIEVER_K", "10"))
RETRIEVER_SCORE_THRESHOLD = float(os.environ.get("RETRIEVER_SCORE_THRESHOLD", "0.2"))
CONTEXT_CHAR_LIMIT_PER_DOC = int(os.environ.get("CONTEXT_CHAR_LIMIT_PER_DOC", "2000"))
```

---

## 3. `src/core/logic.py` (RAG Context Truncation)

**Reasoning:** Protects the local `llama3:8b` context window from overflow by enforcing character caps on both the entire prompt and individual chunks.
**Status:** REQUIRED

### Original
```python
_PROMPT_CONTEXT_LIMIT = 15000

# Inside RAG Logic:
      retriever = _VECTORSTORE.as_retriever(search_kwargs={"k": RETRIEVER_K})
      
      # ...
      
      context_docs = [doc.page_content for doc in docs]
```

### Modified
```python
_PROMPT_CONTEXT_LIMIT = 25000

# Inside RAG Logic:
      retriever = _VECTORSTORE.as_retriever(
          search_type="similarity_score_threshold",
          search_kwargs={"k": RETRIEVER_K, "score_threshold": RETRIEVER_SCORE_THRESHOLD},
      )
      
      # ...
      
      context_docs = [
          doc.page_content[:CONTEXT_CHAR_LIMIT_PER_DOC].strip()
          for doc in docs[:RETRIEVER_K]
          if getattr(doc, "page_content", "").strip()
      ]
```

---

## 4. `src/core/logic.py` (Safety & Injection Guard)

**Reasoning:** Adds prompt injection protection and emergency bypass rules to catch critical inputs before processing them through the slow local LLM.
**Status:** REQUIRED

### Original
```python
# Missing prompt injection logic

def _evaluate_emergency_routing(user_text: str) -> str:
    # Basic logic
```

### Modified
```python
def detect_prompt_injection(text: str) -> bool:
    """Detect common prompt injection patterns to protect system instructions."""
    injection_patterns = [
        r"(?i)\bignore\s+(all\s+)?(previous\s+)?(instructions|directions)\b",
        r"(?i)\byou\s+are\s+(now\s+)?(a\s+)?(developer|admin|root|jailbreak)\b",
        r"(?i)\bforget\s+(what\s+you\s+were\s+told|your\s+prompt)\b",
        r"(?i)\bbypass\s+(security|filters|safeguards)\b",
        r"(?i)\bsimulate\s+(a\s+scenario|developer\s+mode)\b",
        r"(?i)\bdisregard\s+previous\b",
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, text):
            return True
            
    # Exact phrase checks
    exact_phrases = [
        "ignore all previous instructions",
        "system prompt",
        "what are your instructions",
    ]
    lower_text = text.lower()
    for phrase in exact_phrases:
        if phrase in lower_text:
            return True
            
    return False

# ... Inside process_chat_message ...

    if detect_prompt_injection(user_input):
        return {
            "response": "I am a medical triage assistant. I cannot process this request or ignore my safety instructions. Please describe your medical symptoms.",
            "risk_level": "UNKNOWN",
            "sources": []
        }
```

---

## 5. `src/core/logic.py` (Prompt Restructuring)

**Reasoning:** Fixes the chatbot's interrogative behavior, adds ranked clinical reasoning (High/Moderate/Low), and provides Red Flags.
**Status:** REQUIRED

### Original
```python
_DIFFERENTIAL_PROMPT = """You are a medical symptom checker.
Do NOT provide a differential or suggest conditions yet.
Ask follow up questions.
Do not ask further questions at this stage."""
```

### Modified
```python
_DIFFERENTIAL_PROMPT = """Response strategy ?? COMPREHENSIVE ASSESSMENT:
Sufficient information has been gathered. Structure your response EXACTLY as follows:

1. **Symptom Summary:** Briefly summarize the user's key symptoms and timeline.
2. **Likely Contributing Factors:** Explain how their lifestyle, context, or triggers (e.g. stress, sleep) might be playing a role.
3. **Possible Conditions:** Provide a cautious, ranked differential:
   - [Condition Name] (High Likelihood): [Reasoning]
   - [Condition Name] (Moderate Likelihood): [Reasoning]
   - [Condition Name] (Low Likelihood): [Reasoning]
4. **Warning Signs (Red Flags):** List specific severe symptoms the user should watch out for that would require immediate emergency care.
5. **Recommended Actions:** Provide clear, practical next steps and guidance aligned with the Risk Engine (e.g., ROUTINE follow-up, or URGENT same-day evaluation). Do NOT minimize symptoms.
6. **Follow-up Questions:** Ask 1-2 targeted questions to help narrow down the condition further or monitor progression.

Use medically cautious language ?? never present a diagnosis as definitive. Do not behave like an interrogator."""
```
