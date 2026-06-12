# Minimal Local Version Patch (Pre-Defense)

This patch list contains **only** the zero-risk modifications that require no database migrations, no dependency upgrades, and do not increase GPU memory pressure for the local Ollama model.

---

## 1. `.chainlit/config.toml` (Localization Warning Fix)

### Original
```toml
# language = "en-US"
```

### Modified
```toml
language = "en-US"
```

---

## 2. `src/core/logic.py` (Safety & Injection Guard)

### Original
```python
def _evaluate_emergency_routing(user_text: str) -> str:
    # Existing basic emergency logic
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

# ... Inside process_chat_message (before LLM call) ...

    if detect_prompt_injection(user_input):
        return {
            "response": "I am a medical triage assistant. I cannot process this request or ignore my safety instructions. Please describe your medical symptoms.",
            "risk_level": "UNKNOWN",
            "sources": []
        }
```

---

## 3. `src/core/logic.py` (Medical Prompt Restructuring)

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
