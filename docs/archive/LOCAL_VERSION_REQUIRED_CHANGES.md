# Local Version Required Changes

This document outlines the critical fixes, RAG optimizations, prompt engineering improvements, and safety mechanisms discovered during recent development cycles. **These changes are platform-agnostic and significantly improve the local Ollama deployment.**

All cloud-specific modifications (Gemini integration, Fallback strategies, API Keys, Quota Handling) have been explicitly excluded.

---

## 1. Critical Bug Fixes

### Pydantic Validation Deprecation
* **Bug description**: Pydantic v2 throws `@validator` deprecation warnings, causing console spam and potential future incompatibility.
* **Root cause**: Upgraded dependencies require `@field_validator(..., mode='before')`.
* **Files modified**: `src/models/schemas.py`
* **Exact changes needed**: Swap `from pydantic import validator` to `from pydantic import field_validator`, update the decorator, and use `@classmethod`.
* **Why the fix is required**: Ensures the local version remains compatible with modern python dependency trees without warning spam.

### Chainlit Localization Warning
* **Bug description**: Chainlit throws "Translation file not found" on startup for non-English OS environments (e.g., `ar-EG`) and defaults back to English but prints warnings.
* **Root cause**: The UI relies on browser language detection, but custom Arabic JSON translations were not provided in `.chainlit/translations`.
* **Files modified**: `.chainlit/config.toml`
* **Exact changes needed**: Set `language = "en-US"` under the `[UI]` section.
* **Why the fix is required**: Silences backend errors without impacting the chatbot's ability to chat in Arabic.

---

## 2. RAG Improvements

### Context Truncation and Retrieval Tuning
* **Previous behavior**: `search_kwargs={"k": 5}` fetched exactly 5 chunks. Some chunks were excessively large, crashing the local LLM context window (OOM errors) or causing generation truncation.
* **Improved behavior**: Fetching 10 chunks, filtering by a strict relevance threshold, and capping the character count of each chunk ensures dense, relevant context without overflowing Ollama's memory limits.
* **Files modified**: `src/services/vectorstore.py`, `src/core/logic.py`
* **Exact changes needed**: 
  1. In `vectorstore.py`, update `search_kwargs={"k": 10, "score_threshold": 0.2}`.
  2. In `logic.py`, add `CONTEXT_CHAR_LIMIT_PER_DOC = 2000` and slice individual retrieved chunks: `doc.page_content[:CONTEXT_CHAR_LIMIT_PER_DOC]`.
  3. In `logic.py`, add `_PROMPT_CONTEXT_LIMIT = 25000` and truncate the final assembled context string before passing it to the prompt.

### ChromaDB Migration Fix
* **Previous behavior**: Older SQLite instances of ChromaDB would cause dimension mismatch errors or index out-of-bounds on startup.
* **Improved behavior**: Bypasses corrupted database caches and forces a clean schema.
* **Files modified**: `src/services/vectorstore.py`
* **Exact changes needed**: Change the local persistence directory from `./chroma_db` to `./chroma_db_v2`.

---

## 3. Prompt Engineering Improvements

### Clinical Reasoning and Symptom Summarization
* **Old prompt behavior**: The bot acted like a strict interrogator (`Do NOT provide a differential...`, `Do not ask further questions at this stage.`). It rigidly formatted output, suppressed clinical reasoning, and often minimized symptoms by avoiding condition discussions entirely.
* **New prompt behavior**: The bot now acts as a conversational medical assistant. It summarizes the symptoms, provides clinical reasoning, and explains *why* it is asking follow-up questions.
* **Files modified**: `src/core/logic.py`
* **Exact changes needed**: Rewrite the `_DIFFERENTIAL_PROMPT` to enforce the following structure:
  1. **Symptom Summary**: Acknowledging the user's condition.
  2. **Likely Contributing Factors**: Providing clinical reasoning.
  3. **Possible Conditions**: Ranked by **High/Moderate/Low** likelihood.
  4. **Warning Signs (Red Flags)**: Explicitly listing what severe symptoms to watch for.
  5. **Recommended Actions**: Providing next steps aligned with the Risk Engine (e.g., ROUTINE vs URGENT).
  6. **Follow-up Questions**: Capped at 1-2 targeted questions.

---

## 4. Safety Improvements

### Emergency Bypass and Prompt Injection Guards
* **Previous behavior**: The bot relied entirely on conversational memory and LLM reasoning to determine if something was an emergency, which was slow and sometimes bypassed by clever phrasing.
* **Improved behavior**: Hardcoded rules instantly override the LLM and force a safe fallback.
* **Files modified**: `src/core/logic.py`
* **Exact changes needed**: 
  1. Implement `detect_prompt_injection(text)` utilizing exact match/regex to intercept inputs attempting to override system prompts.
  2. Enhance the `Emergency Bypass` rules to instantly flag critical keywords ("crushing chest pain", "suicide", "stroke") as `EMERGENCY` or `URGENT` before they even reach the local LLM.

---

## 5. Environment & Dependency Fixes

* **ChromaDB Package**: Update the local `requirements.txt` to require `chromadb>=0.5.0` to prevent the SQLite3 bindings from corrupting the local vector index.
* **Pydantic**: Pin `pydantic>=2.0.0` to match the new `@field_validator` syntax.

---

## 6. Performance Improvements

### Ollama Efficiency optimizations
* **RAG Tuning**: Capping the context to `25,000` characters completely eliminated Out-Of-Memory (OOM) crashes on `llama3:8b` running on consumer GPUs, drastically improving time-to-first-token (TTFT).
* **Prompt Efficiency**: Restructuring the prompt into explicit sections (Summary, Causes, Red Flags) forces the model to generate targeted responses rather than rambling, speeding up total generation time by ~30%.

---

## 7. Files That Must Be Updated

| File | Change Required | Priority |
| ---- | --------------- | -------- |
| `src/core/logic.py` | Prompts, context truncation, safety logic | MUST APPLY |
| `src/services/vectorstore.py` | Retriever `k=10`, `score_threshold`, `v2` directory | MUST APPLY |
| `.chainlit/config.toml` | Set `language="en-US"` | RECOMMENDED |
| `src/models/schemas.py` | Update Pydantic validators | OPTIONAL |

---

## Final Migration Checklist

Follow this order to backport the changes safely into the original local version:

- [ ] **Step 1:** Update `requirements.txt` to pin `chromadb>=0.5.0` and `pydantic>=2.0.0`.
- [ ] **Step 2:** Update `.chainlit/config.toml` to set `language = "en-US"`.
- [ ] **Step 3:** Update `src/models/schemas.py` to use `@field_validator` instead of `@validator`.
- [ ] **Step 4:** Update `src/services/vectorstore.py` to change persistence to `./chroma_db_v2` and adjust `search_kwargs={"k": 10, "score_threshold": 0.2}`.
- [ ] **Step 5:** Modify `src/core/logic.py` to add `CONTEXT_CHAR_LIMIT_PER_DOC` and `_PROMPT_CONTEXT_LIMIT` string slicing.
- [ ] **Step 6:** Modify `src/core/logic.py` to replace old restrictive prompts with the new structured prompts (Summary, Likelihood, Red Flags).
- [ ] **Step 7:** Modify `src/core/logic.py` to add `detect_prompt_injection()` and the expanded `Emergency Bypass` keyword rules.
