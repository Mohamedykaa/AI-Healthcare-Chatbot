# System Architecture: Healthcare AI Chatbot

## System Overview
The Healthcare AI Chatbot is a localized, privacy-first, educational symptom checker powered by Retrieval-Augmented Generation (RAG). It provides users with medical triage guidance and symptom characterization without performing definitive diagnoses. 

The system leverages a local Large Language Model (LLaMA 3 via Ollama) augmented with a curated vector database of medical knowledge, heavily safeguarded by deterministic prompt injection and risk assessment layers.

## Architecture Flow

```
User
  ↓
Flutter App (Client UI) / Chainlit (Web UI)
  ↓
FastAPI Backend (REST API)
  ↓
Prompt Injection Guard (Deterministic security)
  ↓
Risk Engine (Deterministic triage classifier)
  ↓
RAG Retriever (LangChain + ChromaDB)
  ↓
Vector Store (ChromaDB - all-MiniLM-L6-v2 Embeddings)
  ↓
LLM Service (LLaMA 3 8B via local Ollama)
  ↓
Response Formatting Layer (Sufficiency Triage Logic)
  ↓
Flutter App / Chainlit UI
```

## Core Components

### 1. Frontend Interfaces (Flutter / Chainlit)
The project supports two primary interfaces:
- **Chainlit Web UI:** Used for direct browser-based interaction, local testing, and debugging.
- **Flutter App (via FastAPI):** The primary mobile interface that communicates with the `src/api/main.py` REST API.

### 2. FastAPI Backend
A lightweight API layer (`src/api/main.py`) exposing a stateless `/chat` endpoint. It receives the user message and conversation history, processes them through the AI pipeline, and returns the LLM response, risk level, and source citations.

### 3. Prompt Injection Protection
The first layer of defense. A deterministic regex-based guard (`is_prompt_injection` in `src/core/logic.py`) that intercepts phrases attempting to bypass system instructions or request drug prescriptions (e.g., "ignore previous instructions", "prescribe morphine"). It instantly returns a secure refusal, bypassing the LLM entirely.

### 4. Risk Assessment Engine
A deterministic keyword and scoring engine (`src/core/risk.py`). It analyzes the user's input (and conversation history) for critical emergency markers (e.g., "heart attack", "loss of consciousness").
- **EMERGENCY (Score >= 6):** Bypasses LLM, immediately returns hardcoded emergency escalation.
- **URGENT (Score >= 3):** Flags the response for a same-day medical evaluation warning.
- **ROUTINE:** Standard processing.

### 5. Retrieval-Augmented Generation (RAG)
The system uses LangChain to construct a context-aware prompt. The retriever combines the first user message (the anchor symptom) with recent messages to formulate a search query.

### 6. ChromaDB Vector Store
A local instance of ChromaDB stores chunks of medical datasets (MedQuad, MedMCQA). Queries are embedded using `all-MiniLM-L6-v2` via HuggingFace (CPU) and retrieved using a similarity score threshold.

### 7. LLM Service
The generative core is powered by LLaMA 3 (8B parameters) running locally via Ollama. It synthesizes the retrieved context and the user's symptoms into a structured, educational response in the user's native language (English or Arabic).

### 8. Response Formatting Layer (Sufficiency Triage)
Instead of forcing immediate diagnoses, the logic layer (`src/core/logic.py`) determines if enough information (onset, severity, red flags, context) has been gathered. It guides the LLM to either ask clarifying questions (`CHARACTERIZATION`) or provide a structured, cautious list of possibilities (`DIFFERENTIAL`).

## Data Flow (Request Lifecycle)
1. **Receive:** User sends a message via FastAPI or Chainlit.
2. **Security Check:** System checks for prompt injections. If detected, return Security Alert.
3. **Risk Check:** System calculates the triage score. If EMERGENCY, return hardcoded alert.
4. **Context Retrieval:** System embeds the query, queries ChromaDB, and retrieves top 10 relevant document chunks.
5. **Prompt Assembly:** System truncates context to fit memory limits, prepends strict Arabic/English system instructions, appends triage strategy based on missing information, and loads history.
6. **Inference:** LLaMA 3 generates the response.
7. **Post-Processing:** System applies deterministic safety fallbacks (e.g., replacing dismissive phrases with cautious advice) and appends source citations.
8. **Return:** Final payload delivered to the UI.

## Safety Layers
- **Emergency Bypass:** Hardcoded overriding for critical keywords.
- **Prompt Injection Detection:** Pre-LLM interception of jailbreak attempts.
- **Red Flag Notices:** Deterministic warnings appended if a user receives a differential without addressing severe warning signs.
- **Over-reassurance Guard:** Post-LLM regex replacement of phrases like "nothing to worry about" with safe medical caveats.

## Project Limitations
- **No Diagnostic Authority:** The system is strictly educational and does not provide definitive clinical diagnoses.
- **Dependency on Corpus:** The RAG retriever can only surface information present in the local ChromaDB index; unknown diseases trigger a generic safe response.
- **Hardware Bound:** Inference speed and context window limits are constrained by the local CPU/GPU hardware running Ollama and LLaMA 3 8B.
- **No Treatment Guidance:** The system explicitly refuses to provide medication names, dosages, or therapeutic plans.
