# Defense Summary: AI Healthcare Chatbot

## 1. Project Goal
To develop a privacy-first, educational AI medical assistant that helps users understand their symptoms and provides safe triage guidance on when to seek professional medical care.

## 2. Problem Statement
Patients often turn to the internet for symptom checking, which frequently leads to anxiety (cyberchondria) due to unregulated, non-contextual search results. Conversely, sending all medical queries to cloud-based LLMs poses severe patient data privacy risks. There is a need for a localized, highly safeguarded system that provides grounded medical information without retaining personal health data in the cloud.

## 3. Solution
A locally-hosted Retrieval-Augmented Generation (RAG) chatbot utilizing LLaMA 3. The system operates entirely offline on the host machine, ensuring absolute privacy. It employs deterministic safety wrappers to prevent medical hallucinations and unauthorized diagnostic behavior.

## 4. Architecture
The architecture is streamlined for stability and safety:
- **Backend:** FastAPI for headless REST integration and Chainlit for web UI.
- **AI Core:** LangChain orchestration with local Ollama running LLaMA 3 8B.
- **Knowledge Base:** ChromaDB vector store leveraging HuggingFace embeddings (`all-MiniLM-L6-v2`).
- **Safety Layer:** Multi-tiered deterministic intercepts evaluating risk and prompt integrity before LLM inference.

## 5. Key Features
- **Retrieval-Augmented Generation:** Grounds all responses in verified medical datasets.
- **Sufficiency-Based Triage:** The logic engine demands specific information (onset, severity, red flags) before allowing the LLM to output a differential assessment.
- **Bilingual Support:** Automatically detects and replies in English or Arabic.
- **Stateless API:** Designed to easily integrate with mobile frontends (e.g., Flutter).

## 6. Safety Mechanisms
- **Prompt Injection Guard:** A regex-based layer that instantly blocks malicious system bypass attempts and drug prescription requests.
- **Over-reassurance Guard:** Post-processing regex that replaces dangerous LLM hallucinations (e.g., "nothing to worry about") with safe, medical caveats.
- **Red Flag Enforcement:** Appends a hardcoded warning if the system presents an assessment without the user explicitly confirming or denying critical emergency markers.

## 7. Why RAG was used
Standard LLMs hallucinate medical facts and provide generalized answers. By utilizing RAG, we restrict the LLM's knowledge generation to a specific, curated set of medical documents (MedQuad, MedMCQA). If the answer isn't in the database, the system is instructed to safely admit lack of knowledge rather than guess.

## 8. Why the Risk Engine exists
LLMs are probabilistic and cannot be fully trusted with life-or-death emergency detection. The Risk Engine is a *deterministic* Python script that uses keyword scoring and negation-awareness to intercept inputs like "heart attack" or "suicide" and instantly return a hardcoded emergency protocol, completely bypassing the generative AI.

## 9. Project Limitations
- **No Diagnostics:** The chatbot performs triage education, not definitive diagnosis.
- **Corpus Bound:** It cannot answer medical questions outside the scope of its ingested vector database.
- **Local Hardware Limits:** Inference latency is tied directly to the host machine's hardware capabilities, as the 8B model runs locally.

## 10. Future Work
- Expanding the localized vector database to encompass a wider range of rare diseases.
- Optimizing chunking strategies to improve retrieval density and reduce memory overhead during context assembly.
- Enhancing the deterministic negation logic within the risk engine for more complex sentence structures.
