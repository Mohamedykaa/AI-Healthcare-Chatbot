

### Run the Chatbot
```bash
python run_app.py
```

The chatbot will open automatically at **http://localhost:8000

> ⏱️ First launch after a fresh install may take 1–2 minutes to build the vector database.
> Subsequent launches are instant.

---

## 🩺 Step 2: Demo Scenario — Routine Symptom Consultation

This scenario demonstrates **RAG retrieval**, **multi-turn conversation**, and **source citation**.

### Turn 1 — Initial Symptom Input

**You type:**
```
I have been feeling very tired lately, with frequent headaches and dizziness
```

**What to expect:**
- The bot will explain these symptoms in simple medical terms
- It will mention 2–3 possible conditions (e.g., anemia, dehydration, blood pressure issues)
- It will ask 1–2 follow-up questions to narrow down
- Sources will be cited at the bottom (📚 References)

**What to point out to your professor:**
> "The system retrieved relevant medical documents from ChromaDB using semantic similarity, then used LLaMA 3 to generate a grounded response. Notice the source citations at the bottom — the bot never hallucinates, it always references its knowledge base."

---

### Turn 2 — Answering Follow-Up Questions

**You type:**
```
Yes I have been eating less than usual and I feel cold all the time
```

**What to expect:**
- The bot uses conversation history to narrow down
- It becomes more specific, possibly pointing toward iron-deficiency anemia
- It may ask about additional symptoms (pale skin, shortness of breath)

**What to point out:**
> "The system maintains multi-turn context. It remembers the previous symptoms and uses the new information to refine its analysis — similar to how a doctor conducts a differential diagnosis."

---

### Turn 3 — Getting the Assessment

**You type:**
```
Yes my skin has been pale and I get short of breath when climbing stairs
```

**What to expect:**
- The bot will now provide a more confident assessment
- It will explain *why* this pattern is consistent with a specific condition
- It will recommend seeing a doctor

**What to point out:**
> "After gathering enough information through follow-up questions, the system provides its assessment with medically cautious language. It always recommends professional consultation — it never claims to diagnose."

---

## 🚨 Step 3: Demo Scenario — Emergency Detection

This scenario demonstrates the **deterministic safety layer** that bypasses the LLM entirely.

### Emergency Test

**You type:**
```
I am having a heart attack right now and I can't breathe
```

**What to expect:**
- 🚨 **EMERGENCY ALERT** appears immediately
- The bot does NOT attempt to diagnose
- It provides emergency contacts and action steps
- Response is instant (no LLM delay)

**What to point out:**
> "This is our safety-first architecture. The keyword 'heart attack' triggers a hard-stop in the deterministic risk engine (`risk.py`). This layer runs BEFORE the LLM — it uses pure regex matching with zero dependencies. The response is instant because it bypasses the AI model entirely. This ensures we never give medical advice in life-threatening situations."

---

## ⚠️ Step 4: Demo Scenario — Urgent Detection

This shows the middle tier: not an emergency, but flagged as urgent.

**You type:**
```
I have severe chest pain and cold sweating
```

**What to expect:**
- ⚠️ **URGENT ADVICE REQUIRED** banner appears at the top
- The bot still provides information, but with an urgent warning
- Recommends same-day medical evaluation

**What to point out:**
> "The scoring engine calculates a risk score: 'chest pain' = 3 points, 'cold sweating' = 3 points, 'severe' = 2 points = total 8 points, which exceeds the emergency threshold of 6. This is a calibrated system — a single symptom like mild chest pain won't trigger emergency, but the combination of red flags does."

---

## 🔍 Step 5: Demo Scenario — General Medical Knowledge (RAG)

This shows that the bot can answer general medical questions, not just symptom-checking.

**You type:**
```
What is diabetes and how does it affect the body?
```

**What to expect:**
- A well-structured educational explanation
- Information retrieved from the medical knowledge base (MedQuad/MedMCQA)
- Source citations

**You type next:**
```
What are the early warning signs of kidney disease?
```

**What to expect:**
- Another grounded, cited response
- Shows breadth of the knowledge base

---

## 🧪 Step 6 (Optional): Show the Tests

If the professor asks about testing:

```bash
pytest tests/
```

This runs unit tests covering:
- Emergency keyword detection
- Text normalization
- Source citation formatting
- Safety filters

---

## 📊 Step 7 (Optional): Show Evaluation Metrics

```bash
python scripts/evaluate_rag.py --retrieval
```

This shows quantitative metrics:
- **Triage Accuracy** — deterministic emergency classification
- **Retrieval Hit Rate** — does ChromaDB return relevant documents?

---

## 💡 Key Talking Points During the Demo

| Feature | What to Say |
|---------|-------------|
| **RAG** | "We use Retrieval-Augmented Generation — the LLM doesn't make up answers, it references a curated medical knowledge base" |
| **Safety** | "Emergency detection is deterministic, not AI-based — it can never fail due to hallucination" |
| **Privacy** | "Everything runs locally — no patient data leaves the machine. Ollama runs the LLM on localhost" |
| **Multi-turn** | "The system remembers context across turns, mimicking a doctor's interview process" |
| **Sources** | "Every response includes citations from the knowledge base for traceability" |
| **Evaluation** | "We have a quantitative evaluation pipeline that measures retrieval quality and triage accuracy" |

---

## 🛑 How to Stop the Application

Press `Ctrl+C` in the terminal where the app is running.
