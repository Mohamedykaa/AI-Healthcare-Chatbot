🩺 AI Healthcare Chatbot
An intelligent healthcare assistant built with Streamlit and Machine Learning that helps users describe their symptoms and get a preliminary medical analysis.
The system predicts possible diseases, suggests tests, and provides general advice — all through a simple conversational interface.
🚀 Features
🤖 Medical Chatbot Interface: Interacts with users in natural language to collect symptoms.
🧠 AI-Powered Diagnosis: Uses a TF-IDF + Random Forest model trained on symptom–disease data.
📚 Dynamic Knowledge Base: Provides disease details, medical advice, and recommended tests.
🧾 PDF Report Generation: Automatically creates downloadable medical reports for user queries.
🌐 Dual Operation Modes:
Local Mode: Uses the internal ML model.
API Mode: Connects to a FastAPI backend for model inference.
🧩 Modular Architecture: Cleanly separated modules for chatbot logic, model inference, and knowledge base management.
🧰 Tech Stack
Component
Technology
Frontend / UI
Streamlit
Backend / API
FastAPI
Machine Learning
Scikit-learn (TF-IDF + Random Forest)
Model Serialization
Joblib
Data Handling
Pandas, NumPy
Reporting
ReportLab
Other Tools
Requests, QRCode

📁 Project Structure
AI-Healthcare-Chatbot/
│
├── chatbot_system/               # Core chatbot agents and knowledge base
│   ├── diagnosis_agent.py
│   ├── symptom_agent.py
│   ├── recommendation_agent.py
│   └── knowledge_base.json
│
├── data/                         # Dataset for training
│   └── DiseaseSymptomDescription.csv
│
├── models/                       # Trained ML pipeline and label encoder
│   ├── optimized_nlp_pipeline.joblib
│   └── nlp_label_encoder.joblib
│
├── src/                          # Main Streamlit and API application
│   ├── app.py                    # Streamlit UI
│   ├── main.py                   # FastAPI backend
│   ├── train_model.py            # Model training script
│   └── utils/                    # Helper modules
│
├── requirements.txt              # Python dependencies
├── run_all.bat                   # Windows batch file to run app + API
├── english_knowledge_base.json   # English version of the knowledge base
└── README.md


⚙️ Installation & Setup
Clone the repository:
git clone https://github.com/Mohamedykaa/AI-Healthcare-Chatbot.git
cd AI-Healthcare-Chatbot


Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate   # On Windows


Install all dependencies:
pip install -r requirements.txt


Run the application:
To run only the Streamlit UI in Local Mode:
streamlit run src/app.py

To run both the Streamlit UI and the FastAPI backend for API Mode:
run_all.bat


