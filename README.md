🧠 AI Healthcare Chatbot

An intelligent healthcare assistant powered by Natural Language Processing (NLP) and integrated geolocation services.
This project combines medical knowledge extraction, disease prediction, and real-time service recommendations into a single user-friendly system built with Streamlit.

🚀 Overview

The AI Healthcare Chatbot assists users in:

Analyzing symptoms to predict possible diseases

Suggesting medical precautions and follow-up advice

Locating nearby pharmacies and radiology centers using OpenStreetMap (OSM)

Maintaining chat history for improved diagnosis accuracy

Providing explainable and multilingual interactions (Arabic & English)

✨ Key Features
🩺 Medical Diagnosis System

Uses a trained NLP pipeline to predict diseases based on user-described symptoms.

Employs preprocessing via a TextCleaner class to unify training and inference.

Supports multi-symptom reasoning and confidence scoring.

🧭 Location-based Services

Pharmacy Locator (pharmacy_locator_osm.py): Fetches nearby pharmacies using Overpass API.

Radiology Locator (radiology_locator_osm.py): Finds clinics, labs, and hospitals providing diagnostic imaging.

Distance calculations handled via geopy.

💬 Chatbot Framework

Modular design inside src/chatbot_system/ with specialized agents:

diagnosis_agent.py: Handles disease prediction logic.

symptom_agent.py: Parses and validates symptoms.

followup_manager.py: Suggests next steps and monitors symptom evolution.

recommendation_agent.py: Provides personalized advice and health tips.

The main controller is implemented in chatbot_manager.py.

📊 Interactive Dashboard

app_streamlit.py and app_dashboard.py provide a clean Streamlit interface:

Real-time chat simulation.

History and settings panel.

Map integration for nearby healthcare services.

🧠 Machine Learning & NLP

Robust training pipeline in train_model.py:

Uses TF-IDF + RandomForestClassifier.

Prevents data leakage by splitting data before GridSearchCV.

Automatically saves model, encoder, and a JSON training report.

Supports --fast mode for quick retraining.
---
🗂️ Project Structure
AI-Healthcare-Chatbot/
│
├── data/                     # Medical datasets and merged training data
│   ├── merged_comprehensive_data.csv
│   ├── english_knowledge_base.json
│   └── symptom_precaution.csv
│
├── models/                   # Saved models and training reports
│   ├── optimized_nlp_pipeline.joblib
│   ├── nlp_label_encoder.joblib
│   ├── training_snapshot.csv
│   └── training_report.json
│
├── scripts/                  # Utility scripts for validation and automation
│   ├── check_data_preview.py
│   ├── check_project_integrity.py
│   └── run_all.bat
│
├── src/
│   ├── app_streamlit.py      # Streamlit main app
│   ├── app_dashboard.py      # Dashboard view
│   ├── main.py               # Entry point for CLI usage
│   │
│   ├── chatbot_system/       # Modular chatbot components
│   │   ├── chatbot_manager.py
│   │   ├── diagnosis_agent.py
│   │   ├── symptom_agent.py
│   │   ├── followup_manager.py
│   │   └── recommendation_agent.py
│   │
│   ├── utils/                # Support utilities
│   │   ├── text_cleaner.py
│   │   ├── data_merger.py
│   │   ├── report_generator.py
│   │   └── history_analyzer.py
│   │
│   ├── pharmacy_locator_osm.py
│   └── radiology_locator_osm.py
│
├── tests/                    # Unit & integration tests
│   ├── test_symptom_agent.py
│   ├── test_diagnosis_agent_unit.py
│   ├── test_followup_manager_unit.py
│   └── test_recommendation_agent.py
│
├── requirements.txt
├── README.md
└── pytest.ini
---
⚙️ Installation
1. Clone the repository
git clone https://github.com/Mohamedykaa/AI-Healthcare-Chatbot.git
cd AI-Healthcare-Chatbot

2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows

3. Install dependencies
pip install -r requirements.txt

🧩 Training the Model

To retrain the NLP model using your dataset:

Full GridSearch (recommended)
python src/train_model.py

Fast training (quick test mode)
python src/train_model.py --fast


Output files will be saved in the models/ directory:

optimized_nlp_pipeline.joblib

nlp_label_encoder.joblib

training_report.json

🖥️ Running the Streamlit App
streamlit run src/app_streamlit.py


Then open the local URL displayed in your terminal (default: http://localhost:8501).

🧪 Running Tests

The project uses pytest for automated validation.

pytest -v


This will run unit and integration tests for:

Symptom & diagnosis logic

Follow-up generation

Chatbot flow consistency

🧱 Design Principles

Separation of Concerns: Each module handles a distinct responsibility.

Data Integrity: Training, testing, and inference pipelines share identical preprocessing.

Scalability: Easily extendable to include more medical datasets or model types.

Explainability: Every response is traceable to known symptom-disease mappings.

🛠️ Technologies Used
Category	Technology
Language	Python 3.11
Web App	Streamlit
NLP & ML	Scikit-learn, TfidfVectorizer, RandomForest
Data Processing	Pandas, NumPy
Location Services	Overpass API, Geopy
Testing	Pytest
Storage	Joblib, JSON
🧭 Future Enhancements

🔹 Integrate a multilingual medical knowledge base

🔹 Add visual analytics for diagnosis confidence

🔹 Incorporate doctor/hospital booking APIs

🔹 Support voice-based interaction

👨‍💻 Author

Mohamed Yaser
AI Developer & Researcher
📍 Egypt
🔗 GitHub Profile

🩵 License

This project is licensed under the MIT License — feel free to use, modify, and share with proper attribution.

"Empowering healthcare with intelligent, accessible technology."
— Mohamed Yaser