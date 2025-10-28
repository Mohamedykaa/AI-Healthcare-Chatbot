 🧠 AI Healthcare Chatbot  

An intelligent **medical assistant** that analyzes symptoms, predicts possible diseases, suggests precautions, and locates nearby pharmacies or radiology centers using OpenStreetMap.  
This project integrates **Natural Language Processing (NLP)**, **Machine Learning**, and **interactive dialogue management** to deliver accurate, bilingual healthcare support.

---

## ⚙️ Key Features
- 🩺 **Symptom-based disease prediction** using optimized NLP pipeline (`RandomForest + TF-IDF`).
- 💬 **Dynamic conversation flow** — adaptive follow-up questions to refine diagnosis.
- 🌍 **Bilingual Input Support** (English + Arabic).
- 🏥 **Location Integration** — find nearby **pharmacies** and **radiology centers** via OSM.
- 📊 **Safe ML Training** — includes `TextCleaner`, prevents data leakage, and logs full training reports.
- 🧪 **Comprehensive Testing Suite** with `pytest` (unit + integration tests).
- 🧩 **Modular Architecture** — easy to retrain, extend, or integrate with APIs.
- 🖥️ **Streamlit Dashboard** for chatbot interaction and visualization.

---

## 🧩 Project Structure
AI-Healthcare-Chatbot/
│
├── data/ # Datasets and merged knowledge bases
│ ├── merged_comprehensive_data.csv
│ ├── english_knowledge_base.json
│ └── ...
│
├── models/ # Trained models and reports
│ ├── optimized_nlp_pipeline.joblib
│ ├── nlp_label_encoder.joblib
│ └── training_report.json
│
├── src/
│ ├── chatbot_system/ # Core logic (diagnosis, follow-up, recommendation)
│ ├── utils/ # Tools (data_merger, text_cleaner, report_generator)
│ ├── train_model.py # NLP model training script
│ ├── app_streamlit.py # Streamlit web interface
│ └── main.py # Backend entry point
│
├── scripts/ # Helper scripts (data check, automation)
├── tests/ # Unit and integration tests
├── requirements.txt # Dependencies
└── README.md # Documentation


---

## 🧠 Model Training
Train the disease-prediction NLP pipeline:

```bash
python src/train_model.py
Use fast mode (no GridSearchCV):

bash
نسخ الكود
python src/train_model.py --fast
Outputs:

models/optimized_nlp_pipeline.joblib

models/nlp_label_encoder.joblib

models/training_report.json

The training script automatically:

Splits data safely to prevent leakage

Integrates TextCleaner for consistent preprocessing

Logs model performance and parameters

🚀 Run the Chatbot App
Launch the Streamlit interface:

bash
نسخ الكود
streamlit run src/app_streamlit.py
Then open the app in your browser:
👉 http://localhost:8501

🧪 Running Tests
Execute all tests with verbose output:

bash
نسخ الكود
pytest -v
Tests include:

Unit tests for each agent

Integration flow for chatbot conversation

Follow-up and recommendation logic validation

🧰 Requirements
Install dependencies before running:

bash
نسخ الكود
pip install -r requirements.txt
💡 Example Workflow
User enters symptoms (e.g., “I have fever and cough”).

Chatbot predicts possible diseases using trained NLP model.

System asks follow-up questions to refine diagnosis.

Displays disease description, precautions, and nearest pharmacies/radiology centers.

📦 Future Enhancements
🔗 API integration with verified medical sources.

🧠 Expand multilingual support.

💬 Voice-based chatbot interaction.

☁️ Cloud deployment on Render or Hugging Face Spaces.

👨‍💻 Author
Mohamed Yaser
AI Engineer & ML Developer
📧 [Your Email or GitHub Profile]

⭐ If you find this project useful, consider giving it a star on GitHub! ⭐