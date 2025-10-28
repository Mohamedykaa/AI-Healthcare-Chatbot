# 🧠 AI Healthcare Chatbot

An intelligent healthcare assistant that provides preliminary symptom analysis, condition suggestions, and general medical guidance using Natural Language Processing (NLP) and Machine Learning.  
The chatbot integrates a user-friendly **Streamlit interface** for smooth interaction and features a modular knowledge base for scalability and continuous improvement.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Testing](#testing)
- [Future Improvements](#future-improvements)
- [Credits](#credits)

---

## 🩺 Overview

This project aims to assist users in identifying possible medical conditions based on described symptoms.  
It combines **NLP preprocessing**, a **symptom-disease matching system**, and a **Streamlit-based interface** for real-time diagnosis support.  
The goal is not to replace doctors, but to help users gain quick insights before professional consultation.

---

## ⚙️ Features

- 🔍 **Symptom-Based Diagnosis:** Suggests possible conditions based on user inputs.  
- 💬 **Interactive Chatbot:** Understands natural language queries in both English and Arabic.  
- 🧠 **Knowledge Base Integration:** Medical data is stored in a JSON structure for easy updating.  
- 🧾 **Multi-Symptom Logic:** Handles multiple concurrent symptoms for accurate analysis.  
- 🌐 **Streamlit UI:** Simple, responsive, and clean interface for end-users.  
- 🧰 **Modular Design:** Easily expandable and maintainable project structure.

---

## 🗂️ Project Structure

| Folder / File | Description |
|----------------|-------------|
| `app_streamlit.py` | Streamlit application that runs the chatbot UI. |
| `src/` | Contains core logic for symptom analysis and chatbot functions. |
| `models/` | Machine learning models and training artifacts. |
| `data/` | Medical datasets and JSON-based knowledge base. |
| `scripts/` | Helper scripts for data cleaning, preprocessing, and setup. |
| `tests/` | Unit tests and validation scripts. |
| `requirements.txt` | Dependencies required for running the project. |
| `README.md` | Project documentation (this file). |

---

## 💻 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mohamedykaa/AI-Healthcare-Chatbot.git
   cd AI-Healthcare-Chatbot
Create a virtual environment (recommended):


python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
Install dependencies:


pip install -r requirements.txt
🚀 Usage
Run the chatbot interface:


streamlit run app_streamlit.py
Interact with the bot:

Type your symptoms (e.g., "I have a headache and sore throat").

The system will analyze and provide possible conditions.

You can also ask general health-related questions.

🧠 Model Training
The machine learning component is responsible for predicting conditions based on symptom combinations.
To retrain the model:


python src/train_model.py
Make sure the dataset files are properly located inside the data/ folder.

🧪 Testing
To run all available tests:


pytest tests/
Tests cover:

Knowledge base integrity

Model prediction accuracy

Streamlit app behavior (basic functional checks)

🚧 Future Improvements
🧬 Integration with live medical APIs for real-time data.

🗣️ Advanced NLP model (e.g., fine-tuned transformer) for deeper conversation understanding.

💾 User history tracking and analytics dashboard.

🌍 Multilingual support expansion.

🤖 Integration with voice input/output.

👨‍💻 Credits
Developed by Mohamed Yaser
AI & Machine Learning Enthusiast | 2025

⭐ If you found this project useful, consider giving it a star on GitHub!


