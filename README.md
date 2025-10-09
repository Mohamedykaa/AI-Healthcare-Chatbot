# 🩺 AI Healthcare Chatbot

An intelligent healthcare assistant built with **Streamlit** and **Machine Learning** that helps users describe their symptoms and get a preliminary medical analysis.  
The system predicts possible diseases, suggests tests, and provides general advice — all through a simple conversational interface.

---

## 🚀 Features

- 🤖 **Medical Chatbot Interface**: Interacts with users in natural language to collect symptoms.  
- 🧠 **AI-Powered Diagnosis**: Uses a TF-IDF + Random Forest model trained on symptom–disease data.  
- 📚 **Dynamic Knowledge Base**: Provides disease details, medical advice, and recommended tests.  
- 🧾 **PDF Report Generation**: Automatically creates downloadable medical reports for user queries.  
- 🌐 **Dual Operation Modes**:  
  - **Local Mode**: Uses the internal ML model.  
  - **API Mode**: Connects to a FastAPI backend for model inference.  
- 🧩 **Modular Architecture**: Cleanly separated modules for chatbot logic, model inference, and knowledge base management.

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend / UI** | Streamlit |
| **Backend / API** | FastAPI |
| **Machine Learning** | Scikit-learn (TF-IDF + Random Forest) |
| **Model Serialization** | Joblib |
| **Data Handling** | Pandas, NumPy |
| **Reporting** | ReportLab |
| **Other Tools** | Requests, QRCode |

---

## 📁 Project Structure

