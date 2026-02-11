# API Documentation

## Available FastAPI Endpoints

This project contains **TWO separate FastAPI applications**:

### 1. Primary API: `src/api/main.py`
**Purpose**: Modern, agent-based API using DiagnosisAgent architecture  
**Recommended for**: Production use, new integrations

**Start server:**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Endpoints:**

#### `GET /`
Health check endpoint
- Returns API status and available endpoints

#### `POST /predict`
Upload skin lesion image for classification
- **Request**: Multipart form data with image file
- **Response**: 
  ```json
  {
    "filename": "image.jpg",
    "prediction": "Melanoma",
    "confidence": 0.85,
    "probabilities": {"Melanoma": 0.85, "Benign": 0.15},
    "disclaimer": "..."
  }
  ```

#### `POST /predict_v2`
Symptom-based disease prediction with follow-up questions
- **Request**:
  ```json
  {
    "text": "I have a fever and headache",
    "top_k": 3,
    "follow_up_answers": {
      "q_malaria_fever_1": "yes"
    }
  }
  ```
- **Response**:
  ```json
  {
    "predictions": [
      {
        "disease": "Malaria",
        "probability": 0.85,
        "follow_up_questions": [...]
      }
    ]
  }
  ```

---

### 2. Legacy API: `src/main.py`
**Purpose**: Hybrid NLP+Rule-based prediction engine  
**Note**: Consider migrating to `src/api/main.py` for new projects

**Start server:**
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8001
```

**Endpoints:**

#### `POST /predict` (v1)
Basic symptom prediction

#### `POST /predict_v2` (v2)
Enhanced prediction with knowledge base rules

#### `POST /predict_skin`
Skin lesion classification

---

## Using the API

### Python Example:
```python
import requests

# Symptom prediction
response = requests.post(
    "http://localhost:8000/predict_v2",
    json={
        "text": "I have persistent cough and fever",
        "top_k": 3
    }
)
print(response.json())

# Image classification
files = {'file': open('skin_lesion.jpg', 'rb')}
response = requests.post(
    "http://localhost:8000/predict",
    files=files
)
print(response.json())
```

### cURL Example:
```bash
# Symptom prediction
curl -X POST "http://localhost:8000/predict_v2" \
  -H "Content-Type: application/json" \
  -d '{"text":"I have a headache and nausea","top_k":3}'

# Image upload
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg"
```

---

## Streamlit UI

The Streamlit interface provides a user-friendly way to interact with all features.

**Start Streamlit:**
```bash
streamlit run src/app_streamlit.py
```

The UI includes:
- 🏠 **Chatbot**: Interactive symptom-based diagnosis
- 🔬 **Skin Lesion Classifier**: Upload and analyze images
- 📜 **History**: View past conversations
- ⚙️ **Settings**: Configure language and preferences

---

## Configuration

### Environment Variables
Copy `.env.example` to `.env` and customize:
```bash
cp .env.example .env
```

### Centralized Configuration
All paths are defined in `src/config.py`:
```python
from src.config import PROJECT_ROOT, DATA_DIR, MODELS_DIR
```

---

## Deployment Notes

1. **Production Setup**: Use `uvicorn` with multiple workers
   ```bash
   uvicorn src.api.main:app --workers 4 --host 0.0.0.0 --port 8000
   ```

2. **CORS Configuration**: Add CORS middleware if serving web clients
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   app.add_middleware(CORSMiddleware, allow_origins=["*"])
   ```

3. **SSL/HTTPS**: Use nginx or a reverse proxy for SSL termination

4. **Rate Limiting**: Consider implementing rate limiting for production

---

## Troubleshooting

### Model Not Found
- Ensure models are trained: `python src/train_model.py`
- Check paths in `src/config.py`

### Import Errors
- Verify virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### API Connection Refused
- Check if server is running
- Verify port is not in use
- Check firewall settings
