from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
import os

# --- Custom Text Cleaner (must match the one used in training) ---
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [re.sub(r"[^a-z\s]", "", str(text).lower()) for text in X]

# 🔑 Register TextCleaner so joblib can find it
import __main__
__main__.TextCleaner = TextCleaner

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Symptom-Based Disease Prediction API",
    description="An API that predicts diseases based on symptom descriptions using an NLP model.",
    version="1.0.0"
)

# --- Load ML Model and Label Encoder ---
try:
    print("📂 Current working dir:", os.getcwd())
    print("📂 Files in dir:", os.listdir())

    model_pipeline = joblib.load("optimized_nlp_pipeline.joblib")
    label_encoder = joblib.load("nlp_label_encoder.joblib")
    print("✅ Model and encoder loaded successfully!")

except Exception as e:
    model_pipeline = None
    label_encoder = None
    print("❌ Error loading model files:", e)


# --- Request Schema ---
class SymptomRequest(BaseModel):
    text: str
    top_k: int = 3


# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Symptom-Based Disease Prediction API"}


# --- Prediction Endpoint ---
@app.post("/predict")
def predict_disease(request: SymptomRequest):
    if model_pipeline is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please contact the administrator.")
    
    try:
        # Transform input
        text_input = [request.text]
        probabilities = model_pipeline.predict_proba(text_input)[0]

        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-request.top_k:][::-1]
        predictions = [
            {"disease": label_encoder.inverse_transform([i])[0],
             "probability": float(probabilities[i])}
            for i in top_indices
        ]

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
