import streamlit as st
import requests

# --- Title ---
st.title("🧑‍⚕️ Symptom-based Disease Prediction")

st.write("Enter your symptoms below and the AI model will predict possible diseases.")

# --- Input ---
symptoms = st.text_input("✍️ Enter symptoms (e.g., fever headache chills):")
top_k = st.slider("🔢 Number of predictions to show:", 1, 5, 3)

# --- Button to Predict ---
if st.button("✅ Predict"):
    if symptoms.strip():
        try:
            # Send request to FastAPI
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": symptoms, "top_k": top_k}
            )

            if response.status_code == 200:
                predictions = response.json().get("predictions", [])
                if predictions:
                    st.subheader("✅ Predictions:")
                    for pred in predictions:
                        st.write(f"**{pred['disease']}** - Probability: {pred['probability']:.2f}")
                else:
                    st.warning("⚠️ No predictions returned.")
            else:
                st.error(f"⚠️ API Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"⚠️ Connection Error: {e}")
    else:
        st.warning("⚠️ Please enter symptoms before predicting.")
