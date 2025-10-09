import streamlit as st
import sys
import os
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot_system.chatbot_manager import ChatbotManager
from src.utils.report_generator import generate_medical_report
from src.api_requester import APIRequester

# ---------------------------
# ⚙️ Page Config & Caching
# ---------------------------
st.set_page_config(page_title="AI Healthcare Chatbot", page_icon="🩺", layout="centered")

@st.cache_resource
def load_chatbot_manager():
    return ChatbotManager()

@st.cache_resource
def load_api_requester():
    return APIRequester()

# ---------------------------
# 🎨 Custom CSS
# ---------------------------
st.markdown("""
<style>
    /* CSS remains the same as your well-designed version */
    .chat-message { padding: 10px 16px; margin: 8px 0; border-radius: 12px; line-height: 1.6; font-size: 16px; }
    .chat-message.user { background-color: #DCF8C6; color: #000; text-align: right; border-top-right-radius: 0; }
    .chat-message.bot { background-color: #F1F0F0; color: #000; text-align: left; border-top-left-radius: 0; }
    .title { text-align: center; font-size: 26px; color: #007ACC; font-weight: bold; }
    .sidebar .block-container { padding-top: 1rem; }
    .stChatInput { border-radius: 12px !important; }
    .stMarkdown { font-family: 'Segoe UI', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 🏷️ App Header
# ---------------------------
st.markdown("<div class='title'>🩺 AI Healthcare Chatbot</div>", unsafe_allow_html=True)
st.markdown("Welcome! Describe your symptoms, and I’ll provide a preliminary analysis.")
st.divider()

# ---------------------------
# 🧠 Chat Logic and State
# ---------------------------
chatbot = load_chatbot_manager()
api_requester = load_api_requester()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How are you feeling today? Please describe your symptoms."}]

# ---------------------------
# 🛠️ Sidebar with Controls
# ---------------------------
with st.sidebar:
    st.title("Controls")
    
    current_hour = datetime.now().hour
    greeting = "🌅 Good morning!" if current_hour < 12 else "☀️ Good afternoon!" if current_hour < 18 else "🌙 Good evening!"
    st.write(greeting)

    # --- Mode Switch ---
    mode = st.radio(
        "Select Chatbot Mode:",
        ["Local Mode (Internal Model)", "API Mode (Connect to FastAPI)"],
        help="Local mode runs the model inside this app. API mode connects to a separate server."
    )
    
    if st.button("🔄 New Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "New chat started. How can I help?"}]
        st.rerun()

    if st.button("💾 Save Chat"):
        # Save chat logic remains the same
        chat_history = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages])
        st.download_button("Download Chat History", chat_history, "chat_history.txt", "text/plain")
        st.sidebar.success("Chat history ready for download!")

# ---------------------------
# 💬 Display Chat History
# ---------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# ---------------------------
# 📝 User Input and Chat Logic
# ---------------------------
if prompt := st.chat_input("Describe your symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Analyzing your symptoms... 🧑‍⚕️")
        
        try:
            # --- Emergency Detection ---
            critical_keywords = [
                "chest pain", "shortness of breath", "difficulty breathing", "unconscious",
                "seizure", "stroke", "heart attack", "confusion", "severe bleeding", "loss of vision"
            ]
            lower_input = prompt.lower()
            is_critical = any(word in lower_input for word in critical_keywords)

            if is_critical:
                st.markdown("""
                    <style>
                    @keyframes flash {
                        0% { background-color: #ff0000; }
                        50% { background-color: #fff; }
                        100% { background-color: #ff0000; }
                    }
                    .emergency-box {
                        animation: flash 1s infinite;
                        color: white;
                        background-color: #ff0000;
                        border-radius: 12px;
                        padding: 20px;
                        font-size: 18px;
                        text-align: center;
                        font-weight: bold;
                    }
                    </style>
                    <div class='emergency-box'>
                        🚨 EMERGENCY ALERT! 🚨<br><br>
                        Your symptoms may indicate a serious medical condition.<br>
                        <b>Please call emergency services or go to the nearest hospital immediately.</b><br><br>
                        ⚕️ It’s important to get professional help right away.
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("---")
                st.subheader("Quick Actions")

                col1, col2 = st.columns(2)
                with col1:
                    st.link_button("🚑 Call Emergency (123)", "tel:123", use_container_width=True)
                with col2:
                    st.link_button("🏥 Find Nearest Hospital", "https://www.google.com/maps/search/hospitals+near+me", use_container_width=True)

                st.session_state.messages.append({"role": "assistant", "content": "🚨 Emergency alert triggered! Please take immediate action using the buttons above."})
                st.stop()

            # --- Main Diagnosis Logic ---
            predictions = []
            if "API Mode" in mode:
                result = api_requester.send_request(prompt)
                if "error" in result:
                    full_response = f"⚠️ API Error: {result['error']}"
                    predictions = []
                else:
                    predictions = result.get("predictions", [])
            else: # Local Mode
                symptom_text = chatbot.symptom_agent.collect_symptoms(prompt)
                predictions = chatbot.diagnosis_agent.predict_top_diseases(symptom_text)

            # --- Response Formatting ---
            if predictions:
                full_response = "" # Build the styled response
                for pred in predictions:
                    # ... (The styled box generation code is unchanged)
                    disease, prob = pred["disease"], pred["probability"]
                    rec = chatbot.recommendation_agent.get_recommendations(disease)
                    color, emoji = ("#ff4c4c", "🚨") if rec['risk_level'] == "critical" else ("#ffcc00", "🟡") if rec['risk_level'] == "moderate" else ("#4caf50", "🟢")
                    maps_link = f"https://www.google.com/maps/search/{rec['specialist'].replace(' ', '+')}+near+me"
                    full_response += f"""
                    <div style='background-color:{color}20; border-left:6px solid {color}; padding:15px; border-radius:10px; margin-top:15px;'>
                        <h4>{emoji} <b>{disease}</b></h4>
                        <p><b>📊 Likelihood:</b> {prob:.1%}</p>
                        <p><b>🧪 Recommended Tests:</b> {", ".join(rec['tests'])}</p>
                        <p><b>💡 Advice:</b> {rec['advice']}</p>
                        <p><b>👨‍⚕️ Specialist:</b> {rec['specialist']}</p>
                        <p><b>📍 <a href="{maps_link}" target="_blank">Find Nearby {rec['specialist']}</a></b></p>
                    </div>
                    """
                full_response += "<br><br>⚠️ <b>Disclaimer:</b> This is an AI-generated analysis and not a substitute for professional medical advice."
                message_placeholder.markdown(full_response, unsafe_allow_html=True)

                # --- PDF Report Generation Section ---
                with st.expander("🧾 Download Your Medical Report"):
                    patient_name = st.text_input("Enter your name for the report (optional):", key="patient_name_input")
                    if st.button("📄 Generate & Prepare Download"):
                        report_path = generate_medical_report(
                            user_name=patient_name if patient_name else "Anonymous",
                            user_input=prompt,
                            predictions=predictions,
                            recommendations=[chatbot.recommendation_agent.get_recommendations(p["disease"]) for p in predictions]
                        )
                        with open(report_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download PDF Report", f, os.path.basename(report_path), "application/pdf"
                            )
                        st.success("Your report is ready for download!")
            else:
                full_response = "I couldn't identify a likely condition based on your description. Could you please provide more details?"
                message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"⚠️ An error occurred: {e}"
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    if len(st.session_state.messages) == 3:
        st.rerun()

