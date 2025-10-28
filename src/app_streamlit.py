# src/app_streamlit.py

import streamlit as st
import requests
import json
import os
import traceback # To show detailed errors

# --- Smart Path Handling for Imports ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Local Agent Imports ---
from src.chatbot_system.recommendation_agent import RecommendationAgent
from src.chatbot_system.followup_manager import FollowUpManager
from src.radiology_locator_osm import RadiologyLocatorOSM
from src.pharmacy_locator_osm import PharmacyLocator

# --- Configuration ---
HISTORY_FILE = "history.json"
API_URL = "http://127.0.0.1:8000/predict_v2"

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

st.set_page_config(page_title="AI Healthcare Chatbot", page_icon="🩺", layout="wide")

# --- Texts / i18n (Internationalization) ---
TEXT = {
    "en": {
        "sidebar_title": "AI Healthcare Assistant 🤖",
        "page_chatbot": "🏠 Chatbot",
        "page_history": "📜 History",
        "page_settings": "⚙️ Settings",
        "page_radiology": "🩻 Radiology Locator",
        "page_pharmacy": "💊 Pharmacy Locator",
        "app_title": "AI Healthcare Chatbot",
        "app_sub": "Describe your symptoms for a preliminary analysis.",
        "your_answers": "Your Answers:",
        "final_predictions": "Final Predictions:",
        "disclaimer": "Disclaimer: This is not a medical diagnosis. Please consult a doctor.",
        "save_history": "💾 Save Analysis to History",
        "saving_ok": "📁 Analysis saved successfully!",
        "saving_failed": "⚠️ Failed to save history.",
        "clear_history": "🗑️ Clear History",
        "no_history": "No history yet. Perform an analysis first.",
        "language": "Language",
        "enter_location": "📍 Enter your location (e.g., 30.0444,31.2357):",
        "search_btn": "🔎 Search",
        "radius_m": "Search radius (meters):",
        "medicine_name": "💊 Medicine name (optional):",
        "no_results": "No results found nearby.",
        "invalid_location": "Invalid location format. Please use 'latitude,longitude'.",
        "found_locations": "Found {count} locations:",
        "start_new": "🔄 Start New Analysis",
        "initial_greeting": "Hello! I'm your AI Healthcare Assistant. Please describe your symptoms to begin.",
        "final_diagnosis_header": "### 🩺 Final Diagnosis Results:",
        "initial_diagnosis_header": "### 🩺 Initial Possible Diagnoses:",
        "updated_diagnosis_header": "Based on your answer, here are the updated possibilities:",
        "recommendations_for": "--- Recommendations for {disease} ---",
        "description": "**🧠 Description:**",
        "precautions": "**💡 Precautions:**",
        "thinking": "Analyzing...",
        "no_condition_found": "I couldn't find any matching condition. Please rephrase your symptoms or consult a doctor.",
        "no_clear_diagnosis": "I couldn't determine a clear diagnosis based on your answers. Please consult a doctor.",
        "api_error": "Could not connect to the diagnosis API. Please ensure the backend server is running.",
        "history_cleared": "History cleared successfully.",
        "internal_error": "An internal error occurred. Please check the logs or contact support." # New text
    },
    "ar": {
        "sidebar_title": "مساعد الرعاية الصحية 🤖",
        "page_chatbot": "🏠 البوت",
        "page_history": "📜 السجل",
        "page_settings": "⚙️ الإعدادات",
        "page_radiology": "🩻 مواقع الأشعة",
        "page_pharmacy": "💊 مواقع الصيدليات",
        "app_title": "بوت الرعاية الصحية الذكي",
        "app_sub": "صف أعراضك للحصول على تحليل مبدئي.",
        "your_answers": "إجاباتك:",
        "final_predictions": "التنبؤات النهائية:",
        "disclaimer": "تنبيه: هذا ليس تشخيصًا طبيًا. يرجى استشارة طبيب.",
        "save_history": "💾 حفظ التحليل في السجل",
        "saving_ok": "📁 تم حفظ التحليل بنجاح!",
        "saving_failed": "⚠️ فشل حفظ السجل.",
        "clear_history": "🗑️ مسح السجل",
        "no_history": "لا يوجد سجل بعد. قم بإجراء تحليل أولاً.",
        "language": "اللغة",
        "enter_location": "📍 اكتب موقعك (مثال: 30.0444,31.2357):",
        "search_btn": "🔎 بحث",
        "radius_m": "نطاق البحث (بالمتر):",
        "medicine_name": "💊 اسم الدواء (اختياري):",
        "no_results": "لم يتم العثور على نتائج قريبة.",
        "invalid_location": "صيغة الموقع غير صحيحة. الرجاء استخدام 'خط العرض,خط الطول'.",
        "found_locations": "تم العثور على {count} موقع:",
        "start_new": "🔄 ابدأ تحليل جديد",
        "initial_greeting": "أهلاً بك! أنا مساعدك الصحي الذكي. من فضلك صف أعراضك للبدء.",
        "final_diagnosis_header": "### 🩺 نتائج التشخيص النهائية:",
        "initial_diagnosis_header": "### 🩺 التشخيصات الأولية المحتملة:",
        "updated_diagnosis_header": "بناءً على إجابتك، هذه هي الاحتمالات المحدثة:",
        "recommendations_for": "--- توصيات لـ {disease} ---",
        "description": "**🧠 الوصف:**",
        "precautions": "**💡 الاحتياطات:**",
        "thinking": "جارٍ التحليل...",
        "no_condition_found": "لم أتمكن من إيجاد حالة مطابقة. الرجاء إعادة صياغة الأعراض أو استشارة طبيب.",
        "no_clear_diagnosis": "لم أتمكن من تحديد تشخيص واضح بناءً على إجاباتك. يرجى استشارة طبيب.",
        "api_error": "لا يمكن الاتصال بخادم التشخيص. الرجاء التأكد من أن الواجهة الخلفية تعمل.",
        "history_cleared": "تم مسح السجل بنجاح.",
        "internal_error": "حدث خطأ داخلي. يرجى مراجعة السجلات أو التواصل مع الدعم." # New text
    }
}


def t(key, **kwargs):
    lang = st.session_state.get("lang", "en")
    template = TEXT.get(lang, TEXT["en"]).get(key, key)
    # Basic error handling for formatting
    try:
        return template.format(**kwargs)
    except KeyError as e:
        print(f"Warning: Missing key '{e}' in translation for '{key}' in lang '{lang}'")
        return key # Return the key itself as fallback

def translate_text(text, dest_lang):
    if not text or GoogleTranslator is None or st.session_state.lang == dest_lang:
        return text
    try:
        # Add timeout to translator call
        return GoogleTranslator(source='auto', target=dest_lang).translate(text=text, timeout=5)
    except Exception as e:
        print(f"Translation failed: {e}")
        return text


# --- Singleton Agent Initialization using Streamlit's Caching ---
@st.cache_resource
def get_recommendation_agent():
    """Load agents that are static and don't change state."""
    try:
        rec = RecommendationAgent()
        return rec
    except Exception as e:
        st.error(f"Failed to load RecommendationAgent: {e}")
        return None # Return None if loading fails

rec_agent = get_recommendation_agent()
# followup_manager will be initialized in session_state


# --- Session State Initialization ---
def init_state():
    if "initialized" not in st.session_state:
        try:
            st.session_state.initialized = True
            st.session_state.lang = "en"
            st.session_state.history = []
            st.session_state.messages = []
            st.session_state.analysis_complete = False
            st.session_state.followup_mode = False
            st.session_state.current_question_id = None
            st.session_state.initial_symptom_text = ""
            st.session_state.predictions = []

            # --- Initialize FollowUpManager ---
            if "followup_manager" not in st.session_state:
                st.session_state.followup_manager = FollowUpManager(negative_boost_multiplier=-0.25)
            # ✅ Add check to ensure it's the correct type
            elif not isinstance(st.session_state.followup_manager, FollowUpManager):
                print("Warning: followup_manager in session state is not a FollowUpManager instance. Resetting.")
                st.session_state.followup_manager = FollowUpManager(negative_boost_multiplier=-0.25)


            load_history()

            if not st.session_state.messages:
                add_message("assistant", t("initial_greeting"))
        except Exception as e:
            st.error(f"Error during initial state setup: {e}")
            st.stop() # Stop execution if basic state setup fails


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                st.session_state.history = json.load(f)
        except (json.JSONDecodeError, IOError, Exception) as e: # Catch broader errors
            print(f"Error loading history file: {e}")
            st.session_state.history = [] # Reset history on error


def add_message(role, text):
     # Ensure text is a string
     safe_text = str(text) if text is not None else ""
     st.session_state.messages.append({"role": role, "text": safe_text})


def reset_chat():
    try:
        st.session_state.followup_manager = FollowUpManager(negative_boost_multiplier=-0.25)

        st.session_state.messages = []
        st.session_state.analysis_complete = False
        st.session_state.followup_mode = False
        st.session_state.current_question_id = None
        st.session_state.initial_symptom_text = ""
        st.session_state.predictions = []
        add_message("assistant", t("initial_greeting"))
    except Exception as e:
        st.error(f"Error resetting chat: {e}")


# --- Call FastAPI backend ---
def get_diagnosis_from_api(symptoms, followup_answers=None):
    payload = {
        "text": symptoms,
        "top_k": 3,
        "follow_up_answers": followup_answers or {}
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=20)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json()
    # More specific error handling
    except requests.exceptions.ConnectionError:
        print(f"API Connection Error: Failed to connect to {API_URL}")
        return None # Indicate connection failure
    except requests.exceptions.Timeout:
        print(f"API Timeout Error: Request to {API_URL} timed out")
        return None # Indicate timeout
    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
        return None # Indicate general request failure
    except json.JSONDecodeError as e:
        # Check if response object exists before accessing text
        response_text = "N/A"
        if 'response' in locals() and hasattr(response, 'text'):
            response_text = response.text[:200]
        print(f"API Response JSON decode failed: {e}. Response text: {response_text}")
        return {"error": "Invalid API response format"} # Indicate bad response format


# --- Chatbot Page ---
def render_chatbot_page():
    try: # Wrap main rendering logic in try-except
        st.markdown(f"<h1 style='text-align:center;'>🩺 {t('app_title')}</h1>", unsafe_allow_html=True)

        if st.button(t("start_new")):
            reset_chat()
            st.rerun()

        # Display existing messages
        for msg in st.session_state.get("messages", []): # Use .get for safety
            with st.chat_message(msg.get("role", "assistant")): # Default role for safety
                st.markdown(msg.get("text", "")) # Default text

        # Handle new user input
        if prompt := st.chat_input("Describe your symptoms or answer the question..."):
            add_message("user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process the input and get bot response
            with st.chat_message("assistant"):
                with st.spinner(t("thinking")):
                    api_response = None

                    # --- Follow-up Logic ---
                    if st.session_state.get("followup_mode"): # Use .get
                        try:
                            low = prompt.strip().lower()
                            normalized_answer = "partial_yes" # Default
                            if hasattr(st.session_state.followup_manager, '_normalize_answer'):
                                normalized_answer = st.session_state.followup_manager._normalize_answer(low)

                            q_id = st.session_state.get("current_question_id")
                            if q_id:
                                st.session_state.followup_manager.record_answer(q_id, normalized_answer)
                            else:
                                print("Warning: Tried to record answer but current_question_id is missing.")


                            simple_answers = {}
                            if hasattr(st.session_state.followup_manager, 'get_all_answers_simple'):
                                simple_answers = st.session_state.followup_manager.get_all_answers_simple()

                            api_response = get_diagnosis_from_api(
                                st.session_state.get("initial_symptom_text", ""), # Use .get
                                simple_answers
                            )
                        except Exception as e:
                            st.error(f"{t('internal_error')} (Follow-up handling): {e}")
                            print(traceback.format_exc()) # Print full traceback to console
                            return # Stop processing this turn on error

                    # --- Initial Symptom Logic ---
                    else:
                        try:
                            st.session_state.initial_symptom_text = prompt
                            # Clear only if it's a *new* initial symptom description
                            if st.session_state.followup_manager.user_answers or st.session_state.followup_manager.pending_questions:
                                st.session_state.followup_manager.clear()
                            api_response = get_diagnosis_from_api(prompt, {})
                        except Exception as e:
                            st.error(f"{t('internal_error')} (Initial processing): {e}")
                            print(traceback.format_exc())
                            return

                    # --- Process API Response ---
                    if api_response is None:
                        # Error message is already printed by get_diagnosis_from_api if connection failed
                        st.error(t("api_error"))
                        add_message("assistant", t("api_error"))
                        return
                    if isinstance(api_response, dict) and api_response.get("error"):
                        # Handle specific API response format error
                        st.error(f"API Response Error: {api_response['error']}")
                        add_message("assistant", f"API Response Error: {api_response['error']}")
                        return


                    try:
                        st.session_state.predictions = api_response.get("predictions", [])
                        if not isinstance(st.session_state.predictions, list):
                            print(f"Warning: API returned non-list predictions: {st.session_state.predictions}")
                            st.session_state.predictions = []

                        all_followups = []
                        for pred in st.session_state.predictions:
                            if isinstance(pred, dict):
                                follow_ups = pred.get("follow_up_questions", [])
                                if isinstance(follow_ups, list):
                                    all_followups.extend(follow_ups)

                        # Add questions without clearing first (fix for infinite loop)
                        st.session_state.followup_manager.add_questions(all_followups)

                        # Build response text
                        header = t("updated_diagnosis_header") if st.session_state.get("followup_mode") else t("initial_diagnosis_header")
                        response_text = header + "\n"
                        if not st.session_state.predictions:
                            response_text += t("no_condition_found")
                        else:
                            sorted_preds = sorted(st.session_state.predictions, key=lambda p: p.get('probability', 0.0), reverse=True)
                            for p in sorted_preds:
                                disease = p.get("disease", "Unknown")
                                prob = p.get("probability", 0.0)
                                try:
                                    prob_text = f"{prob:.2%}"
                                except (TypeError, ValueError):
                                    prob_text = "N/A"
                                response_text += f"- **{disease}** (Likelihood: {prob_text})\n"

                        st.markdown(response_text)
                        add_message("assistant", response_text)

                        # Ask next question or finalize
                        top_disease = None
                        if st.session_state.predictions:
                            sorted_preds_for_next_q = sorted(st.session_state.predictions, key=lambda p: p.get('probability', 0.0), reverse=True)
                            # Ensure there's at least one prediction before accessing index 0
                            if sorted_preds_for_next_q:
                                top_disease = sorted_preds_for_next_q[0].get("disease")


                        next_q = None
                        if hasattr(st.session_state.followup_manager, 'get_next_question_for_active_disease'):
                            next_q = st.session_state.followup_manager.get_next_question_for_active_disease(top_disease)
                        elif hasattr(st.session_state.followup_manager, 'has_pending_questions') and st.session_state.followup_manager.has_pending_questions():
                            next_q = st.session_state.followup_manager.get_next_question()

                        if next_q:
                            st.session_state.followup_mode = True
                            q_id = next_q.get("id")
                            q_text_raw = next_q.get("text", "") or next_q.get("question", "")

                            if not q_id or not q_text_raw:
                                print(f"Warning: Got invalid next question object: {next_q}")
                                # Attempt to get the next global question as a fallback if the first was invalid
                                next_q = st.session_state.followup_manager.get_next_question() # Try again
                                if next_q and isinstance(next_q, dict):
                                    q_id = next_q.get("id")
                                    q_text_raw = next_q.get("text", "") or next_q.get("question", "")
                                    if not q_id or not q_text_raw: # Still invalid? Give up.
                                        st.session_state.followup_mode = False
                                        st.session_state.analysis_complete = True
                                        display_final_recommendations()
                                        return
                                else: # Still invalid or None? Give up.
                                    st.session_state.followup_mode = False
                                    st.session_state.analysis_complete = True
                                    display_final_recommendations()
                                    return

                            # Now we should have a valid q_id and q_text_raw
                            st.session_state.current_question_id = q_id
                            q_text = translate_text(q_text_raw, st.session_state.lang)
                            q_response = f"{q_text} (yes/no/maybe)"
                            st.markdown(q_response)
                            add_message("assistant", q_response)

                        else: # No more pending questions
                            st.session_state.followup_mode = False
                            st.session_state.analysis_complete = True
                            display_final_recommendations()

                    except Exception as e:
                        st.error(f"{t('internal_error')} (Processing response): {e}")
                        print(traceback.format_exc())
                        # Attempt to gracefully end if response processing fails
                        st.session_state.followup_mode = False
                        st.session_state.analysis_complete = True
                        display_final_recommendations() # Show recommendations based on last known state

            # --- ✅ START: Corrected Indentation & MODIFIED Save Logic ---
            # --- Save to History Button ---
            if st.session_state.get("analysis_complete"): # Use .get
                if st.button(t("save_history")):
                    
                    # ✅ Import datetime here, only when needed
                    from datetime import datetime 
                    
                    # Get the full answers data {qid: {answer: 'yes', ...}}
                    full_answers_data = {}
                    try:
                        if hasattr(st.session_state.followup_manager, 'get_all_answers'):
                            full_answers_data = st.session_state.followup_manager.get_all_answers()
                    except Exception as e:
                        print(f"Error getting full answers for history: {e}")

                    # Get the question metadata {qid: {text: '...', ...}}
                    question_metadata = {}
                    try:
                        if hasattr(st.session_state.followup_manager, 'question_meta'):
                            question_metadata = st.session_state.followup_manager.question_meta
                    except Exception as e:
                        print(f"Error getting question_meta for history: {e}")

                    # --- ✅ NEW: Build the answers dictionary with text ---
                    answers_for_history = {}
                    if isinstance(full_answers_data, dict):
                        for qid, ans_data in full_answers_data.items():
                            question_text = "N/A" # Default
                            if qid in question_metadata and isinstance(question_metadata[qid], dict):
                                question_text = question_metadata[qid].get("text", "N/A")
                            
                            answer_text = "N/A" # Default
                            if isinstance(ans_data, dict):
                                answer_text = ans_data.get("answer", "N/A")
                            elif ans_data: # Fallback if data is not a dict
                                answer_text = str(ans_data)

                            answers_for_history[qid] = {
                                "question_text": question_text,
                                "answer_text": answer_text
                            }
                    
                    entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symptoms": st.session_state.get("initial_symptom_text", ""),
                        "answers": answers_for_history, # Use the new, richer dictionary
                        "predictions": st.session_state.get("predictions", []),
                        "lang": st.session_state.get("lang", "en")
                    }

                    # Ensure history is initialized
                    if "history" not in st.session_state: st.session_state.history = []
                    st.session_state.history.append(entry)

                    try:
                        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
                        st.success(t("saving_ok"))
                    except Exception as e:
                        st.error(f"{t('saving_failed')}: {e}")
            # --- ✅ END: Corrected Indentation & MODIFIED Save Logic ---

    except Exception as e: # Catch errors in the main render function
        st.error(f"An unexpected error occurred in the chatbot interface: {e}")
        print(traceback.format_exc())


def display_final_recommendations():
    try: # Wrap in try-except
        final_preds = st.session_state.get("predictions", []) # Use .get
        if not final_preds or not isinstance(final_preds, list): # Check if list and not empty
            response = t("no_clear_diagnosis")
            st.markdown(response)
            add_message("assistant", response)
            return

        # Safely sort and get top prediction
        top_disease = "N/A"
        top_prob = 0.0
        try:
            # Filter out potential non-dict items before sorting
            valid_preds = [p for p in final_preds if isinstance(p, dict)]
            if valid_preds:
                sorted_final_preds = sorted(valid_preds, key=lambda p: p.get('probability', 0.0), reverse=True)
                top_disease = sorted_final_preds[0].get("disease", "N/A")
                top_prob = sorted_final_preds[0].get("probability", 0.0)
        except Exception as e:
            print(f"Error sorting final predictions: {e}")
            # Keep default N/A values


        rec = {}
        if top_disease != "N/A" and rec_agent: # Check if rec_agent loaded
            try:
                rec = rec_agent.get_details(top_disease)
            except Exception as e:
                print(f"Error getting recommendations for {top_disease}: {e}")
                rec = {}

        # Format probability safely
        try:
            prob_text = f"{top_prob:.2%}"
        except (TypeError, ValueError):
            prob_text = "N/A"

        conclusion_message = f"✅ Diagnosis complete. Based on the analysis, the most likely condition is **{top_disease}** (Likelihood: {prob_text}).\n\n"
        st.markdown(conclusion_message)
        add_message("assistant", conclusion_message)


        rec_response = t("recommendations_for", disease=top_disease or "N/A") + "\n"
        # Safely get description and precautions
        description = rec.get('description', 'N/A')
        precautions = rec.get('precautions', [])

        rec_response += t("description") + f" {description}\n"


        if precautions and isinstance(precautions, list): # Ensure precautions is a list
            rec_response += t("precautions") + "\n"
            safe_precautions = [str(p) for p in precautions if p is not None]
            if safe_precautions: # Only add list if there are actual precautions
                for p in safe_precautions:
                    rec_response += f"- {p}\n"
            else:
                rec_response += f"\n_No specific precautions listed. Please consult a healthcare professional._\n"
        else:
            rec_response += f"\n_Please consult a healthcare professional for specific advice regarding {top_disease or 'this condition'}._\n"


        st.markdown(rec_response)
        add_message("assistant", rec_response)
    except Exception as e:
        st.error(f"Error displaying final recommendations: {e}")
        print(traceback.format_exc())


# --- History Page ---
def render_history_page():
    try: # Wrap in try-except
        st.title(t("page_history"))
        history_list = st.session_state.get("history", []) # Use .get
        if not history_list:
            st.info(t("no_history"))
        else:
            sorted_history = sorted(history_list, key=lambda x: x.get('timestamp', ''), reverse=True)

            for entry in sorted_history:
                lang = entry.get("lang", "en")
                heading = f"**{entry.get('timestamp', '')}** — {entry.get('symptoms', '')}"
                with st.expander(heading):
                    st.write(f"**{t('final_predictions')}**")
                    preds_in_history = entry.get("predictions", [])
                    if isinstance(preds_in_history, list) and preds_in_history:
                        sorted_preds_hist = sorted(preds_in_history, key=lambda p: p.get('probability', 0.0), reverse=True)
                        for p in sorted_preds_hist:
                            disease = translate_text(p.get("disease", ""), lang) if p.get("disease") else ""
                            prob_val = p.get('probability', 0.0)
                            try:
                                prob = f"{prob_val * 100:.1f}%" if isinstance(prob_val, (int, float)) else "N/A"
                            except (TypeError, ValueError):
                                prob = "N/A"
                            st.write(f"- {disease} ({prob})")
                    else:
                        st.write("_(No prediction data available)_")

                    # --- ✅ START: MODIFIED History Answer Rendering ---
                    st.write(f"**{t('your_answers')}**")
                    answers = entry.get("answers", {}) or {}
                    if isinstance(answers, dict) and answers:
                        for q_id in sorted(answers.keys()):
                            ans_data = answers[q_id] # This is now {'question_text': '...', 'answer_text': '...'}
                            
                            # --- ✅ NEW: Read the saved text ---
                            if isinstance(ans_data, dict):
                                q_display = f"_{ans_data.get('question_text', 'N/A')}_"
                                ans_text = ans_data.get('answer_text', 'N/A')
                            else:
                                # Fallback for old history format
                                q_display = f"**{q_id}**"
                                ans_text = str(ans_data)

                            st.write(f"- {q_display}: {ans_text.capitalize()}")
                    else:
                        st.write("- None")
                    # --- ✅ END: MODIFIED History Answer Rendering ---

            # Move button outside the loop
            if st.button(t("clear_history")):
                st.session_state.history = []
                try:
                    if os.path.exists(HISTORY_FILE):
                        os.remove(HISTORY_FILE)
                    st.success(t("history_cleared"))
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear history: {e}")
    except Exception as e:
        st.error(f"Error rendering history page: {e}")
        print(traceback.format_exc())


# --- Radiology Locator Page ---
def render_radiology_page():
    try: # Wrap in try-except
        st.title(t("page_radiology"))
        latlon_input = st.text_input(t("enter_location"), value="30.0444,31.2357", key="rad_latlon")
        radius = st.number_input(t("radius_m"), value=5000, min_value=500, step=500, key="rad_radius")

        if st.button(t("search_btn"), key="rad_search"):
            try:
                if ',' not in latlon_input: raise ValueError("Comma missing")
                lat_str, lon_str = latlon_input.replace(" ", "").split(",")
                lat, lon = float(lat_str), float(lon_str)
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError("Latitude/Longitude out of range")

                with st.spinner("Searching for radiology centers..."):
                    locator = RadiologyLocatorOSM()
                    results = locator.get_nearby_radiology_centers(lat, lon, radius)

                if not results or "error" in results:
                    st.error(results.get("error", "An error occurred during search."))
                elif not results.get("results"):
                    st.info(t("no_results"))
                else:
                    res_list = results['results']
                    st.success(t("found_locations", count=len(res_list)))
                    for r in res_list[:20]:
                        st.markdown(f"#### {r.get('name', 'N/A')}")
                        dist_km = r.get('distance_km', 'N/A')
                        dist_text = f"{dist_km} km" if isinstance(dist_km, (int, float)) else "N/A"
                        st.markdown(f"📍 **Address:** {r.get('address', 'N/A')} | 🚗 **Distance:** {dist_text}")
                        website = r.get('website', 'N/A')
                        website_link = f"[{website}]({website})" if website and website != 'N/A' and website.startswith('http') else 'N/A' # Add http check
                        st.markdown(f"📞 **Phone:** {r.get('phone', 'N/A')} | 🌐 **Website:** {website_link}", unsafe_allow_html=True)
                        st.markdown("---")

            except (ValueError, IndexError) as e:
                print(f"Location input error: {e}")
                st.error(t("invalid_location"))
            except Exception as e:
                print(f"Radiology search failed unexpectedly: {e}")
                st.error("An unexpected error occurred during the search.")
    except Exception as e:
        st.error(f"Error rendering radiology page: {e}")
        print(traceback.format_exc())


# --- Pharmacy Locator Page ---
def render_pharmacy_page():
    try: # Wrap in try-except
        st.title(t("page_pharmacy"))
        latlon_input = st.text_input(t("enter_location"), value="30.0444,31.2357", key="ph_latlon")
        radius = st.number_input(t("radius_m"), value=5000, min_value=500, step=500, key="ph_radius")
        med_name = st.text_input(t("medicine_name"), key="ph_med_name")

        if st.button(t("search_btn"), key="ph_search"):
            try:
                if ',' not in latlon_input: raise ValueError("Comma missing")
                lat_str, lon_str = latlon_input.replace(" ", "").split(",")
                lat, lon = float(lat_str), float(lon_str)
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError("Latitude/Longitude out of range")

                with st.spinner("Searching for pharmacies..."):
                    locator = PharmacyLocator()
                    search_med_name = med_name.strip() if med_name else None
                    results = locator.get_nearby_pharmacies(lat, lon, radius, search_med_name)

                if not results or "error" in results:
                    st.error(results.get("error", "An error occurred during search."))
                elif not results.get("results"):
                    st.info(t("no_results"))
                else:
                    res_list = results['results']
                    st.success(t("found_locations", count=len(res_list)))
                    for r in res_list[:20]:
                        st.markdown(f"#### {r.get('name', 'N/A')}")
                        dist_km = r.get('distance_km', 'N/A')
                        dist_text = f"{dist_km} km" if isinstance(dist_km, (int, float)) else "N/A"
                        st.markdown(f"📍 **Address:** {r.get('address', 'N/A')} | 🚗 **Distance:** {dist_text}")
                        website = r.get('website', 'N/A')
                        website_link = f"[{website}]({website})" if website and website != 'N/A' and website.startswith('http') else 'N/A' # Add http check
                        st.markdown(f"📞 **Phone:** {r.get('phone', 'N/A')} | 🌐 **Website:** {website_link}", unsafe_allow_html=True)
                        st.markdown("---")

            except (ValueError, IndexError) as e:
                print(f"Location input error: {e}")
                st.error(t("invalid_location"))
            except Exception as e:
                print(f"Pharmacy search failed unexpectedly: {e}")
                st.error("An unexpected error occurred during the search.")
    except Exception as e:
        st.error(f"Error rendering pharmacy page: {e}")
        print(traceback.format_exc())


# --- Settings Page ---
def render_settings_page():
    try: # Wrap in try-except
        st.title(t("page_settings"))

        current_lang_index = 0 if st.session_state.get("lang", "en") == "en" else 1

        lang_choice = st.radio(
            t("language"),
            options=["English", "العربية"],
            index=current_lang_index,
            key="lang_radio"
        )

        new_lang = "en" if lang_choice == "English" else "ar"

        if new_lang != st.session_state.get("lang", "en"):
            st.session_state.lang = new_lang
            st.rerun()
    except Exception as e:
        st.error(f"Error rendering settings page: {e}")
        print(traceback.format_exc())


# --- Main App Router ---
def main():
    try: # Wrap main execution
        init_state()

        page_keys = {
            "chatbot": "page_chatbot",
            "radiology": "page_radiology",
            "pharmacy": "page_pharmacy",
            "history": "page_history",
            "settings": "page_settings",
        }

        page_render_funcs = {
            "chatbot": render_chatbot_page,
            "radiology": render_radiology_page,
            "pharmacy": render_pharmacy_page,
            "history": render_history_page,
            "settings": render_settings_page,
        }

        st.sidebar.title(t("sidebar_title"))

        selection = st.sidebar.radio(
            "Navigation",
            options=list(page_keys.keys()),
            format_func=lambda key: t(page_keys.get(key, key)), # Safe format_func
            label_visibility="collapsed",
            key="navigation_select"
        )

        page_to_render = page_render_funcs.get(selection)

        if page_to_render:
            page_to_render()
        else:
            st.error("Selected page not found.")
            render_chatbot_page() # Default to chatbot page
            
    except Exception as e:
        # Display error prominently in the UI if main fails
        st.error(f"An unexpected error occurred in the main application: {e}")
        print(traceback.format_exc()) # Also log it


if __name__ == "__main__":
    main()