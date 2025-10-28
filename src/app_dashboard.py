# D:\disease_prediction_project\src\app_dashboard.py
import json
import os
import streamlit as st
import pandas as pd
from collections import Counter
from statistics import mean
import plotly.express as px


# ===============================
# Load chatbot history
# ===============================
def load_history(file_path="history.json"):
    if not os.path.exists(file_path):
        st.warning("⚠️ No history file found.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            st.error("❌ Error: history.json is not a valid JSON file.")
            return []


# ===============================
# Analyze the data
# ===============================
def analyze_history(history):
    if not history:
        return None

    all_diseases = []
    followup_counts = []
    yes_count = 0
    no_count = 0

    for session in history:
        # Diseases diagnosed
        for pred in session.get("predictions", []):
            all_diseases.append(pred.get("disease"))

        # ✅ --- START: MODIFIED LOGIC ---
        # Read from the new "answers" key, not the old "followups" key
        
        # Get the new 'answers' dictionary {qid: {answer_text: '...', ...}}
        answers = session.get("answers", {}) 
        
        # Count follow-up Qs
        count = len(answers) # Count is just the number of keys
        followup_counts.append(count)

        # Count yes/no from the new structure
        for qid, ans_data in answers.items():
            if isinstance(ans_data, dict):
                # Get the saved answer text
                answer_text = ans_data.get("answer_text", "").lower() 
                
                if answer_text == "yes": # The normalized answer
                    yes_count += 1
                elif answer_text == "no": # The normalized answer
                    no_count += 1
        # ✅ --- END: MODIFIED LOGIC ---

    return {
        "total_sessions": len(history),
        "avg_followups": mean(followup_counts) if followup_counts else 0,
        "yes_count": yes_count,
        "no_count": no_count,
        "top_diseases": Counter(all_diseases).most_common(10),
        "followup_counts_list": followup_counts # ✅ Pass the list for the trend chart
    }


# ===============================
# Streamlit Dashboard UI
# ===============================
def main():
    st.set_page_config(page_title="AI Healthcare Chatbot Dashboard", layout="wide")

    st.title("🏥 AI Healthcare Chatbot Dashboard")
    st.markdown("### 🔍 Performance & User Interaction Analytics")

    # Load and analyze
    # ✅ Use the project root to find the history file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    history_path = os.path.join(base_dir, 'history.json')
    
    history = load_history(history_path)
    data = analyze_history(history)

    if not data:
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sessions", data["total_sessions"])
    col2.metric("Avg Follow-ups per Session", f"{data['avg_followups']:.2f}")
    col3.metric("Total Questions Answered", data["yes_count"] + data["no_count"])

    st.divider()

    # ✅ Yes/No ratio chart (This part was correct)
    st.subheader("🗣️ Yes vs No Answers")
    yes_no_df = pd.DataFrame({
        "Answer": ["Yes", "No"],
        "Count": [data["yes_count"], data["no_count"]]
    })
    fig1 = px.pie(yes_no_df, names="Answer", values="Count", title="Follow-up Answers Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # 🏥 Top diseases chart (This part was correct)
    st.subheader("🏥 Top 10 Predicted Diseases")
    disease_df = pd.DataFrame(data["top_diseases"], columns=["Disease", "Count"])
    fig2 = px.bar(disease_df, x="Disease", y="Count", color="Disease",
                 title="Most Frequently Predicted Diseases")
    st.plotly_chart(fig2, use_container_width=True)

    # 🔁 Average follow-ups
    st.subheader("📊 Follow-up Questions per Session")
    
    # ✅ --- START: MODIFIED LOGIC ---
    # Get the correct list from the analysis, don't recalculate it incorrectly
    followup_counts = data.get("followup_counts_list", [])
    # ✅ --- END: MODIFIED LOGIC ---
    
    followup_df = pd.DataFrame({"Session": range(1, len(followup_counts) + 1), "Follow-up Count": followup_counts})
    fig3 = px.line(followup_df, x="Session", y="Follow-up Count", markers=True,
                  title="Follow-up Questions Trend Over Time")
    st.plotly_chart(fig3, use_container_width=True)

    st.success("✅ Dashboard loaded successfully!")


if __name__ == "__main__":
    main()