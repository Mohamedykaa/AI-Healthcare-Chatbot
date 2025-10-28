import json
import os
from collections import Counter
from statistics import mean, StatisticsError

class HistoryAnalyzer:
    """Analyzes chatbot conversation history and extracts useful insights."""

    def __init__(self, history_path="history.json"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        self.history_path = os.path.join(project_root, history_path)
        self.history = []
        self.load_history()

    def load_history(self):
        """Loads JSON history from file."""
        if not os.path.exists(self.history_path):
            print(f"⚠️ No history file found at '{self.history_path}'. Ensure the file exists.")
            return
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                self.history = json.load(f)
                print(f"✅ History loaded successfully from '{self.history_path}' ({len(self.history)} sessions).")
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON format in history file '{self.history_path}'.")
            self.history = []
        except Exception as e:
            print(f"❌ Error loading history file: {e}")
            self.history = []

    def analyze(self):
        """Generate full report based on history data."""
        if not self.history:
            print("⚠️ No data available to analyze.")
            return None

        total_sessions = len(self.history)
        all_diseases = []
        answer_counts_per_session = []
        yes_count = 0
        no_count = 0

        for session in self.history:
            for pred in session.get("predictions", []):
                disease_name = pred.get("disease")
                if disease_name:
                    all_diseases.append(disease_name)

            answers = session.get("answers", {})
            answer_counts_per_session.append(len(answers))

            for qid, answer_data in answers.items():
                if isinstance(answer_data, dict):
                    answer_text = answer_data.get("answer_text", "").lower()
                    if answer_text == "yes":
                        yes_count += 1
                    elif answer_text == "no":
                        no_count += 1
                else: # Fallback for old format
                    answer_text_old = str(answer_data).lower()
                    if answer_text_old == "yes":
                        yes_count += 1
                    elif answer_text_old == "no":
                        no_count += 1

        # Calculate statistics
        most_common = Counter(all_diseases).most_common(5) if all_diseases else []
        try:
             avg_answers = mean(answer_counts_per_session) if answer_counts_per_session else 0
        except StatisticsError: # Handle case if list is empty or contains non-numeric
             avg_answers = 0

        # ✅ --- START: Added Yes/No Ratio Calculation ---
        total_yes_no = yes_count + no_count
        yes_ratio = (yes_count / total_yes_no * 100) if total_yes_no > 0 else 0
        # ✅ --- END: Added Yes/No Ratio Calculation ---

        summary_data = {
            "total_sessions": total_sessions,
            "avg_answers_per_session": round(avg_answers, 2),
            "total_yes_answers": yes_count,
            "total_no_answers": no_count,
            "yes_answer_ratio_percent": round(yes_ratio, 1), # ✅ Added ratio to summary
            "top_diseases": [{"disease": d, "count": c} for d, c in most_common]
        }

        # Print summary report
        print("\n📊 === Chatbot History Analysis Report ===")
        print(f"🗓️ Total chat sessions analyzed: {summary_data['total_sessions']}")
        print(f"💬 Average questions answered per session: {summary_data['avg_answers_per_session']:.2f}")
        print(f"👍 Total 'yes' answers recorded: {summary_data['total_yes_answers']}")
        print(f"👎 Total 'no' answers recorded: {summary_data['total_no_answers']}")
        # ✅ Added ratio printout
        if total_yes_no > 0:
             print(f"📈 Positive response ratio ('Yes' answers): {summary_data['yes_answer_ratio_percent']:.1f}%")

        if summary_data['top_diseases']:
            print("\n🏥 Top 5 most predicted diseases:")
            for i, item in enumerate(summary_data['top_diseases'], 1):
                print(f"   {i}. {item['disease']}: {item['count']} times")
        else:
             print("\n🏥 No disease predictions found in history.")

        print("\n✅ Report generated successfully!\n")

        return summary_data

    def export_summary(self, output_path="data/analysis_summary.json"):
        """Save the summary calculated by analyze() in a new JSON file."""
        summary = self.analyze()
        if summary is None:
            print("⚠️ No data to export (history empty or analysis failed).")
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        absolute_output_path = os.path.join(project_root, output_path)

        try:
            os.makedirs(os.path.dirname(absolute_output_path), exist_ok=True)
            with open(absolute_output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            print(f"💾 Summary exported successfully to {absolute_output_path}")
        except Exception as e:
            print(f"❌ Error exporting summary to {absolute_output_path}: {e}")

if __name__ == "__main__":
    analyzer = HistoryAnalyzer(history_path="history.json")
    analyzer.export_summary()