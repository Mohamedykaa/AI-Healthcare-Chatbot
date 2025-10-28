import json
import os
from collections import Counter
from statistics import mean

class HistoryAnalyzer:
    """Analyzes chatbot conversation history and extracts useful insights."""

    def __init__(self, history_path="history.json"):
        # Adjust path calculation: Assume history_path is relative to the PROJECT ROOT
        # Get the directory of the current script (src/utils)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels to get the project root
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        # Combine project root with the relative history_path
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
            return None # Return None if no analysis possible

        total_sessions = len(self.history)
        all_diseases = []
        answer_counts_per_session = [] # Store count for each session
        yes_count = 0
        no_count = 0
        # Optional: Count other answers like 'maybe' if needed
        # maybe_count = 0

        for session in self.history:
            # Aggregate all diseases from final predictions
            for pred in session.get("predictions", []):
                disease_name = pred.get("disease")
                if disease_name:
                    all_diseases.append(disease_name)

            # ✅ --- START: CORRECTED ANSWER ANALYSIS ---
            answers = session.get("answers", {}) # Get the answers dict {qid: {data}}
            answer_counts_per_session.append(len(answers)) # Count questions answered in this session

            # Iterate through the answer data dictionaries
            for qid, answer_data in answers.items():
                if isinstance(answer_data, dict):
                    # Get the 'answer_text' safely
                    answer_text = answer_data.get("answer_text", "").lower()
                    if answer_text == "yes":
                        yes_count += 1
                    elif answer_text == "no":
                        no_count += 1
                    # elif answer_text == "maybe": # Optional
                    #     maybe_count += 1
                else:
                    # Fallback for potentially old format (if any exists)
                    # This part might not be needed if history is always new format
                    answer_text_old = str(answer_data).lower()
                    if answer_text_old == "yes":
                        yes_count += 1
                    elif answer_text_old == "no":
                        no_count += 1
            # ✅ --- END: CORRECTED ANSWER ANALYSIS ---


        # Calculate statistics
        most_common = Counter(all_diseases).most_common(5) if all_diseases else []
        avg_answers = mean(answer_counts_per_session) if answer_counts_per_session else 0

        # Prepare summary dictionary (useful for returning data)
        summary_data = {
            "total_sessions": total_sessions,
            "avg_answers_per_session": round(avg_answers, 2),
            "total_yes_answers": yes_count,
            "total_no_answers": no_count,
            "top_diseases": [{"disease": d, "count": c} for d, c in most_common]
        }

        # Print summary report
        print("\n📊 === Chatbot History Analysis Report ===")
        print(f"🗓️ Total chat sessions analyzed: {summary_data['total_sessions']}")
        print(f"💬 Average questions answered per session: {summary_data['avg_answers_per_session']:.2f}")
        print(f"👍 Total 'yes' answers recorded: {summary_data['total_yes_answers']}")
        print(f"👎 Total 'no' answers recorded: {summary_data['total_no_answers']}")

        if summary_data['top_diseases']:
            print("\n🏥 Top 5 most predicted diseases:")
            for i, item in enumerate(summary_data['top_diseases'], 1):
                print(f"   {i}. {item['disease']}: {item['count']} times")
        else:
             print("\n🏥 No disease predictions found in history.")

        print("\n✅ Report generated successfully!\n")

        return summary_data # Return the calculated data


    def export_summary(self, output_path="data/analysis_summary.json"):
        """Save the summary calculated by analyze() in a new JSON file."""
        # Call analyze() first to get the summary data
        summary = self.analyze()

        if summary is None: # analyze returns None if no history
            print("⚠️ No data to export (history empty or analysis failed).")
            return

        # Calculate absolute output path relative to the project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        absolute_output_path = os.path.join(project_root, output_path)

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(absolute_output_path), exist_ok=True)
            with open(absolute_output_path, "w", encoding="utf-8") as f:
                # Dump the summary dictionary returned by analyze()
                json.dump(summary, f, indent=4, ensure_ascii=False)
            print(f"💾 Summary exported successfully to {absolute_output_path}")
        except Exception as e:
            print(f"❌ Error exporting summary to {absolute_output_path}: {e}")

if __name__ == "__main__":
    # When running this script directly (python src/utils/history_analyzer.py),
    # history.json is expected to be in the project root directory.
    analyzer = HistoryAnalyzer(history_path="history.json")
    # analyze() is called within export_summary()
    analyzer.export_summary()