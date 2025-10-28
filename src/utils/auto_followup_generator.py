# D:\disease_prediction_project\src\utils\auto_followup_generator.py
import pandas as pd
import json
import os
from pathlib import Path
import re # Import re for question cleaning

def generate_knowledge_base_final_wide():
    """
    Final, consolidated knowledge base generator. Handles WIDE format DiseaseAndSymptoms.csv.
    Combines:
     - Performance optimizations (dict lookups).
     - Accurate data standardization (symptoms & lowercase diseases).
     - Natural question generation.
     - Tiered severity boosting.
     - Melting logic for wide symptom file.
     - ✅ Added summary report at the end.
    """
    print("🚀 Starting Final Knowledge Base Generation (Handles WIDE format)...")

    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    output_file = data_dir / "english_knowledge_base.json"

    # --- 1. Load Datasets ---
    try:
        disease_symptoms_wide = pd.read_csv(data_dir / "DiseaseAndSymptoms.csv")
        severity_df = pd.read_csv(data_dir / "Symptom-severity.csv")
        descriptions_df = pd.read_csv(data_dir / "symptom_Description.csv")
        print("✅ Core datasets loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ Critical Error: Missing required file - {e}.")
        return
    except pd.errors.EmptyDataError as e:
        print(f"❌ Critical Error: File is empty - {e}.")
        return
    except Exception as e:
        print(f"❌ Critical Error: Could not load data - {e}.")
        return


    # --- 2. Clean and Standardize Column Names ---
    dfs_to_clean = [disease_symptoms_wide, severity_df, descriptions_df]
    for df in dfs_to_clean:
        if df is None: continue
        try:
            df.columns = [str(col).strip().lower().rstrip('_') for col in df.columns]
        except Exception as e:
            print(f"⚠️ Warning: Could not clean columns for a DataFrame: {e}")


    # --- 3. Standardize Symptom and Disease Values ---
    def standardize_text(s):
        try:
            return str(s).strip().lower().replace('_', ' ')
        except:
            return ""

    try:
        if 'disease' in disease_symptoms_wide.columns:
            disease_symptoms_wide["disease"] = disease_symptoms_wide["disease"].apply(standardize_text)
        else:
             print("❌ Critical Error: 'disease' column not found in DiseaseAndSymptoms.csv.")
             return

        if 'symptom' in severity_df.columns:
            severity_df["symptom"] = severity_df["symptom"].apply(standardize_text)
        else:
            print("❌ Critical Error: 'symptom' column not found in Symptom-severity.csv.")
            return

        if "disease" in descriptions_df.columns and "symptom" not in descriptions_df.columns:
            descriptions_df.rename(columns={"disease": "symptom"}, inplace=True)
        if 'symptom' in descriptions_df.columns:
            descriptions_df["symptom"] = descriptions_df["symptom"].apply(standardize_text)
        else:
             print("❌ Critical Error: 'symptom' column not found in symptom_Description.csv.")
             return

        severity_df.dropna(subset=['symptom'], inplace=True)
        descriptions_df.dropna(subset=['symptom'], inplace=True)
        severity_df = severity_df[severity_df['symptom'] != '']
        descriptions_df = descriptions_df[descriptions_df['symptom'] != '']


        print("✅ Symptom and disease values standardized (lowercase, spaces, trimmed).")
    except Exception as e:
        print(f"❌ Error during data standardization: {e}")
        return

    # --- 4. Create Lookup Maps for Performance ---
    try:
        # Ensure no duplicate symptoms exist before setting index
        severity_df_unique = severity_df.drop_duplicates(subset=['symptom'], keep='first')
        descriptions_df_unique = descriptions_df.drop_duplicates(subset=['symptom'], keep='first')

        severity_map = severity_df_unique.set_index('symptom')['weight'].to_dict()
        description_map = descriptions_df_unique.set_index('symptom')['description'].to_dict()
        print(f"✅ Created {len(severity_map)} severity lookups and {len(description_map)} description lookups.")
    except KeyError as e:
         print(f"❌ Error creating lookup maps. Missing column: {e}. Please check CSV headers ('weight' or 'description').")
         return
    except Exception as e:
        print(f"❌ Error creating lookup maps: {e}")
        return


    # --- 5. Melt Wide Symptom Data and Build Disease Dictionary ---
    disease_dict = {}
    try:
        id_vars = ['disease']
        symptom_cols = [col for col in disease_symptoms_wide.columns if re.match(r'^symptom_\d+$', col)]

        if not symptom_cols:
             print("❌ Error: No 'symptom_NUMBER' columns found in DiseaseAndSymptoms.csv after cleaning headers.")
             return

        print(f"ℹ️ Melting based on disease column '{id_vars[0]}' and symptom columns: {symptom_cols[:3]}...")

        melted = pd.melt(
            disease_symptoms_wide,
            id_vars=id_vars,
            value_vars=symptom_cols,
            var_name="symptom_source_column",
            value_name="symptom"
        )

        melted.dropna(subset=["symptom"], inplace=True)
        melted["symptom"] = melted["symptom"].apply(standardize_text)
        melted = melted[melted['symptom'] != '']

        for _, row in melted.iterrows():
            disease = row["disease"]
            symptom = row["symptom"]
            if not disease or not symptom: continue
            if disease not in disease_dict:
                disease_dict[disease] = set()
            disease_dict[disease].add(symptom)

        disease_dict = {d: sorted(list(s)) for d, s in disease_dict.items() if s}
        print(f"✅ Built disease dictionary from melted wide data ({len(disease_dict)} diseases).")

    except Exception as e:
        print(f"❌ Error during melting or building disease dictionary: {e}")
        import traceback
        traceback.print_exc()
        return


    # --- 6. Generate Knowledge Base Rules ---
    knowledge_base = {"rules": []}
    print("ℹ️ Generating rules and follow-up questions...")
    missing_severity_symptoms = set()
    missing_description_symptoms = set()

    for disease, symptoms in disease_dict.items():
        if not symptoms: continue

        rule = {
            "symptoms": symptoms,
            "conditions": [{"name": disease, "score": 0.65}],
            "follow_ups": []
        }

        for symptom in symptoms:
            severity_level = severity_map.get(symptom, 1)
            desc = description_map.get(symptom)

            if symptom not in severity_map: missing_severity_symptoms.add(symptom)
            if symptom not in description_map: missing_description_symptoms.add(symptom)

            if severity_level >= 6: boost = 0.35
            elif severity_level >= 3: boost = 0.2
            else: boost = 0.1

            if desc: question_text = f"Have you been experiencing {desc.lower()}?"
            else: question_text = f"Have you been suffering from {symptom} lately?"

            question_text = re.sub(r'\s+', ' ', question_text).strip().capitalize()
            symptom_id_part = re.sub(r'\W+', '_', symptom).strip('_')[:50]
            disease_id_part = re.sub(r'\W+', '_', disease).strip('_')[:50]

            rule["follow_ups"].append({
                "id": f"q_{disease_id_part}_{symptom_id_part}",
                "question": question_text,
                "boosts": [{"name": disease, "value": boost}],
                "severity": int(severity_level)
            })

        knowledge_base["rules"].append(rule)

    if missing_severity_symptoms: print(f"\n⚠️ Warning: Could not find severity for {len(missing_severity_symptoms)} symptoms. Examples: {list(missing_severity_symptoms)[:5]}")
    if missing_description_symptoms: print(f"⚠️ Warning: Could not find description for {len(missing_description_symptoms)} symptoms. Examples: {list(missing_description_symptoms)[:5]}")

    # --- 7. Save the New Knowledge Base ---
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(knowledge_base, f, indent=4, ensure_ascii=False)
        print(f"\n🎉 Final knowledge base with {len(knowledge_base['rules'])} rules generated successfully!")
        print(f"📁 Saved to: {output_file}")

        # --- ✅ 8. Print Summary Report ---
        total_diseases = len(disease_dict)
        all_mapped_symptoms = set()
        for symptoms_list in disease_dict.values():
            all_mapped_symptoms.update(symptoms_list)
        total_symptoms_mapped = len(all_mapped_symptoms)
        missing_sev_count = len(missing_severity_symptoms)
        missing_desc_count = len(missing_description_symptoms)

        print("\n" + "="*20 + " Report Summary " + "="*20)
        print(f"📊 Total unique diseases processed : {total_diseases}")
        print(f"🌿 Total unique symptoms mapped    : {total_symptoms_mapped}")
        print(f"❓ Symptoms missing severity data : {missing_sev_count}")
        print(f"❓ Symptoms missing description   : {missing_desc_count}")
        print("="*56 + "\n")

    except Exception as e:
         # ✅ Modified to catch errors during report generation as well
         print(f"\n❌ Error saving knowledge base to JSON or generating report: {e}")


if __name__ == "__main__":
    generate_knowledge_base_final_wide()