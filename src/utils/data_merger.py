# D:\disease_prediction_project\src\utils\data_merger.py
import pandas as pd
import joblib
import os
import re

# --- Configuration: Define all data file paths ---
DATA_DIR = "data"
MODELS_DIR = "models"

# Input files
FILES = {
    "base_symptoms": os.path.join(DATA_DIR, 'DiseaseSymptomDescription.csv'),
    "kaggle_symptoms": os.path.join(DATA_DIR, 'dataset.csv'), 
    "description": os.path.join(DATA_DIR, 'symptom_Description.csv'),
    "severity": os.path.join(DATA_DIR, 'Symptom-severity.csv'),
    "precaution": os.path.join(DATA_DIR, 'symptom_precaution.csv'),
}

# Output file
MERGED_DATA_PATH = os.path.join(DATA_DIR, "merged_comprehensive_data.csv") 

def clean_text(text):
    """A simple function to clean text data."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_kaggle_symptom_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the wide-format Kaggle symptom dataset (dataset.csv)
    into a long format with a single 'symptom_text' column.
    """
    df.columns = [re.sub(r'\s+', '', col).lower() for col in df.columns]
    
    try:
        disease_col = 'disease'
        symptom_cols = [c for c in df.columns if c.startswith('symptom_')]
        if not symptom_cols:
            raise ValueError("No symptom columns found after cleaning.")
    except (IndexError, ValueError) as e:
        print(f"❌ CRITICAL ERROR: Could not find 'disease' or 'symptom_' columns in 'dataset.csv': {e}.")
        return pd.DataFrame(columns=['disease', 'symptom_text'])

    df['symptom_text'] = df[symptom_cols].apply(
        lambda row: ', '.join(
            str(s).strip().replace('_', ' ') for s in row if pd.notna(s) and str(s).strip()
        ),
        axis=1
    )

    print(f"✅ Processed Kaggle dataset successfully.")
    return df[[disease_col, 'symptom_text']].rename(columns={disease_col: 'disease'})

def load_and_merge_all_data():
    """
    Loads all datasets, standardizes them, merges them into a single comprehensive
    dataframe, and creates a combined text feature for training.
    """
    print("🚀 Starting Comprehensive Data Merge Process...")
    
    try:
        df_base = pd.read_csv(FILES["base_symptoms"])
        df_kaggle = pd.read_csv(FILES["kaggle_symptoms"])
        df_desc = pd.read_csv(FILES["description"])
        df_sev = pd.read_csv(FILES["severity"])
        df_prec = pd.read_csv(FILES["precaution"])
        print("✅ All source datasets loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: File not found - {e}. Please ensure all datasets are in the '{DATA_DIR}' folder.")
        return None

    # --- Process and Standardize Each DataFrame Individually ---
    df_base.columns = [col.strip().lower() for col in df_base.columns]
    df_base.rename(columns={'label': 'disease', 'text': 'symptom_text'}, inplace=True)
    df_kaggle_processed = process_kaggle_symptom_df(df_kaggle)
    df_desc.columns = [col.strip().lower() for col in df_desc.columns]
    df_sev.columns = [col.strip().lower() for col in df_sev.columns]
    df_sev.rename(columns={'symptom': 'symptom', 'weight': 'severity_weight'}, inplace=True)
    df_prec.columns = [col.strip().lower() for col in df_prec.columns]

    # --- Combine and Merge ---
    all_symptoms = pd.concat([df_base[['disease', 'symptom_text']], df_kaggle_processed], ignore_index=True)
    
    for df in [all_symptoms, df_desc, df_prec]: 
        if 'disease' in df.columns:
            df['disease'] = df['disease'].str.strip()
    all_symptoms['symptom_text'] = all_symptoms['symptom_text'].apply(clean_text)

    # Merge description and precautions
    merged = all_symptoms.merge(df_desc, on='disease', how='left')
    merged = merged.merge(df_prec, on='disease', how='left')

    merged.drop_duplicates(subset=['disease', 'symptom_text'], inplace=True)
    merged.fillna('', inplace=True)
    
    # --- Feature Engineering: Create a rich text feature for the model ---
    merged['training_text'] = (
        merged['symptom_text'] + ' ' + 
        merged.get('description', '') + ' ' +
        merged.get('precaution_1', '') + ' ' +
        merged.get('precaution_2', '') + ' ' +
        merged.get('precaution_3', '') + ' ' +
        merged.get('precaution_4', '')
    )
    merged['training_text'] = merged['training_text'].apply(clean_text)

    merged = merged[merged['training_text'] != ''].copy()

    print(f"✅ Data merging and processing complete. Total unique records: {len(merged)}")
    merged.to_csv(MERGED_DATA_PATH, index=False)
    
    return merged

# --- ✅ MODIFIED ---
# Removed the old 'train_and_save_model' function.
# This file's ONLY job is to create the merged CSV.
# Training is handled by 'src/train_model.py'.
# ---

if __name__ == "__main__":
    final_dataframe = load_and_merge_all_data()
    
    if final_dataframe is not None:
        print(f"\n🎉 Merged data saved successfully to '{MERGED_DATA_PATH}'")
        print("ℹ️ Next step: Run 'python src/train_model.py' to train the model on this new data.")
    else:
        print("\n❌ Data merging failed. Please check errors above.")