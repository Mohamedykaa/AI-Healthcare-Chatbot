import pandas as pd
import os

def analyze_data():
    train_path = r'd:\disease_prediction_project\data\Training.csv'
    test_path = r'd:\disease_prediction_project\data\Testing.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Files not found.")
        return

    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    print("--- Training Data Analysis ---")
    print(f"Total Rows: {len(df_train)}")
    print(f"Total Columns: {len(df_train.columns)}")
    
    # Check for duplicates
    # We exclude the 'prognosis' column for input feature duplicates, but here we care about exact row duplicates including label
    duplicates = df_train.duplicated().sum()
    print(f"Duplicate Rows (Exact match including label): {duplicates}")
    print(f"Unique Rows: {len(df_train) - duplicates}")
    
    # Class distribution
    class_counts = df_train['prognosis'].value_counts()
    print(f"\nClass Distribution (Top 5):")
    print(class_counts.head())
    print(f"Class Distribution (Bottom 5):")
    print(class_counts.tail())
    print(f"Number of Classes: {len(class_counts)}")
    
    print("\n--- Testing Data Analysis ---")
    print(f"Total Rows: {len(df_test)}")
    
    # Check for leakage
    common_cols = list(set(df_train.columns) & set(df_test.columns))
    
    # Check if test rows exist in training
    leakage_count = 0
    train_set = set(tuple(x) for x in df_train[common_cols].values)
    
    for row in df_test[common_cols].values:
        if tuple(row) in train_set:
            leakage_count += 1
            
    print(f"\nData Leakage Check:")
    print(f"Test rows found exactly in Training data: {leakage_count} out of {len(df_test)}")

if __name__ == "__main__":
    analyze_data()
