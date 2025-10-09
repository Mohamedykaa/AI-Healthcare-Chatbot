import pandas as pd
import joblib
import re
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. Custom Transformer for Text Cleaning ---
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [re.sub(r'[^a-zA-Z\s]', '', str(text)).lower().strip() for text in X]

# --- 2. Load Data ---
print("\nLoading the text-based dataset...")
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'DiseaseSymptomDescription.csv')
df = pd.read_csv(data_path)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
X = df['text']
y = df['label_encoded']

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Create the Full Machine Learning Pipeline ---
# Using 'rf' for RandomForestClassifier to be descriptive
pipeline = Pipeline([
    ('cleaner', TextCleaner()),
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('rf', RandomForestClassifier(random_state=42)) 
])

# --- 5. Define Parameter Grid ---
param_grid = {
    'tfidf__max_features': [1500, 2000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'rf__n_estimators': [100, 200],
}

# --- 6. Train with GridSearchCV ---
print("\nStarting GridSearchCV...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

# --- 7. Evaluate the Best Model ---
print(f"\nBest parameters found: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n--- Final Model Evaluation on Test Set ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- 8. Save the Final Pipeline ---
print("\nSaving the final pipeline...")
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(models_dir, exist_ok=True) 

joblib.dump(best_model, os.path.join(models_dir, "optimized_nlp_pipeline.joblib"))
joblib.dump(le, os.path.join(models_dir, "nlp_label_encoder.joblib"))
print("✅ Final pipeline and encoder saved successfully!")

