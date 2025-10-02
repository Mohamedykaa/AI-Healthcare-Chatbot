# train_model.py (Final Polished Version)

# --- Import Libraries ---
import pandas as pd
import joblib
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. Custom Transformer for Text Cleaning ---
class TextCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer to clean text by removing punctuation and converting to lowercase."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.apply(lambda text: re.sub(r"[^a-z\s]", "", str(text).lower()))

# --- 2. Load Data ---
print("\nLoading the text-based dataset...")
df = pd.read_csv("../data/DiseaseSymptomDescription.csv")
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
pipeline = Pipeline([
    ('cleaner', TextCleaner()),
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', RandomForestClassifier(random_state=42))
])

# --- 5. Define Parameter Grid ---
param_grid = {
    'tfidf__max_features': [1500, 2000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__n_estimators': [100, 200],
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
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# NEW: Top-3 Accuracy
y_pred_proba = best_model.predict_proba(X_test)
top3_accuracy = top_k_accuracy_score(y_test, y_pred_proba, k=3)
print(f"Top-3 Accuracy: {top3_accuracy:.4f}")

# NEW: Confusion Matrix
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# --- 8. Save the Final Pipeline ---
print("\nSaving the final pipeline...")
joblib.dump(best_model, "../optimized_nlp_pipeline.joblib")
joblib.dump(le, "../nlp_label_encoder.joblib")
print("Final pipeline and encoder saved successfully!")
