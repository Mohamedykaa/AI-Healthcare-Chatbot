"""
Robust training script for the disease-prediction NLP pipeline.

Key safety fixes:
- Ensures TextCleaner is part of pipeline (so training & inference use identical preprocessing).
- Prevents data leakage: split data first, then run GridSearchCV on the training set only.
- Saves model, encoder, training snapshot and a JSON training report.
- Default mode: full GridSearch (use --fast for quick training).
"""

import os
import json
import argparse
import joblib
import pandas as pd
import tempfile
from datetime import datetime
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# --- Paths ---
DATA_PATH = os.path.join("data", "merged_comprehensive_data.csv")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "optimized_nlp_pipeline.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "nlp_label_encoder.joblib")
REPORT_PATH = os.path.join(MODELS_DIR, "training_report.json")
TRAIN_SNAPSHOT_PATH = os.path.join(MODELS_DIR, "training_snapshot.csv")

# --- Import TextCleaner (project-specific), with fallback ---
try:
    # Ensure src is on the path if running from root (python src/train_model.py)
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    from src.utils.text_cleaner import TextCleaner
    print("✅ Imported TextCleaner from src.utils.text_cleaner")
except Exception as e:
    print(f"⚠️ Could not import src.utils.text_cleaner: {e}")
    # Minimal fallback that attempts to mimic expected cleaning (keeps English & Arabic letters + spaces)
    from sklearn.base import BaseEstimator, TransformerMixin
    import re
    class TextCleaner(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): return self
        def transform(self, X, y=None):
            pattern = re.compile(r"[^a-zA-Z0-9\u0621-\u064A\s,]")
            def clean_one(text):
                try:
                    s = str(text)
                    s = pattern.sub(" ", s)
                    s = re.sub(r"\s+", " ", s).strip().lower()
                    return s
                except Exception:
                    return ""
            if isinstance(X, pd.Series):
                return X.apply(clean_one)
            if isinstance(X, list):
                return pd.Series(X).apply(clean_one).tolist()
            # fallback convert
            if isinstance(X, str):
                return clean_one(X)
            return ""

    print("⚠️ Using fallback TextCleaner (basic). It's strongly recommended to use the project TextCleaner.")


def load_data():
    """Load merged dataset and verify required columns."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run 'python src/utils/data_merger.py' first.")
    df = pd.read_csv(DATA_PATH)
    if "training_text" not in df.columns or "disease" not in df.columns:
        raise ValueError("Dataset missing required columns ['training_text', 'disease'].")
    # Drop exact-empty rows
    df = df.dropna(subset=["training_text", "disease"])
    df = df[df["training_text"].astype(str).str.strip() != ""].copy()
    return df


def safe_joblib_dump(obj, final_path):
    """Save with a temp file and atomic replace to reduce risk of corrupted files."""
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    dirn = os.path.dirname(final_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dirn, suffix=".tmp")
    os.close(fd)
    joblib.dump(obj, tmp_path)
    os.replace(tmp_path, final_path)


def train_model(fast_mode=False, random_state=42):
    """Main training routine. Splits data first, then (optionally) GridSearch on training set."""
    print("🚀 Loading data...")
    df = load_data()
    print(f"ℹ️ Total records available: {len(df)}")

    X = df["training_text"].astype(str)
    y = df["disease"].astype(str)

    # Save a training snapshot for reproducibility/inspection
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        df.to_csv(TRAIN_SNAPSHOT_PATH, index=False)
        print(f"💾 Training snapshot saved to: {TRAIN_SNAPSHOT_PATH}")
    except Exception as e:
        print(f"⚠️ Could not save training snapshot: {e}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split first to avoid data leakage!
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.20, random_state=random_state, stratify=y_encoded
        )
    except ValueError:
        # Fallback for small datasets where stratify fails
        print("⚠️ Stratify failed (likely small dataset). Splitting without stratify.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.20, random_state=random_state
        )
        
    print(f"✅ Split into Train={len(X_train)} and Test={len(X_test)}")

    # Build pipeline including the TextCleaner
    pipeline = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)),
        ("rf", RandomForestClassifier(random_state=random_state, class_weight="balanced_subsample"))
    ])

    best_params = None
    cv_used = None

    if fast_mode:
        print("⚡ Running FAST mode: direct fit on training set (no GridSearch).")
        pipeline.set_params(rf__n_estimators=150, rf__max_depth=25)
        pipeline.fit(X_train, y_train)
    else:
        # Choose cv safely based on smallest class count in training set
        min_class_count = pd.Series(y_train).value_counts().min()
        cv = max(2, min(3, int(min_class_count)))  # between 2 and 3 folds, but not more than class count
        cv_used = cv
        print(f"🔍 Running GridSearchCV on training set only (cv={cv}) — this may take time...")

        param_grid = {
            "tfidf__max_features": [3000, 5000],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "rf__n_estimators": [150, 300],
            "rf__max_depth": [20, 30, None],
            "rf__min_samples_split": [2, 5]
        }

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1_weighted",  # robust to class imbalance
            cv=cv,
            n_jobs=-1,
            verbose=2
        )
        grid.fit(X_train, y_train)
        pipeline = grid.best_estimator_
        best_params = grid.best_params_
        print(f"✅ GridSearch finished. Best params: {best_params}")

    # Final evaluation on the held-out test set
    print("🔎 Evaluating on held-out test set...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cls_report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

    print(f"\n🎯 Test Accuracy: {acc:.4f}")
    print(f"🎯 Test F1-weighted: {f1:.4f}")
    print("\n--- Classification Report ---")
    print(cls_report)

    # Save model and encoder safely
    print("\n💾 Saving model and encoder...")
    try:
        safe_joblib_dump(pipeline, MODEL_PATH)
        safe_joblib_dump(le, ENCODER_PATH)
        print(f"✅ Model saved to: {MODEL_PATH}")
        print(f"✅ Encoder saved to: {ENCODER_PATH}")
    except Exception as e:
        print(f"❌ Error saving model or encoder: {e}")

    # Save training report with metadata
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fast_mode": bool(fast_mode),
        "n_records_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_accuracy": float(acc),
        "test_f1_weighted": float(f1),
        "best_params": best_params,
        "cv_used_on_training": cv_used,
        "model_path": MODEL_PATH,
        "encoder_path": ENCODER_PATH
    }
    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=4, ensure_ascii=False)
        print(f"🧾 Training report saved to: {REPORT_PATH}")
    except Exception as e:
        print(f"⚠️ Could not save training report: {e}")

    return pipeline, le, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NLP disease prediction model (safe: no data leakage).")
    parser.add_argument("--fast", action="store_true", help="Run quick training without GridSearch (fast mode).")
    args = parser.parse_args()

    print("🚀 Starting training (default = full GridSearch)...")
    train_model(fast_mode=args.fast)
    print("✅ Training complete.")