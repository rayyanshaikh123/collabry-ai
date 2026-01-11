"""
Retrain intent classifier and save artifacts compatible with the runtime loader.

This script trains a TF-IDF vectorizer + LogisticRegression classifier
from `data/dataset.jsonl` and saves the following files:

- models/intent_classifier/intentclf.clf.joblib
- models/intent_classifier/intentclf.le.joblib
- models/intent_classifier/intentclf.meta.json
- models/intent_classifier/intentclf/vectorizer.joblib
- models/intent_classifier/intentclf_pipeline.joblib  (optional convenience)

Run with: python training\retrain_intent_classifier.py
"""

import json
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


DATA_PATH = Path("data/dataset.jsonl")
OUT_DIR = Path("models") / "intent_classifier"
OUT_DIR.mkdir(parents=True, exist_ok=True)

samples = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except Exception:
            continue

texts = [s["text"] for s in samples if "text" in s]
labels = [s["label"] for s in samples if "label" in s]

if not texts or not labels or len(texts) != len(labels):
    raise SystemExit("Insufficient or invalid training data in data/dataset.jsonl")

# Label encoder
le = LabelEncoder()
y = le.fit_transform(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)

# Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Classifier
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train_vec, y_train)

# Evaluation
print("\nEvaluation:")
try:
    print(classification_report(y_test, clf.predict(X_test_vec), target_names=le.classes_, zero_division=0))
except Exception as e:
    # Fallback: print basic accuracy and a brief class info
    from sklearn.metrics import accuracy_score
    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"Classification report unavailable: {e}")
    print(f"Accuracy: {acc:.4f}")
    unique_labels = set(y_test)
    print(f"Classes in test set: {len(unique_labels)} (encoder has {len(le.classes_)})")

# Save files in the layout expected by runtime loader
joblib.dump(clf, OUT_DIR / "intentclf.clf.joblib")
joblib.dump(le, OUT_DIR / "intentclf.le.joblib")

meta = {"num_labels": len(le.classes_), "labels": list(le.classes_)}
json.dump(meta, open(OUT_DIR / "intentclf.meta.json", "w"), indent=2)

# Save vectorizer inside 'intentclf' subfolder as 'vectorizer.joblib'
vec_dir = OUT_DIR / "intentclf"
vec_dir.mkdir(exist_ok=True)
joblib.dump(vectorizer, vec_dir / "vectorizer.joblib")
# Also save a copy using legacy/runtime filename expected by loader
joblib.dump(vectorizer, OUT_DIR / "intentclf.vec.joblib")

# Save a convenience pipeline (vectorizer + clf) for future use
from sklearn.pipeline import Pipeline
pipeline = Pipeline([("vectorizer", vectorizer), ("clf", clf)])
joblib.dump(pipeline, OUT_DIR / "intentclf_pipeline.joblib")

print("\nSaved retrained artifacts to:")
print(" -", OUT_DIR / "intentclf.clf.joblib")
print(" -", OUT_DIR / "intentclf.le.joblib")
print(" -", OUT_DIR / "intentclf.meta.json")
print(" -", vec_dir / "vectorizer.joblib")
print(" -", OUT_DIR / "intentclf_pipeline.joblib")
