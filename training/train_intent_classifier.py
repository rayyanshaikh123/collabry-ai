"""
TRAIN INTENT CLASSIFIER (Scikit-Learn Version)
Compatible with COLLABRY NLP pipeline
Produces:
    intentclf.tfidf.joblib
    intentclf.clf.joblib
    intentclf.le.joblib
    intentclf.meta.json
"""

import json
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# --------------------------
# Load Dataset
# --------------------------

DATA_PATH = Path("data/dataset.jsonl")
samples = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            samples.append(json.loads(line))
        except:
            pass

texts = [s["text"] for s in samples]
labels = [s["label"] for s in samples]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42
)

# --------------------------
# Vectorizer
# --------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    lowercase=True,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --------------------------
# Classifier
# --------------------------

clf = LogisticRegression(max_iter=2000)
clf.fit(X_train_vec, y_train)

# --------------------------
# Evaluation
# --------------------------

print("\nEvaluation:")
print(classification_report(
    y_test,
    clf.predict(X_test_vec),
    labels=list(range(len(le.classes_))),
    target_names=le.classes_,
    zero_division=0
))


# --------------------------
# Save Model Files
# --------------------------

OUTPUT_DIR = Path("models/intent_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(vectorizer, OUTPUT_DIR / "intentclf.tfidf.joblib")
joblib.dump(clf, OUTPUT_DIR / "intentclf.clf.joblib")
joblib.dump(le, OUTPUT_DIR / "intentclf.le.joblib")

meta = {
    "num_labels": len(le.classes_),
    "labels": list(le.classes_),
}
json.dump(meta, open(OUTPUT_DIR / "intentclf.meta.json", "w"), indent=2)

print("\nSaved model to:", OUTPUT_DIR)
