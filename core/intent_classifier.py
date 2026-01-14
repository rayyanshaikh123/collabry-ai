# core/intent_classifier.py
"""
Intent Classifier using Local Sentence Transformers

REVERTED: From Hugging Face API back to local lightweight models.

NEW ARCHITECTURE:
- Uses sentence-transformers for similarity-based intent classification
- Local processing only
- Lightweight models

SUPPORTED INTENTS:
- chat, qa, summarize, explain, analyze, plan, generate, search

This file maintains backward compatibility with the old classifier interface
while using local sentence transformers underneath.
"""

from pathlib import Path
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Import local sentence transformers classifier
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import sentence-transformers: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Fallback: Try optional import of joblib (legacy classical ML)
try:
    import joblib  # type: ignore
except Exception:
    joblib = None


class IntentClassifier:
    """
    Intent classifier with sentence-transformers backend.

    Maintains backward compatibility with old sklearn-based classifier
    while using local sentence-transformers for improved accuracy.
    """

    def __init__(self, model_path: str = "models/intent_classifier"):
        """
        Initialize intent classifier.

        Args:
            model_path: Ignored (kept for API compatibility)
        """
        self.mode = "sentence_transformers"  # Use sentence transformers by default
        self.clf = None
        self.vectorizer = None
        self.label_encoder = None

        # Intent examples for similarity-based classification
        self.intent_examples = {
            "chat": [
                "explain this concept",
                "help me understand",
                "what is the meaning of",
                "can you clarify",
                "teach me about",
                "how does this work",
                "what are the basics of",
                "break this down for me"
            ],
            "qa": [
                "what is",
                "how do",
                "why does",
                "when did",
                "where is",
                "who was",
                "which one",
                "tell me about"
            ],
            "summarize": [
                "summarize this",
                "give me a summary",
                "what's the main point",
                "key points",
                "overview of",
                "brief explanation"
            ],
            "explain": [
                "explain",
                "how does",
                "why does",
                "what happens when",
                "describe",
                "elaborate on"
            ],
            "analyze": [
                "analyze this",
                "break down",
                "examine",
                "evaluate",
                "assess",
                "critique"
            ],
            "plan": [
                "create a plan",
                "how to",
                "steps to",
                "guide me through",
                "help me plan",
                "what should I do"
            ],
            "generate": [
                "generate",
                "create",
                "make",
                "write",
                "produce",
                "build"
            ],
            "search": [
                "find",
                "search for",
                "look up",
                "research",
                "discover",
                "locate"
            ]
        }

        # Initialize sentence transformers model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
                # Pre-compute embeddings for intent examples
                self.intent_embeddings = {}
                for intent, examples in self.intent_examples.items():
                    self.intent_embeddings[intent] = self.model.encode(examples, convert_to_tensor=True)
                logger.info("✓ Using sentence-transformers-powered intent classifier")
            except Exception as e:
                logger.error(f"Failed to initialize sentence-transformers: {e}")
                self.model = None
                self.mode = "fallback"
        else:
            self.model = None
            self.mode = "fallback"

    def is_ready(self) -> bool:
        """Return True if intent classifier is available."""
        return self.model is not None or self.mode == "classical"

    def predict(self, text: str) -> str:
        """
        Return a single intent label string.
        
        Args:
            text: User input text
            
        Returns:
            Intent label (chat, qa, summarize, etc.)
        """
        # Try sentence transformers first
        if self.mode == "sentence_transformers" and self.model:
            try:
                query_embedding = self.model.encode(text, convert_to_tensor=True)

                best_intent = "chat"
                best_score = 0.0

                # Compare with each intent's examples
                for intent, examples_embedding in self.intent_embeddings.items():
                    # Compute cosine similarity
                    similarities = util.cos_sim(query_embedding, examples_embedding)
                    max_sim = float(similarities.max())

                    if max_sim > best_score:
                        best_score = max_sim
                        best_intent = intent

                # Only return intent if confidence is above threshold
                return best_intent if best_score > 0.5 else "chat"

            except Exception as e:
                logger.error(f"Sentence transformers intent prediction failed: {e}")
        
        # Fallback to classical if available
        if self.mode == "classical" and self.clf and self.vectorizer and self.label_encoder:
            try:
                X = self.vectorizer.transform([text])
                pred = self.clf.predict(X)[0]
                return self.label_encoder.inverse_transform([pred])[0]
            except Exception as e:
                logger.error(f"Classical intent prediction failed: {e}")
        
        # Last resort
        return "chat"

    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Return probability distribution as a dict.
        
        Args:
            text: User input text
            
        Returns:
            Dictionary of {intent: probability}
        """
        # Try sentence transformers first
        if self.mode == "sentence_transformers" and self.model:
            try:
                query_embedding = self.model.encode(text, convert_to_tensor=True)

                intent_scores = {}

                # Compare with each intent's examples
                for intent, examples_embedding in self.intent_embeddings.items():
                    # Compute cosine similarity
                    similarities = util.cos_sim(query_embedding, examples_embedding)
                    max_sim = float(similarities.max())
                    intent_scores[intent] = max_sim

                # Normalize scores to sum to 1 (simple approach)
                total = sum(intent_scores.values())
                if total > 0:
                    intent_scores = {intent: score/total for intent, score in intent_scores.items()}

                return intent_scores

            except Exception as e:
                logger.error(f"Sentence transformers predict_proba failed: {e}")
        
        # Fallback to classical if available
        if self.mode == "classical" and self.clf and self.vectorizer and self.label_encoder:
            try:
                X = self.vectorizer.transform([text])
                proba = self.clf.predict_proba(X)[0]
                classes = self.label_encoder.classes_
                return dict(zip(classes, proba))
            except Exception as e:
                logger.error(f"Classical predict_proba failed: {e}")
        
        # Last resort - equal probability
        return {"chat": 0.5, "qa": 0.5}

    def classify(self, text: str, context: Optional[str] = None) -> Dict[str, object]:
        """
        Backward-compatible classify() method expected by tests.

        Returns a dict with keys: intent, confidence, entities, tool_calls
        """
        # Prefer sentence transformers classifier if available
        try:
            if self.mode == "sentence_transformers" and self.model:
                intent = self.predict(text)
                probs = self.predict_proba(text)
                confidence = float(probs.get(intent, 0.0)) if isinstance(probs, dict) else 0.0
                return {
                    "intent": intent,
                    "confidence": confidence,
                    "entities": {},
                    "tool_calls": []
                }
        except Exception as e:
            logger.error(f"Sentence transformers classify() failed: {e}")

        # Fallback to predict/predict_proba
        intent = self.predict(text)
        probs = self.predict_proba(text)
        confidence = float(probs.get(intent, 0.0)) if isinstance(probs, dict) else 0.0
        return {"intent": intent, "confidence": confidence, "entities": {}, "tool_calls": []}
