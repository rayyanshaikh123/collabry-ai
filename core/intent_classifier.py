# core/intent_classifier.py
"""
GEMINI-POWERED INTENT CLASSIFIER

MIGRATION: Replaced HuggingFace/sklearn classifier with Google Gemini (2024)

NEW ARCHITECTURE:
- Uses Gemini for zero-shot intent classification
- No model training/loading required
- Better accuracy with GPT-class reasoning
- Backward-compatible API maintained

SUPPORTED INTENTS:
- chat, qa, summarize, explain, analyze, plan, generate, search

This file maintains backward compatibility with the old classifier interface
while using Gemini underneath for superior performance.
"""

from pathlib import Path
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Import Gemini-powered intent classifier
try:
    from core.gemini_intent import GeminiIntentClassifier, create_intent_classifier
    from core.gemini_service import create_gemini_service
    GEMINI_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import Gemini classifier: {e}")
    GEMINI_AVAILABLE = False

# Fallback: Try optional import of joblib (legacy classical ML)
try:
    import joblib  # type: ignore
except Exception:
    joblib = None


class IntentClassifier:
    """
    Intent classifier with Gemini backend.
    
    Maintains backward compatibility with old sklearn-based classifier
    while using Gemini for superior accuracy.
    """
    
    def __init__(self, model_path: str = "models/intent_classifier"):
        """
        Initialize intent classifier.
        
        Args:
            model_path: Ignored (kept for API compatibility)
        """
        self.mode = "gemini"  # Use Gemini by default
        self.clf = None
        self.vectorizer = None
        self.label_encoder = None
        
        # Initialize Gemini classifier
        if GEMINI_AVAILABLE:
            try:
                gemini_service = create_gemini_service()
                self.gemini_classifier = GeminiIntentClassifier(gemini_service)
                logger.info("✓ Using Gemini-powered intent classifier")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini classifier: {e}")
                self.gemini_classifier = None
                self.mode = "fallback"
        else:
            self.gemini_classifier = None
            self.mode = "fallback"

    def is_ready(self) -> bool:
        """Return True if intent classifier is available."""
        return self.gemini_classifier is not None or self.mode == "classical"

    def predict(self, text: str) -> str:
        """
        Return a single intent label string.
        
        Args:
            text: User input text
            
        Returns:
            Intent label (chat, qa, summarize, etc.)
        """
        # Try Gemini first
        if self.mode == "gemini" and self.gemini_classifier:
            try:
                result = self.gemini_classifier.classify(text)
                return result.intent
            except Exception as e:
                logger.error(f"Gemini intent prediction failed: {e}")
        
        # Fallback to classical if available
        if self.mode == "classical" and self.clf and self.vectorizer and self.label_encoder:
            try:
                X = self.vectorizer.transform([text])
                pred = self.clf.predict(X)[0]
                return self.label_encoder.inverse_transform([pred])[0]
            except Exception as e:
                logger.error(f"Classical intent prediction failed: {e}")
        
        # Last resort
        return "unknown"

    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Return probability distribution as a dict.
        
        Args:
            text: User input text
            
        Returns:
            Dictionary of {intent: probability}
        """
        # Try Gemini first
        if self.mode == "gemini" and self.gemini_classifier:
            try:
                result = self.gemini_classifier.classify(text)
                return {result.intent: result.confidence}
            except Exception as e:
                logger.error(f"Gemini probability prediction failed: {e}")
        
        # Fallback to classical if available
        if self.mode == "classical" and self.clf and self.vectorizer and self.label_encoder:
            try:
                X = self.vectorizer.transform([text])
                probs = self.clf.predict_proba(X)[0]
                labels = self.label_encoder.classes_
                return {str(labels[i]): float(probs[i]) for i in range(len(labels))}
            except Exception as e:
                logger.error(f"Classical probability prediction failed: {e}")
        
        # Last resort
        return {"unknown": 1.0}
