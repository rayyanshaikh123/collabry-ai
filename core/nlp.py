# core/nlp.py
"""
COLLABRY NLP PIPELINE - HUGGING FACE CLOUD MODELS

MIGRATION: Replaced spaCy and Gemini with Hugging Face cloud models (2024)

NEW ARCHITECTURE:
- Hugging Face NER model for entity extraction
- Hugging Face intent classification model
- Cloud-based processing via Hugging Face Inference API
- No local model dependencies

BENEFITS:
- No local model loading (faster startup)
- Cloud-based processing (scalable)
- Access to state-of-the-art models
- Unified API interface
- No dependency on spaCy or Gemini

BACKWARD COMPATIBILITY:
- analyze() function maintains same interface
- Returns same dictionary structure
- All existing code using analyze() continues to work
"""

import logging
import requests
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------
# Hugging Face API Configuration
# ---------------------------------------------------------------
HF_API_BASE = "https://router.huggingface.co"

class HuggingFaceNLP:
    """Hugging Face cloud NLP processing."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.ner_model = "dbmdz/bert-large-cased-finetuned-conll03-english"
        self.intent_model = "facebook/bart-large-mnli"

    def _call_hf_api(self, model: str, inputs: Any, options: Dict = None) -> Optional[Any]:
        """Call Hugging Face Inference API."""
        try:
            url = f"https://router.huggingface.co/hf-inference/models/{model}"
            payload = {"inputs": inputs}
            if options:
                # For classification models, parameters go under "parameters" key
                if "classification" in model or "mnli" in model:
                    payload["parameters"] = options
                else:
                    payload.update(options)

            response = requests.post(url, headers=self.headers, json=payload, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Hugging Face API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Hugging Face API call failed: {e}")
            return None

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using Hugging Face NER model."""
        try:
            result = self._call_hf_api(self.ner_model, text)

            if not result:
                return []

            entities = []
            current_entity = None

            for token in result:
                entity_tag = token.get("entity", "")
                word = token.get("word", "").replace("##", "")

                if entity_tag.startswith("B-"):
                    # Start of new entity
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "text": word,
                        "label": entity_tag[2:],  # Remove B- prefix
                        "start": token.get("start"),
                        "end": token.get("end")
                    }
                elif entity_tag.startswith("I-") and current_entity:
                    # Continue current entity
                    current_entity["text"] += word
                    current_entity["end"] = token.get("end")
                elif current_entity:
                    # End current entity
                    entities.append(current_entity)
                    current_entity = None

            # Add final entity if exists
            if current_entity:
                entities.append(current_entity)

            # Convert to expected format
            formatted_entities = []
            for entity in entities:
                formatted_entities.append({
                    "text": entity["text"],
                    "label": entity["label"],
                    "confidence": 0.9  # Default confidence for HF models
                })

            return formatted_entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    def classify_intent(self, text: str) -> str:
        """Classify intent using zero-shot classification."""
        try:
            # Use predefined intent categories
            candidate_labels = [
                "study help", "question answering", "document analysis",
                "note taking", "quiz preparation", "research", "general chat"
            ]

            result = self._call_hf_api(
                self.intent_model,
                text,
                {"candidate_labels": candidate_labels, "multi_label": False}
            )

            if result and "labels" in result and "scores" in result:
                # Return the highest scoring intent
                return result["labels"][0]

            return "general chat"  # Default fallback

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return "general chat"

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment (placeholder - could use another HF model)."""
        # For now, return neutral sentiment
        return {"score": 0.0, "magnitude": 0.5}


# Global instance
_nlp_service = None

def _get_nlp_service():
    """Lazy initialization of Hugging Face NLP service."""
    global _nlp_service
    if _nlp_service is None:
        # Get API key from config
        try:
            from config import CONFIG
            api_key = CONFIG.get("huggingface_api_key")
        except:
            api_key = None

        _nlp_service = HuggingFaceNLP(api_key=api_key)
        logger.info("✓ Hugging Face NLP service initialized")
    return _nlp_service


# ---------------------------------------------------------------
# MAIN ANALYSIS FUNCTION (Hugging Face-powered)
# ---------------------------------------------------------------
def analyze(text: str) -> Dict[str, Any]:
    """
    Unified NLP analysis powered by Hugging Face cloud models.

    Performs:
    1. Intent classification using zero-shot classification
    2. Named Entity Recognition (NER) using BERT model
    3. Basic sentiment analysis

    Args:
        text: Input text to analyze

    Returns:
        Dictionary with analysis results:
        {
            "text": original text,
            "corrected": spell-corrected text (same as input for now),
            "intent": detected intent,
            "intent_proba": confidence scores,
            "entities": list of entity dictionaries,
            "sentiment": sentiment analysis results
        }
    """
    # Skip NLP for very long texts (> 1MB)
    if len(text) > 1000000:
        logger.info(f"Skipping NLP for long text ({len(text)} chars)")
        return {
            "text": text,
            "corrected": text,
            "intent": "unknown",
            "intent_proba": {},
            "entities": [],
            "sentiment": {"score": 0.0, "magnitude": 0.0}
        }

    try:
        nlp_service = _get_nlp_service()

        # Spell correction (placeholder - same as input for now)
        corrected = text

        # Intent classification
        intent = nlp_service.classify_intent(corrected)

        # Entity extraction
        entities = nlp_service.extract_entities(corrected)

        # Sentiment analysis
        sentiment = nlp_service.analyze_sentiment(corrected)

        return {
            "text": text,
            "corrected": corrected,
            "intent": intent,
            "intent_proba": {"confidence": 0.8},  # Placeholder confidence
            "entities": entities,
            "sentiment": sentiment,
            "language": "en"
        }

    except Exception as e:
        logger.error(f"Hugging Face NLP analysis failed: {e}")
        # Return minimal response on failure
        return {
            "text": text,
            "corrected": text,
            "intent": "general chat",
            "intent_proba": {},
            "entities": [],
            "sentiment": {"score": 0.0, "magnitude": 0.0}
        }


def _analyze_with_gemini(text: str) -> Dict[str, Any]:
    """
    Perform NLP analysis using Gemini.
    
    Args:
        text: Input text
        
    Returns:
        Analysis dictionary
    """
    gemini = _get_gemini_service()
    intent_clf = _get_intent_classifier()
    
    # -------------------------
    # 1) SPELL CORRECTION (optional)
    # -------------------------
    # For now, skip spell correction to reduce API calls
    # Can be enabled later if needed
    corrected = text
    
    # -------------------------
    # 2) INTENT CLASSIFICATION
    # -------------------------
    try:
        intent_result = intent_clf.classify(corrected)
        
        # Handle both dict and object responses
        if isinstance(intent_result, dict):
            intent = intent_result.get("intent", "unknown")
            confidence = intent_result.get("confidence", 0.8)
        else:
            # IntentResult object
            intent = intent_result.intent
            confidence = intent_result.confidence
        
        # Convert confidence to probability format
        proba = {intent: confidence}
        
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        intent = "unknown"
        proba = {}
    
    # -------------------------
    # 3) NAMED ENTITY RECOGNITION
    # -------------------------
    try:
        entities_dict = gemini.extract_entities(corrected)
        
        # Convert dict format {"LABEL": [entities]} to list of tuples [(text, label)]
        entities = []
        for label, texts in entities_dict.items():
            for text in texts:
                entities.append((text, label))
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        entities = []
    
# ---------------------------------------------------------------
# ADDITIONAL HELPER FUNCTIONS
# ---------------------------------------------------------------
def extract_entities(text: str, entity_types: Optional[List[str]] = None) -> List[tuple]:
    """
    Extract named entities from text.

    Args:
        text: Input text
        entity_types: Specific entity types to extract (optional)

    Returns:
        List of (entity_text, entity_type) tuples
    """
    try:
        nlp_service = _get_nlp_service()
        entities = nlp_service.extract_entities(text)

        # Filter by entity types if specified
        if entity_types:
            entities = [e for e in entities if e["label"] in entity_types]

        # Convert to tuple format for backward compatibility
        return [(e["text"], e["label"]) for e in entities]
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return []


def classify_intent(text: str) -> str:
    """
    Classify the intent of the input text.

    Args:
        text: Input text

    Returns:
        Intent category string
    """
    try:
        nlp_service = _get_nlp_service()
        return nlp_service.classify_intent(text)
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return "general chat"
    
    # Filter by entity types if specified
    if entity_types:
        entities = [(text, label) for text, label in entities if label in entity_types]
    
    return entities


def classify_intent(text: str, context: Optional[str] = None) -> str:
    """
    Classify user intent.
    
    Args:
        text: User input
        context: Optional conversation context
        
    Returns:
        Detected intent string
    """
    if GEMINI_AVAILABLE:
        try:
            intent_clf = _get_intent_classifier()
            result = intent_clf.classify(text, context)
            return result.intent
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
    
    return "unknown"
