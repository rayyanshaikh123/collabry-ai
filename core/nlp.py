# core/nlp.py
"""
COLLABRY NLP PIPELINE - LOCAL LIGHTWEIGHT MODELS

ARCHITECTURE:
- Sentence-transformers for intent classification (similarity-based)
- Transformers pipeline for NER (small local model)
- Local processing only
- Lightweight models

BENEFITS:
- No cloud dependencies for NLP
- Fast local processing
- Lightweight models
"""

import logging
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer, util
import spacy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Intent examples for similarity-based classification
INTENT_EXAMPLES = {
    "study": [
        "explain this concept",
        "help me understand",
        "what is the meaning of",
        "can you clarify",
        "teach me about",
        "how does this work",
        "what are the basics of",
        "break this down for me"
    ],
    "quiz": [
        "create a quiz",
        "test my knowledge",
        "give me questions",
        "make a test",
        "quiz me on",
        "practice questions"
    ],
    "course_finder": [
        "find courses about",
        "recommend courses",
        "best online courses",
        "learn about",
        "courses for"
    ],
    "mindmap": [
        "create a mind map",
        "visualize this",
        "mind map for",
        "diagram of"
    ],
    "ppt": [
        "create a presentation",
        "powerpoint for",
        "slides about",
        "presentation on"
    ],
    "general": [
        "hello",
        "hi",
        "how are you",
        "what can you do"
    ]
}


class LocalNLP:
    """Local lightweight NLP processing."""

    def __init__(self):
        self.intent_model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
        self.nlp = spacy.load("en_core_web_sm")  # Small NER model
        
        # Pre-compute embeddings for intent examples
        self.intent_embeddings = {}
        for intent, examples in INTENT_EXAMPLES.items():
            self.intent_embeddings[intent] = self.intent_model.encode(examples, convert_to_tensor=True)
        
        logger.info("✓ Local NLP service initialized with sentence-transformers and spaCy NER")

    def classify_intent(self, text: str) -> Optional[str]:
        """Classify intent using sentence-transformers similarity."""
        try:
            text_embedding = self.intent_model.encode(text, convert_to_tensor=True)
            
            best_intent = None
            best_score = 0.0
            
            for intent, examples_embeddings in self.intent_embeddings.items():
                # Compute cosine similarities
                similarities = util.cos_sim(text_embedding, examples_embeddings)[0]
                max_sim = similarities.max().item()
                
                if max_sim > best_score:
                    best_score = max_sim
                    best_intent = intent
            
            # Only return intent if confidence is above threshold
            return best_intent if best_score > 0.5 else None
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return None

    def extract_entities(self, text: str) -> List[tuple]:
        """Extract entities using spaCy."""
        try:
            doc = self.nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []


# Global instance
nlp_service = None

def init_nlp():
    """Initialize local NLP service."""
    global nlp_service
    try:
        nlp_service = LocalNLP()
    except Exception as e:
        logger.error(f"Failed to initialize local NLP: {e}")
        nlp_service = None

# Initialize on import
init_nlp()


# ---------------------------------------------------------------
# MAIN ANALYSIS FUNCTION (Local models)
# ---------------------------------------------------------------
def analyze(text: str) -> Dict[str, Any]:
    """
    Unified NLP analysis using local lightweight models.

    Performs:
    1. Intent classification using sentence-transformers similarity
    2. Named Entity Recognition using spaCy

    Args:
        text: Input text to analyze

    Returns:
        Dictionary with analysis results
    """
    # Skip NLP for very long texts (> 1MB)
    if len(text) > 1000000:
        logger.info(f"Skipping NLP for long text ({len(text)} chars)")
        return {
            "corrected": text,
            "intent": None,
            "entities": []
        }

    try:
        # Spell correction (placeholder - same as input for now)
        corrected = text

        # Intent classification
        intent = nlp_service.classify_intent(corrected) if nlp_service else None

        # Entity extraction
        entities = nlp_service.extract_entities(corrected) if nlp_service else []

        return {
            "corrected": corrected,
            "intent": intent,
            "entities": entities
        }

    except Exception as e:
        logger.error(f"NLP analysis failed: {e}")
        return {
            "corrected": text,
            "intent": None,
            "entities": []
        }

