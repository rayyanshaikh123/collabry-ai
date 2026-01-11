"""
Gemini-powered Intent Classifier

REPLACES: Old HuggingFace/Sentence-Transformers classifier

NEW ARCHITECTURE:
- Uses Gemini for zero-shot intent classification
- Deterministic classification via low temperature
- Structured JSON outputs
- No model training/loading required
- Supports dynamic intent expansion

BENEFITS:
- No local model files
- Better accuracy (GPT-class reasoning)
- Faster startup (no model loading)
- Easy to update prompts
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Intent classification result"""
    intent: str
    confidence: float
    entities: Dict[str, Any]
    tool_calls: List[str]
    reasoning: str


class GeminiIntentClassifier:
    """
    Intent classifier powered by Google Gemini.
    
    Replaces the old HuggingFace-based classifier with Gemini's
    zero-shot classification capabilities.
    """
    
    def __init__(self, gemini_service):
        """
        Initialize classifier with Gemini service.
        
        Args:
            gemini_service: Initialized GeminiService instance
        """
        self.gemini = gemini_service
        
        # Define supported intents (can be dynamically expanded)
        self.intents = {
            "chat": "Casual conversation, greetings, general questions",
            "qa": "Specific questions requiring factual answers",
            "summarize": "Requests to summarize documents or topics",
            "explain": "Requests for explanations or clarifications",
            "analyze": "Requests for analysis or insights",
            "plan": "Study planning, scheduling, goal setting",
            "generate": "Content generation (quiz, mindmap, flashcards, notes)",
            "search": "Web search or external information lookup"
        }
        
        # Tool mapping for each intent
        self.intent_tools = {
            "search": ["web_search"],
            "generate": ["doc_generator", "mindmap_generator", "ppt_generator"],
            "plan": ["write_file"],
            "qa": ["read_file"],
            "summarize": ["read_file"]
        }
        
        logger.info("✓ GeminiIntentClassifier initialized")
    
    def classify(
        self,
        query: str,
        context: Optional[str] = None
    ) -> IntentResult:
        """
        Classify user intent using Gemini.
        
        Args:
            query: User query
            context: Optional conversation context
            
        Returns:
            IntentResult with classification and metadata
        """
        return self.gemini.classify_intent(query, context)
    
    def classify_batch(
        self,
        queries: List[str]
    ) -> List[IntentResult]:
        """
        Classify multiple queries.
        
        Args:
            queries: List of user queries
            
        Returns:
            List of IntentResult objects
        """
        return [self.classify(q) for q in queries]
    
    def add_intent(
        self,
        intent_name: str,
        description: str,
        tools: Optional[List[str]] = None
    ):
        """
        Dynamically add a new intent.
        
        Args:
            intent_name: Name of the new intent
            description: Description for classification
            tools: Tools to invoke for this intent
        """
        self.intents[intent_name] = description
        if tools:
            self.intent_tools[intent_name] = tools
        
        logger.info(f"Added new intent: {intent_name}")


def create_intent_classifier(gemini_service) -> GeminiIntentClassifier:
    """
    Factory function to create Gemini-powered intent classifier.
    
    Args:
        gemini_service: Initialized GeminiService
        
    Returns:
        GeminiIntentClassifier instance
    """
    return GeminiIntentClassifier(gemini_service)


# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

class IntentClassifier:
    """
    Backward-compatible wrapper for old IntentClassifier API.
    
    Maintains the same interface but uses Gemini underneath.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize classifier (ignores model_path - Gemini doesn't need it).
        
        Args:
            model_path: Ignored (for API compatibility)
        """
        # Import here to avoid circular dependency
        from core.gemini_service import create_gemini_service
        
        try:
            gemini_service = create_gemini_service()
            self.classifier = GeminiIntentClassifier(gemini_service)
            logger.info("✓ Using Gemini-powered IntentClassifier")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini classifier: {e}")
            raise
    
    def classify(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify intent (backward-compatible API).
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            Dictionary with intent classification
        """
        result = self.classifier.classify(query, context)
        
        # Convert to old format
        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "entities": result.entities,
            "tool_calls": result.tool_calls
        }
