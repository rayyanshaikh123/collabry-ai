"""
Hugging Face–powered Intent Classifier

REPLACES: Old HuggingFace/Sentence-Transformers classifier

NEW ARCHITECTURE:
- Uses Hugging Face LLM for zero-shot intent classification
- Deterministic classification via low temperature
- Structured JSON outputs
- No model training/loading required
- Supports dynamic intent expansion

BENEFITS:
- No local model files
- Better accuracy with LLM reasoning
- Faster startup (no model loading)
- Easy to update prompts
"""

import logging
import json
from typing import Dict, List, Optional, Any

from config import CONFIG
from core.huggingface_service import create_hf_service

logger = logging.getLogger(__name__)


class HFIntentClassifier:
    """Intent classifier implemented using the Hugging Face LLM.

    This uses the configured HF model to perform zero-shot intent
    classification by prompting the model to return a JSON object
    describing the intent.
    """

    def __init__(self, hf_service=None):
        self.hf = hf_service or create_hf_service(model=CONFIG.get("llm_model"))
        self.intents = [
            "chat", "qa", "summarize", "explain", "analyze", "plan", "generate", "search"
        ]

    def _build_prompt(self, query: str, context: Optional[str] = None) -> str:
        prompt = (
            "You are an intent classifier. Given the user query, return a JSON object with the keys:"
            "\n- intent (one of: chat, qa, summarize, explain, analyze, plan, generate, search)"
            "\n- confidence (0.0-1.0)"
            "\n- entities (object)"
            "\n- tool_calls (array of tool names)"
            "\nOnly output valid JSON and no other text.\n\n"
        )
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += f"Query: {query}\n\nRespond with JSON."
        return prompt

    def classify(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        prompt = self._build_prompt(query, context)
        # Use low temperature for deterministic classification
        raw = self.hf.generate(prompt=prompt, temperature=0.0, max_tokens=200)

        # Try to parse JSON result
        try:
            parsed = json.loads(raw)
            # Ensure keys exist and sanitize
            intent = parsed.get("intent") if isinstance(parsed.get("intent"), str) else "chat"
            confidence = float(parsed.get("confidence", 0.5))
            entities = parsed.get("entities", {}) or {}
            tool_calls = parsed.get("tool_calls", []) or []
            return {"intent": intent, "confidence": confidence, "entities": entities, "tool_calls": tool_calls}
        except Exception:
            # If parsing fails, fall back to simple heuristic via model text
            text = str(raw).lower()
            for intent in self.intents:
                if intent in text:
                    return {"intent": intent, "confidence": 0.6, "entities": {}, "tool_calls": []}
            return {"intent": "chat", "confidence": 0.5, "entities": {}, "tool_calls": []}


def create_intent_classifier(hf_service) -> HFIntentClassifier:
    """
    Factory function to create an HF-backed intent classifier.

    Args:
        hf_service: Initialized Hugging Face service

    Returns:
        HFIntentClassifier instance
    """
    # Accept hf_service parameter for compatibility
    return HFIntentClassifier(hf_service=hf_service)


# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

class IntentClassifier:
    """
    Backward-compatible wrapper for old IntentClassifier API.

    Maintains the same interface but uses Hugging Face underneath.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize classifier (ignores model_path - not required).

        Args:
            model_path: Ignored (for API compatibility)
        """
        # Use Hugging Face service exclusively for intent classification
        hf = create_hf_service(model=CONFIG.get("llm_model"))
        self.classifier = HFIntentClassifier(hf_service=hf)
        logger.info("✓ Using HuggingFace-based IntentClassifier")
    
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

        # If the classifier returns our dataclass result, normalize it
        if isinstance(result, dict):
            return result

        return {
            "intent": getattr(result, "intent", "chat"),
            "confidence": getattr(result, "confidence", 0.5),
            "entities": getattr(result, "entities", {}),
            "tool_calls": getattr(result, "tool_calls", [])
        }


# Backwards compatibility alias
GeminiIntentClassifier = HFIntentClassifier
