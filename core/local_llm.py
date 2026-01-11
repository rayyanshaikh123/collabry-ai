# core/local_llm.py
"""
COLLABRY AI ENGINE - GEMINI-POWERED LLM

MIGRATION: Replaced Ollama with Google Gemini (2024)

OLD ARCHITECTURE:
- LocalLLM → Ollama API (localhost:11434)
- Llama 3.1 model
- Requires local model running

NEW ARCHITECTURE:
- LocalLLM → Google Gemini API
- Gemini 1.5 Flash/Pro model
- Cloud-based, no local dependencies
- Faster, more reliable, cost-free hosting

BACKWARD COMPATIBILITY:
- LocalLLM class maintains same interface
- create_llm() factory function preserved
- All existing code using LocalLLM continues to work
- LangChain compatibility maintained

BENEFITS:
- No local model setup required
- Better reasoning and context understanding
- Faster response times
- Structured JSON outputs
- Free tier hosting
"""

import json
import logging
from typing import Any, List, Mapping, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import Gemini service
from core.gemini_service import GeminiService, create_gemini_service
# Import Ollama service as fallback
from core.ollama_service import OllamaService, create_ollama_service

# LangChain compatibility
from langchain_core.language_models.llms import LLM


class LocalLLM(LLM):
    """
    Gemini-powered LLM wrapper with Ollama fallback.
    
    REPLACES: Old Ollama-based LocalLLM
    
    Maintains the same interface for backward compatibility,
    uses Google Gemini with Ollama fallback when Gemini is unavailable.
    
    Features:
    - Primary: Google Gemini API
    - Fallback: Local Ollama API
    - Automatic retry with exponential backoff
    - Streaming support
    - JSON mode for structured outputs
    - LangChain compatibility
    """
    gemini_service: Any  # GeminiService instance
    ollama_service: Any  # OllamaService instance (fallback)
    model_name: str
    temperature: float
    timeout: int = 120
    max_retries: int = 3
    last_response: Optional[str] = None

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Generate response from Gemini with Ollama fallback.
        
        Args:
            prompt: Input prompt
            stop: Stop sequences (not used in Gemini)
            **kwargs: Additional arguments
            
        Returns:
            Generated text response
        """
        # Try Gemini first
        try:
            response = self.gemini_service.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=kwargs.get('max_tokens')
            )
            
            # gemini_service.generate() returns a string, not a response object
            self.last_response = response
            logger.info("[Gemini] Generation successful")
            return self.last_response
            
        except Exception as e:
            logger.warning(f"[Gemini] Generation failed: {e}, trying Ollama fallback")
            
            # Fallback to Ollama
            try:
                response = self.ollama_service.generate(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=kwargs.get('max_tokens')
                )
                self.last_response = response
                logger.info("[Ollama] Fallback generation successful")
                return self.last_response
                
            except Exception as e2:
                logger.error(f"[Ollama] Fallback also failed: {e2}")
                # Return error in JSON format (for agent compatibility)
                return json.dumps({"tool": None, "answer": f"LLM error: Gemini ({e}) and Ollama ({e2}) both failed"})

    def _stream(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any):
        """
        Generate streaming response from Gemini with Ollama fallback.
        
        Args:
            prompt: Input prompt
            stop: Stop sequences (not used in Gemini)
            **kwargs: Additional arguments
            
        Yields:
            Text chunks as they are generated
        """
        # Try Gemini first
        try:
            full_response = ""
            for chunk in self.gemini_service.generate_stream(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=kwargs.get('max_tokens')
            ):
                full_response += chunk
                yield chunk
            
            self.last_response = full_response
            logger.info("[Gemini] Streaming successful")
            return
            
        except Exception as e:
            logger.warning(f"[Gemini] Streaming failed: {e}, trying Ollama fallback")
            
            # Fallback to Ollama
            try:
                full_response = ""
                for chunk in self.ollama_service.generate_stream(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=kwargs.get('max_tokens')
                ):
                    full_response += chunk
                    yield chunk
                
                self.last_response = full_response
                logger.info("[Ollama] Fallback streaming successful")
                return
                
            except Exception as e2:
                logger.error(f"[Ollama] Fallback streaming also failed: {e2}")
                yield f"[Error] Both Gemini ({e}) and Ollama ({e2}) failed"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "llm_type": "gemini"
        }

    # Additional helper methods for Gemini-specific features
    
    def generate_structured(
        self,
        prompt: str,
        output_schema: dict,
        temperature: Optional[float] = None
    ) -> dict:
        """
        Generate structured JSON output.
        
        Args:
            prompt: Input prompt
            output_schema: Expected JSON schema
            temperature: Override default temperature
            
        Returns:
            Parsed JSON object
        """
        return self.gemini_service.generate_structured(
            prompt=prompt,
            output_schema=output_schema,
            temperature=temperature or self.temperature
        )
    
    def classify_intent(
        self,
        query: str,
        context: Optional[str] = None
    ):
        """
        Classify user intent.
        
        Args:
            query: User query
            context: Optional conversation context
            
        Returns:
            IntentResult with classification
        """
        return self.gemini_service.classify_intent(query, context)
    
    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ):
        """
        Extract named entities.
        
        Args:
            text: Input text
            entity_types: Specific entity types to extract
            
        Returns:
            List of (entity_text, entity_type) tuples
        """
        return self.gemini_service.extract_entities(text, entity_types)


def create_llm(config):
    """
    Factory function to create Gemini-powered LLM with Ollama fallback.
    
    REPLACES: Old Ollama LocalLLM creation
    
    Primary: Google Gemini API
    Fallback: Local Ollama API
    
    Supports ENV-based configuration:
    - GEMINI_API_KEY: Google AI Studio API key
    - GEMINI_MODEL: Model name (default: gemini-2.0-flash-lite)
    - GEMINI_MAX_TOKENS: Max output tokens (default: 8192)
    - GEMINI_TIMEOUT: Request timeout in seconds (default: 120)
    - OLLAMA_BASE_URL: Ollama API endpoint (default: http://localhost:11434)
    - OLLAMA_MODEL: Ollama model name (default: llama3.1)
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LocalLLM instance with Gemini primary and Ollama fallback
    """
    # Create Gemini service
    gemini_service = create_gemini_service(config)
    
    # Create Ollama service as fallback
    ollama_service = create_ollama_service(
        base_url=config.get("ollama_host", "http://localhost:11434"),
        model=config.get("llm_model", "llama3.1"),
        timeout=config.get("ollama_timeout", 180)
    )
    
    # Create LocalLLM wrapper
    llm = LocalLLM(
        gemini_service=gemini_service,
        ollama_service=ollama_service,
        model_name=config.get("gemini_model", "gemini-2.0-flash-lite"),
        temperature=config.get("temperature", 0.2),
        timeout=config.get("gemini_timeout", 120),
        max_retries=config.get("gemini_max_retries", 3)
    )
    
    logger.info(f"✓ [Gemini+Ollama] Initialized LLM: primary={llm.model_name}, fallback=ollama:{ollama_service.model}, temperature={llm.temperature}")
    return llm


# ==============================================================================
# LEGACY OLLAMA SUPPORT (Deprecated)
# ==============================================================================

def _sanitize_host(host: str) -> str:
    """
    [DEPRECATED] Ollama host sanitization.
    
    Kept for backward compatibility only.
    """
    from urllib.parse import urlparse
    
    if not host:
        return "http://localhost:11434"
    
    if "://" not in host:
        host = "http://" + host
    
    parsed = urlparse(host)
    
    if parsed.hostname in ("0.0.0.0", "0.0.0.0:11434"):
        host = host.replace("0.0.0.0", "localhost")
    
    host = host.rstrip("/")
    
    return host
