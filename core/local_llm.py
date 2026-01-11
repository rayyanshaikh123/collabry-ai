# core/local_llm.py
"""
COLLABRY AI ENGINE - OLLAMA-POWERED LLM

Primary LLM using Ollama for local AI processing.

ARCHITECTURE:
- LocalLLM → Ollama API (localhost:11434 or configured host)
- Configurable model (llama3.1, mistral, etc.)
- Local processing, no external API dependencies

BACKWARD COMPATIBILITY:
- LocalLLM class maintains same interface
- create_llm() factory function preserved
- All existing code using LocalLLM continues to work
- LangChain compatibility maintained

BENEFITS:
- Full control over AI model
- No API rate limits or costs
- Privacy - all processing local
- Customizable models and parameters
"""

import json
import logging
from typing import Any, List, Mapping, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import Ollama service
from core.ollama_service import OllamaService, create_ollama_service

# LangChain compatibility
from langchain_core.language_models.llms import LLM


class LocalLLM(LLM):
    """
    Ollama-powered LLM wrapper.

    REPLACES: Old Gemini-based LocalLLM

    Uses Ollama as the primary LLM for local AI processing.

    Features:
    - Primary: Local Ollama API
    - Streaming support
    - JSON mode for structured outputs
    - LangChain compatibility
    """
    ollama_service: Any  # OllamaService instance
    model_name: str
    temperature: float
    timeout: int = 180
    max_retries: int = 3
    last_response: Optional[str] = None

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Generate response from Ollama.

        Args:
            prompt: Input prompt
            stop: Stop sequences
            **kwargs: Additional arguments

        Returns:
            Generated text response
        """
        try:
            response = self.ollama_service.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=kwargs.get('max_tokens')
            )

            self.last_response = response
            logger.info("[Ollama] Generation successful")
            return self.last_response

        except Exception as e:
            logger.error(f"[Ollama] Generation failed: {e}")
            # Return error in JSON format (for agent compatibility)
            return json.dumps({"tool": None, "answer": f"LLM error: Ollama failed - {e}"})

    def _stream(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any):
        """
        Generate streaming response from Ollama.

        Args:
            prompt: Input prompt
            stop: Stop sequences
            **kwargs: Additional arguments

        Yields:
            Text chunks as they are generated
        """
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
            logger.info("[Ollama] Streaming successful")
            return

        except Exception as e:
            logger.error(f"[Ollama] Streaming failed: {e}")
            yield f"[Error] Ollama streaming failed: {e}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "llm_type": "ollama"
        }

    # Additional helper methods for Ollama-specific features

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
        try:
            # Use JSON mode if available
            structured_prompt = f"{prompt}\n\nReturn your response as valid JSON matching this schema: {json.dumps(output_schema)}"

            response = self.ollama_service.generate(
                prompt=structured_prompt,
                temperature=temperature or self.temperature,
                max_tokens=4096
            )

            # Try to parse as JSON
            try:
                return json.loads(response.strip())
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw response wrapped in error format
                logger.warning(f"[Ollama] Failed to parse structured response as JSON: {response[:200]}...")
                return {"error": "Failed to parse response as JSON", "raw_response": response}

        except Exception as e:
            logger.error(f"[Ollama] Structured generation failed: {e}")
            return {"error": f"Ollama structured generation failed: {e}"}


def create_llm(config: dict) -> LocalLLM:
    """
    Factory function to create LocalLLM instance.

    Primary: Local Ollama API

    Supports ENV-based configuration:
    - OLLAMA_BASE_URL: Ollama API endpoint (default: http://localhost:11434)
    - OLLAMA_MODEL: Ollama model name (default: llama3.1)
    - OLLAMA_TIMEOUT: Request timeout in seconds (default: 180)

    Args:
        config: Configuration dictionary

    Returns:
        LocalLLM instance with Ollama primary
    """
    # Create Ollama service
    ollama_service = create_ollama_service(
        base_url=config.get("ollama_host", "http://localhost:11434"),
        model=config.get("llm_model", "llama3.1"),
        timeout=config.get("ollama_timeout", 180)
    )

    # Create LocalLLM wrapper
    llm = LocalLLM(
        ollama_service=ollama_service,
        model_name=config.get("llm_model", "llama3.1"),
        temperature=config.get("temperature", 0.2),
        timeout=config.get("ollama_timeout", 180),
        max_retries=config.get("ollama_max_retries", 3)
    )

    logger.info(f"✓ [Ollama] Initialized LLM: model={llm.model_name}, temperature={llm.temperature}")
    return llm
