# core/local_llm.py
"""
COLLABRY AI ENGINE - HUGGING FACE LLM

Primary LLM using Hugging Face Inference API for cloud AI processing.

ARCHITECTURE:
- LocalLLM → Hugging Face Inference API
- Configurable model (Mistral, GPT-2, etc.)
- Cloud processing via HF API

BACKWARD COMPATIBILITY:
- LocalLLM class maintains same interface
- create_llm() factory function preserved
- All existing code using LocalLLM continues to work
- LangChain compatibility maintained

BENEFITS:
- Access to state-of-the-art open-source models
- No local hardware requirements
- Scalable and reliable
- Easy model switching
"""

import json
import logging
import inspect
from typing import Any, List, Mapping, Optional, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import Hugging Face service (cloud LLM)
from core.huggingface_service import HuggingFaceService, create_hf_service
from tools import load_tools

# LangChain compatibility
from langchain_core.language_models.llms import LLM


class LocalLLM(LLM):
    """
    Hugging Face-powered LLM wrapper.

    REPLACES: Old Ollama-based LocalLLM (now uses Hugging Face service)

    Uses Hugging Face Inference API as the primary LLM for cloud AI processing.

    Features:
    - Primary: Cloud Hugging Face API
    - Streaming support
    - JSON mode for structured outputs
    - LangChain compatible
    - Configurable models (Mistral, GPT-2, etc.)
    """
    hf_service: Any  # Hugging Face service instance
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
        return "huggingface"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Generate response from Hugging Face.

        Args:
            prompt: Input prompt
            stop: Stop sequences
            **kwargs: Additional arguments

        Returns:
            Generated text response
        """
        try:
            # Build functions metadata from available tools so the model can call tools
            functions = []
            function_call = "auto"
            
            # Disable function calling for synthesis prompts that expect JSON output
            if "Return ONLY this JSON" in prompt or "Output only:" in prompt:
                function_call = "none"
            else:
                try:
                    tools = load_tools()
                    for tname, tdef in tools.items():
                        func = tdef.get('func') if isinstance(tdef, dict) else tdef
                        desc = tdef.get('description', '') if isinstance(tdef, dict) else (func.__doc__ or '')
                        properties = {}
                        required = []
                        try:
                            sig = inspect.signature(func)
                            for p in sig.parameters.values():
                                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                                    properties[p.name] = {"type": "string"}
                                    if p.default is p.empty:
                                        required.append(p.name)
                        except Exception:
                            # ignore signature parsing errors
                            pass

                        functions.append({
                            "name": tname,
                            "description": desc,
                            "parameters": {
                                "type": "object",
                                "properties": properties,
                                "required": required,
                            },
                        })
                except Exception:
                    functions = []

            response = self.hf_service.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=kwargs.get('max_tokens'),
                functions=functions if function_call != "none" else None,
                function_call=function_call,
            )

            self.last_response = response
            logger.info("[HuggingFace] Generation successful")
            return self.last_response

        except Exception as e:
            logger.error(f"[HuggingFace] Generation failed: {e}")
            # Return error in JSON format (for agent compatibility)
            return json.dumps({"tool": None, "answer": f"LLM error: HuggingFace failed - {e}"})

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
            for chunk in self.hf_service.generate_stream(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=kwargs.get('max_tokens')
            ):
                full_response += chunk
                yield chunk

            self.last_response = full_response
            logger.info("[HuggingFace] Streaming successful")
            return

        except Exception as e:
            logger.error(f"[HuggingFace] Streaming failed: {e}")
            yield f"[Error] HuggingFace streaming failed: {e}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "llm_type": "huggingface"
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
                logger.warning(f"[HuggingFace] Failed to parse structured response as JSON: {response[:200]}...")
                return {"error": "Failed to parse response as JSON", "raw_response": response}

        except Exception as e:
            logger.error(f"[HuggingFace] Structured generation failed: {e}")
            return {"error": f"HuggingFace structured generation failed: {e}"}


def create_llm(config: dict) -> LocalLLM:
    """
    Factory function to create LocalLLM instance.

    Primary: Hugging Face Inference API

    Supports ENV-based configuration:
    - HUGGINGFACE_MODEL: HF model name (default: mistralai/Mistral-7B-Instruct-v0.1)
    - HUGGINGFACE_API_KEY: HF API key
    - LLM_TIMEOUT: Request timeout in seconds (default: 180)

    Args:
        config: Configuration dictionary

    Returns:
        LocalLLM instance with Hugging Face primary
    """
    # Create Hugging Face service (cloud LLM)
    hf_service = create_hf_service(
        model=config.get("llm_model"),
        timeout=config.get("llm_timeout", 180)
    )

    # Create LocalLLM wrapper backed by HuggingFaceService
    llm = LocalLLM(
        hf_service=hf_service,
        model_name=config.get("llm_model", "openai/gpt-oss-120b:groq"),
        temperature=config.get("temperature", 0.2),
        timeout=config.get("llm_timeout", 180),
        max_retries=config.get("llm_max_retries", 3)
    )

    logger.info(f"✓ [HuggingFace] Initialized LLM: model={llm.model_name}, temperature={llm.temperature}")
    return llm
