# core/ollama_service.py
"""
COLLABRY AI ENGINE - OLLAMA SERVICE

Fallback LLM service using Ollama API.
Used when Gemini API is unavailable.
"""

import json
import logging
import requests
from typing import Any, Dict, Iterator, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class OllamaService:
    """Ollama API service for local LLM inference."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1", timeout: int = 180):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: Optional[int] = None) -> str:
        """Generate text using Ollama API."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens if max_tokens else 2048
                }
            }

            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate_stream(self, prompt: str, temperature: float = 0.2, max_tokens: Optional[int] = None) -> Iterator[str]:
        """Generate streaming text using Ollama API."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens if max_tokens else 2048
                }
            }

            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'response' in chunk:
                            yield chunk['response']
                        if chunk.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            yield f"[Error] Ollama request failed: {e}"

def create_ollama_service(base_url: str = "http://localhost:11434", model: str = "llama3.1", timeout: int = 180) -> OllamaService:
    """Factory function to create Ollama service."""
    return OllamaService(base_url=base_url, model=model, timeout=timeout)