"""core/huggingface_service.py

Lightweight Hugging Face Router API wrapper using OpenAI client.

This module provides two primary methods:
- `generate(prompt, temperature, max_tokens)` — synchronous text generation
- `generate_stream(prompt, temperature, max_tokens)` — generator that yields
  the full response as a single chunk

Configuration:
 - HF_TOKEN in environment (REQUIRED)
 - model name passed in factory or via config (e.g. 'openai/gpt-oss-120b:groq', etc.)
"""
import logging
import json
from typing import Iterator, Optional, List, Dict, Any
from openai import OpenAI

from config import CONFIG  # project-level config

logger = logging.getLogger(__name__)


class HuggingFaceService:
    """Simple wrapper over the Hugging Face Router API using OpenAI client."""

    def __init__(self, model: str = "openai/gpt-oss-120b:groq", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or CONFIG.get("hf_token") or CONFIG.get("huggingface_api_key")
        if not self.api_key:
            raise ValueError("HF_TOKEN or HUGGINGFACE_API_KEY environment variable is required")
        
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.api_key
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[str] = None,
    ) -> str:
        """Generate text using Hugging Face Router API.

        Supports optional function-calling by passing `functions` metadata and
        `function_call` (e.g. 'auto'|'none'|{'name':...}). If the model returns a
        function call, this method will translate it into the agent's expected
        JSON decision format: {"tool": "<name>", "args": { ... }}.
        """
        try:
            req = dict(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens or 1000,
            )

            if functions:
                req["functions"] = functions
            if function_call is not None:
                req["function_call"] = function_call

            completion = self.client.chat.completions.create(**req)

            # Access message (be defensive: may be attr or dict)
            choice = completion.choices[0]
            message = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)

            # Helper to read nested fields with compatibility for both object/dict
            def _get(obj, key):
                if obj is None:
                    return None
                try:
                    return getattr(obj, key)
                except Exception:
                    try:
                        return obj.get(key)
                    except Exception:
                        return None

            # If model returned a function call, translate to agent tool JSON
            func_call = None
            if message:
                func_call = _get(message, "function_call")

            if func_call:
                # func_call may be an object with name/arguments or a dict-like
                name = None
                args_raw = None
                try:
                    name = func_call.name
                except Exception:
                    name = func_call.get("name") if isinstance(func_call, dict) else None

                try:
                    args_raw = func_call.arguments
                except Exception:
                    args_raw = func_call.get("arguments") if isinstance(func_call, dict) else None

                # arguments are often a JSON string - try to parse
                args = {}
                if args_raw:
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except Exception:
                            # Fallback: try to replace single quotes
                            try:
                                args = json.loads(args_raw.replace("'", '"'))
                            except Exception:
                                args = {"raw": args_raw}
                    elif isinstance(args_raw, dict):
                        args = args_raw

                decision = {"tool": name, "args": args}
                return json.dumps(decision)

            # Otherwise return normal content text
            content = _get(message, "content") if message else None
            if content:
                return content

            # As fallback, try to stringify the choice
            return str(choice)

        except Exception as e:
            logger.error(f"HuggingFace Router generation failed: {e}")
            raise

    def generate_stream(self, prompt: str, temperature: float = 0.2, max_tokens: Optional[int] = None) -> Iterator[str]:
        """Streaming support - yields the full response as one chunk."""
        try:
            text = self.generate(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
            yield text
        except Exception as e:
            logger.error(f"HuggingFace Router streaming failed: {e}")
            yield f"[Error] HuggingFace Router request failed: {e}"


def create_hf_service(model: str = None, timeout: int = 60) -> HuggingFaceService:
    model_name = model or CONFIG.get("llm_model") or "openai/gpt-oss-120b:groq"
    return HuggingFaceService(model=model_name)
