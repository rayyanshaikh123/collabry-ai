"""
Unified LLM Client - Single Source of Truth for all LLM access.

This is the ONLY file that directly imports LLM SDKs.
All other modules must import from this file.

Supports any OpenAI-compatible API endpoint:
- OpenAI (production)
- LM Studio (local development)
- Ollama with OpenAI compatibility (local)
- Together AI, OpenRouter, etc.

Provider switching requires only environment variable changes.
"""

import os
import time
import asyncio
from typing import Optional
from openai import OpenAI, AsyncOpenAI
from langchain_openai import ChatOpenAI


# Rate limiting
_last_request_time = 0
_min_request_interval = 1.0  # Minimum seconds between requests


class LLMConfig:
    """Configuration for OpenAI-compatible LLM client with multi-provider support."""
    
    def __init__(self):
        # Determine provider
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        # Provider-specific configurations
        provider_configs = {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY", "dummy"),
                "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            },
            "groq": {
                "api_key": os.getenv("GROQ_API_KEY", ""),
                "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            },
            "ollama": {
                "api_key": "ollama",  # Ollama doesn't need a real key
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
            },
            "together": {
                "api_key": os.getenv("TOGETHER_API_KEY", ""),
                "base_url": os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
                "model": os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
            }
        }
        
        # Get provider config or default to OpenAI
        config = provider_configs.get(provider, provider_configs["openai"])
        
        self.provider = provider
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        self.model = config["model"]
        
        # Support fine-tuned models (OpenAI only)
        if provider == "openai":
            self.finetuned_model = os.getenv("OPENAI_FINETUNED_MODEL")
            self.use_finetuned = os.getenv("USE_FINETUNED_MODEL", "false").lower() == "true"
            if self.use_finetuned and self.finetuned_model:
                self.model = self.finetuned_model
        
        # Shared parameters (work with all providers)
        self.temperature = float(os.getenv("LLM_TEMPERATURE", os.getenv("OPENAI_TEMPERATURE", "0.7")))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", os.getenv("OPENAI_MAX_TOKENS", "4096")))
        self.streaming = os.getenv("LLM_STREAMING", os.getenv("OPENAI_STREAMING", "true")).lower() == "true"
        
        # Rate limiting
        self.request_delay = float(os.getenv("LLM_REQUEST_DELAY", os.getenv("OPENAI_REQUEST_DELAY", "1.0")))
    
    def __repr__(self):
        return f"LLMConfig(provider={self.provider}, base_url={self.base_url}, model={self.model})"


# Singleton instance
_config: Optional[LLMConfig] = None
_client: Optional[OpenAI] = None
_async_client: Optional[AsyncOpenAI] = None
_langchain_client: Optional[ChatOpenAI] = None


def get_llm_config() -> LLMConfig:
    """Get or create LLM configuration singleton."""
    global _config
    if _config is None:
        _config = LLMConfig()
    return _config


def get_openai_client() -> OpenAI:
    """
    Get OpenAI client for synchronous operations.
    
    Returns:
        OpenAI client configured with environment variables
        
    Example:
        >>> client = get_openai_client()
        >>> response = client.chat.completions.create(...)
    """
    global _client
    if _client is None:
        config = get_llm_config()
        _client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    return _client


def get_async_openai_client() -> AsyncOpenAI:
    """
    Get async OpenAI client for async operations.
    
    Returns:
        AsyncOpenAI client configured with environment variables
        
    Example:
        >>> client = get_async_openai_client()
        >>> response = await client.chat.completions.create(...)
    """
    global _async_client
    if _async_client is None:
        config = get_llm_config()
        _async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    return _async_client


def get_langchain_llm() -> ChatOpenAI:
    """
    Get LangChain-compatible LLM for agent orchestration.
    
    This is used by the agent executor for tool calling.
    Rate-limited to prevent 429 errors.
    
    Returns:
        ChatOpenAI instance configured with environment variables
        
    Example:
        >>> llm = get_langchain_llm()
        >>> agent = create_openai_tools_agent(llm, tools, prompt)
    """
    global _langchain_client, _last_request_time
    
    # Apply rate limiting
    config = get_llm_config()
    current_time = time.time()
    time_since_last = current_time - _last_request_time
    
    if time_since_last < config.request_delay:
        # Sleep to respect rate limit
        time.sleep(config.request_delay - time_since_last)
    
    _last_request_time = time.time()
    
    if _langchain_client is None:
        try:
            # Try newer parameter names first
            _langchain_client = ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                openai_api_key=config.api_key,
                openai_api_base=config.base_url,
                streaming=config.streaming,
            )
        except TypeError:
            # Fallback to older parameter names
            _langchain_client = ChatOpenAI(
                model_name=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                openai_api_key=config.api_key,
                openai_api_base=config.base_url,
                streaming=config.streaming,
            )
    return _langchain_client


def reset_clients():
    """Reset all client singletons. Useful for testing or config changes."""
    global _config, _client, _async_client, _langchain_client
    _config = None
    _client = None
    _async_client = None
    _langchain_client = None


# Convenience function for direct chat completions
async def chat_completion(
    messages: list[dict],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    **kwargs
):
    """
    Convenience function for chat completions.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        temperature: Override default temperature
        max_tokens: Override default max tokens
        stream: Enable streaming
        **kwargs: Additional OpenAI API parameters
    
    Returns:
        Chat completion response
    """
    config = get_llm_config()
    client = get_async_openai_client()
    
    return await client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=temperature or config.temperature,
        max_tokens=max_tokens or config.max_tokens,
        stream=stream,
        **kwargs
    )
