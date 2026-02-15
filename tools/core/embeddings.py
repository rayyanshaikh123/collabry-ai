"""
Unified Embeddings Client - Provider-agnostic embeddings.

Supports multiple embedding providers:
- OpenAI (production, text-embedding-3-small)
- Sentence-Transformers (local, free)
- HuggingFace Inference API (cloud)

Provider switching via environment variables only.
"""

import os
from typing import List, Optional
from langchain_core.embeddings import Embeddings
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    from langchain_community.embeddings import OpenAIEmbeddings


class EmbeddingConfig:
    """Configuration for embedding provider."""
    
    def __init__(self):
        self.provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
        
        # Provider-specific configs
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.hf_api_key = os.getenv("HF_API_KEY")
    
    def __repr__(self):
        return f"EmbeddingConfig(provider={self.provider}, model={self.model})"


class SentenceTransformerEmbeddings(Embeddings):
    """Local sentence-transformers embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()


class HuggingFaceCloudEmbeddings(Embeddings):
    """HuggingFace Inference API embeddings."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", api_key: Optional[str] = None):
        import requests
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.session = requests.Session()
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts via HF API."""
        import requests
        
        response = self.session.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"HuggingFace API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                return result
            elif isinstance(result[0], float):
                return [result]
        
        raise Exception(f"Unexpected HuggingFace API response format: {type(result)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        # Process in batches to avoid rate limits
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._embed_batch([text])[0]


# Singleton instance
_config: Optional[EmbeddingConfig] = None
_embeddings: Optional[Embeddings] = None


def get_embedding_config() -> EmbeddingConfig:
    """Get or create embedding configuration singleton."""
    global _config
    if _config is None:
        _config = EmbeddingConfig()
    return _config


def get_embedding_client() -> Embeddings:
    """
    Get embedding client based on configuration.
    
    Returns:
        LangChain Embeddings instance
        
    Environment Variables:
        EMBEDDING_PROVIDER: openai | sentence-transformers | huggingface
        EMBEDDING_MODEL: Provider-specific model name
        
    Examples:
        # OpenAI
        EMBEDDING_PROVIDER=openai
        EMBEDDING_MODEL=text-embedding-3-small
        OPENAI_API_KEY=sk-...
        
        # Local sentence-transformers
        EMBEDDING_PROVIDER=sentence-transformers
        EMBEDDING_MODEL=all-MiniLM-L6-v2
        
        # HuggingFace API
        EMBEDDING_PROVIDER=huggingface
        EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
        HF_API_KEY=hf_...
    """
    global _embeddings
    
    if _embeddings is None:
        config = get_embedding_config()
        
        if config.provider == "openai":
            try:
                _embeddings = OpenAIEmbeddings(
                    model=config.model,
                    openai_api_key=config.openai_api_key,
                    openai_api_base=config.openai_base_url,
                )
            except TypeError:
                # Fallback for older versions
                _embeddings = OpenAIEmbeddings(
                    model=config.model,
                    openai_api_key=config.openai_api_key,
                )
        
        elif config.provider == "sentence-transformers":
            _embeddings = SentenceTransformerEmbeddings(model_name=config.model)
        
        elif config.provider == "huggingface":
            _embeddings = HuggingFaceCloudEmbeddings(
                model_name=config.model,
                api_key=config.hf_api_key
            )
        
        else:
            raise ValueError(
                f"Unknown embedding provider: {config.provider}. "
                f"Supported: openai, sentence-transformers, huggingface"
            )
    
    return _embeddings


def reset_embedding_client():
    """Reset embedding client singleton. Useful for testing."""
    global _config, _embeddings
    _config = None
    _embeddings = None


# Convenience functions
async def embed_text(text: str) -> List[float]:
    """Embed a single text string."""
    client = get_embedding_client()
    return await client.aembed_query(text)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed multiple text strings."""
    client = get_embedding_client()
    return await client.aembed_documents(texts)


# ========== BACKWARD COMPATIBILITY ==========
# For legacy code still using EmbeddingModel
class EmbeddingModel:
    """Backward compatibility wrapper for old embedding API."""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """Initialize with model_name and api_key (ignored, uses config)."""
        # Ignore parameters, use global config instead
        self._client = None
    
    def get_model(self):
        """Get the embeddings client."""
        if self._client is None:
            self._client = get_embedding_client()
        return self._client
    
    def embed_query(self, text: str) -> List[float]:
        """Synchronous embedding (wraps async)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        client = self.get_model()
        if loop.is_running():
            # We're already in an async context
            import nest_asyncio
            nest_asyncio.apply()
        
        return loop.run_until_complete(client.aembed_query(text))
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch embedding (wraps async)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        client = self.get_model()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        
        return loop.run_until_complete(client.aembed_documents(texts))
