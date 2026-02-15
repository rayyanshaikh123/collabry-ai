"""
Configuration - Environment-based configuration loader.

All configuration is loaded from environment variables.
No hardcoded values, no complex logic.

Provider switching happens via environment variables only.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)


def _first_non_empty_env(*keys: str) -> str | None:
    """Return the first non-empty environment variable value among keys."""
    for key in keys:
        val = os.getenv(key)
        if val is None:
            continue
        val = val.strip()
        if val:
            return val
    return None


def _mongo_uri_from_env() -> str | None:
    """Resolve MongoDB URI from common env var names.

    Supports both backend-style `MONGO_URI` and standard `MONGODB_URI`.
    Also considers `DATABASE_URL` if it looks like a Mongo URI.
    """
    uri = _first_non_empty_env(
        "MONGODB_URI",
        "MONGO_URI",
        "MONGO_URL",
        "MONGODB_URL",
        "DATABASE_URL",
    )
    if not uri:
        return None
    # Avoid accidentally treating non-mongo DATABASE_URLs as Mongo
    if not uri.lower().startswith("mongodb"):
        return None
    return uri


# Normalize Mongo URI env var for any modules that directly read MONGODB_URI.
_resolved_mongo_uri = _mongo_uri_from_env()
if _resolved_mongo_uri and not (os.getenv("MONGODB_URI") or "").strip():
    os.environ["MONGODB_URI"] = _resolved_mongo_uri


class Config:
    """Application configuration from environment variables."""
    
    # ========== Server Configuration ==========
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    RELOAD = os.getenv("RELOAD", "false").lower() == "true"
    
    # ========== LLM Configuration (OpenAI-compatible) ==========
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
    OPENAI_STREAMING = os.getenv("OPENAI_STREAMING", "true").lower() == "true"
    
    # Fine-tuned model configuration
    OPENAI_FINETUNED_MODEL = os.getenv("OPENAI_FINETUNED_MODEL")
    USE_FINETUNED_MODEL = os.getenv("USE_FINETUNED_MODEL", "false").lower() == "true"
    
    # ========== Embeddings Configuration ==========
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    
    # Provider-specific
    HF_API_KEY = os.getenv("HF_API_KEY")  # HuggingFace API key
    
    # ========== Vector Store Configuration ==========
    VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss").lower()
    
    # FAISS
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    
    # Verified Knowledge Base (Separate FAISS Index)
    VERIFIED_FAISS_INDEX_PATH = os.getenv("VERIFIED_FAISS_INDEX_PATH", "./data/verified_faiss_index")
    VERIFIED_KNOWLEDGE_ENABLED = os.getenv("VERIFIED_KNOWLEDGE_ENABLED", "true").lower() == "true"
    
    # Chroma
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")
    
    # Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west-2")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX", "study-assistant")
    
    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
    
    # ========== Database Configuration ==========
    MONGODB_URI = _mongo_uri_from_env() or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB = _first_non_empty_env("MONGODB_DB", "MONGO_DB", "DB_NAME") or "study_assistant"
    MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "conversations")
    
    # ========== Documents/RAG Storage ==========
    DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "documents")
    
    # ========== Authentication ==========
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-production")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "1440"))  # 24 hours
    
    # ========== Agent Configuration ==========
    MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))
    CONVERSATION_HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "10"))
    
    # ========== Enhanced Planner Configuration ==========
    PLANNER_SHOW_THINKING = os.getenv("PLANNER_SHOW_THINKING", "true").lower() == "true"
    PLANNER_MIN_CONFIDENCE = float(os.getenv("PLANNER_MIN_CONFIDENCE", "0.7"))
    PLANNER_USE_LLM_SANITIZATION = os.getenv("PLANNER_USE_LLM_SANITIZATION", "true").lower() == "true"
    PLANNER_VALIDATE_SCHEMAS = os.getenv("PLANNER_VALIDATE_SCHEMAS", "true").lower() == "true"
    
    
    # ========== RAG Configuration ==========
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    
    # ========== Backend Integration ==========
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")
    BACKEND_API_PREFIX = os.getenv("BACKEND_API_PREFIX", "/api")
    BACKEND_TIMEOUT = int(os.getenv("BACKEND_TIMEOUT", "30"))  # seconds
    
    # ========== CORS Configuration ==========
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]
    CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    
    # ========== Rate Limiting ==========
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    
    # Legacy compatibility for dict-style access
    _KEY_MAPPING = {
        "mongo_uri": "MONGODB_URI",
        "mongo_db": "MONGODB_DB",
        "jwt_secret_key": "JWT_SECRET_KEY",
        "jwt_algorithm": "JWT_ALGORITHM",
        "cors_origins": "CORS_ORIGINS",
        "ollama_host": "OPENAI_BASE_URL",  # Temporary mapping
        "documents_path": "DOCUMENTS_PATH",  # RAG retriever
        "faiss_index_path": "FAISS_INDEX_PATH",
        "embedding_model": "EMBEDDING_MODEL",
        "huggingface_api_key": "HF_API_KEY",
        "memory_collection": "MEMORY_COLLECTION",
        "retrieval_top_k": "RETRIEVAL_TOP_K",
    }
    
    def __getitem__(self, key):
        """Support dict-style access for backward compatibility."""
        # Check if there's a mapping
        mapped_key = self._KEY_MAPPING.get(key, key.upper())
        if hasattr(self, mapped_key):
            return getattr(self, mapped_key)
        # Try the key as-is
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Configuration key '{key}' not found")
    
    def get(self, key, default=None):
        """Support dict.get() method for backward compatibility."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def __contains__(self, key):
        """Support 'in' operator."""
        mapped_key = self._KEY_MAPPING.get(key, key.upper())
        return hasattr(self, mapped_key) or hasattr(self, key)
    
    @classmethod
    def validate(cls):
        """Validate critical configuration."""
        errors = []
        
        # Check required fields
        if cls.OPENAI_API_KEY == "dummy" and "openai.com" in cls.OPENAI_BASE_URL:
            errors.append("OPENAI_API_KEY is required when using OpenAI API")
        
        if not cls.MONGODB_URI:
            errors.append("MONGODB_URI is required")
        
        if cls.JWT_SECRET_KEY == "dev-secret-change-in-production":
            print("WARNING: Using default JWT secret. Change in production!")
        
        # Check vector store specific requirements
        if cls.VECTOR_STORE == "pinecone" and not cls.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY is required when using Pinecone")
        
        if cls.VECTOR_STORE == "qdrant" and not cls.QDRANT_URL:
            errors.append("QDRANT_URL is required when using Qdrant")
        
        # Check embedding provider requirements
        if cls.EMBEDDING_PROVIDER == "huggingface" and not cls.HF_API_KEY:
            errors.append("HF_API_KEY is required when using HuggingFace embeddings")
        
        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration (excluding secrets)."""
        print("=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        print(f"Server: {cls.HOST}:{cls.PORT}")
        print(f"LLM: {cls.OPENAI_MODEL} @ {cls.OPENAI_BASE_URL}")
        print(f"Embeddings: {cls.EMBEDDING_PROVIDER} / {cls.EMBEDDING_MODEL}")
        print(f"Vector Store: {cls.VECTOR_STORE}")
        print(f"Database: {cls.MONGODB_DB}")
        print("=" * 50)


# Singleton config instance
config = Config()

# Backward compatibility alias (uppercase)
CONFIG = config


# Validate on import
try:
    config.validate()
except ValueError as e:
    print(f"Configuration Error: {e}")
    # Don't fail on import, let the application decide


if __name__ == "__main__":
    # Print config when run directly
    config.print_config()
