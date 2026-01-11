"""Configuration for COLLABRY.

All sensitive configuration should be in .env file (never committed to git).
Copy .env.example to .env and fill in your values.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

# Base project directory (assumes this file lives in project root)
ROOT = Path(__file__).parent

# Load environment variables from .env file if it exists
# This allows local development with .env while production uses system ENV
dotenv_path = ROOT / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path)
    print(f"✓ Loaded environment from: {dotenv_path}")
else:
    print(f"⚠️  No .env file found at {dotenv_path}, using system environment variables")
    print(f"   Copy .env.example to .env for local configuration")

CONFIG = {
    # ==============================================================================
    # GOOGLE GEMINI CONFIGURATION (Primary AI Engine)
    # ==============================================================================
    # Google AI Studio API key (get from: https://aistudio.google.com/app/apikey)
    "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
    
    # Gemini model selection
    # Options: gemini-2.0-flash-lite (fast, stable), gemini-1.5-pro (more capable)
    "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite"),
    
    # Gemini generation parameters
    "gemini_max_tokens": int(os.environ.get("GEMINI_MAX_TOKENS", "8192")),
    "gemini_timeout": int(os.environ.get("GEMINI_TIMEOUT", "120")),
    
    # ==============================================================================
    # LEGACY OLLAMA CONFIGURATION (Deprecated - kept for fallback)
    # ==============================================================================
    # Model name to use (change to 'mistral' or 'llama3.1' as installed)
    "llm_model": os.environ.get("OLLAMA_MODEL", os.environ.get("COLLABRY_LLM_MODEL", "llama3.1")),

    # Ollama base URL (standardized ENV variable)
    "ollama_host": os.environ.get("OLLAMA_BASE_URL", os.environ.get("OLLAMA_HOST", "http://localhost:11434")),
    
    # Ollama request timeout in seconds (default: 180s for artifact generation)
    "ollama_timeout": int(os.environ.get("OLLAMA_TIMEOUT", "180")),
    
    # Ollama retry configuration
    "ollama_max_retries": int(os.environ.get("OLLAMA_MAX_RETRIES", "3")),
    "ollama_retry_delay": float(os.environ.get("OLLAMA_RETRY_DELAY", "1.0")),

    # MongoDB settings (REQUIRED for memory persistence)
    "mongo_uri": os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
    "mongo_db": os.environ.get("MONGO_DB", "collabry"),
    "memory_collection": os.environ.get("MEMORY_COLLECTION", "conversations"),

    # Security settings (JWT authentication for production)
    "jwt_secret_key": os.environ.get("JWT_SECRET_KEY", "dev-secret-key-change-in-production"),
    "jwt_algorithm": os.environ.get("JWT_ALGORITHM", "HS256"),
    # jwt_expiration": int(os.environ.get("JWT_EXPIRATION", "86400")),  # 24 hours
    
    # CORS configuration (comma-separated origins)
    "cors_origins": os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,https://colab-back.onrender.com,http://127.0.0.1:3000,http://127.0.0.1:5000"
    ).split(","),

    # External API keys (optional, loaded from ENV only for security)
    "serper_api_key": os.environ.get("SERPER_API_KEY"),
    "huggingface_api_key": os.environ.get("HUGGINGFACE_API_KEY"),
    "stable_diffusion_api": os.environ.get("STABLE_DIFFUSION_API", "http://127.0.0.1:7860"),

    # Agent options
    "max_tool_calls": 3,

    # Temperature for LLM responses
    "temperature": float(os.environ.get("COLLABRY_TEMPERATURE", "0.2")),
    # Embedding model name (Hugging Face) - UPDATED for cloud API
    "embedding_model": os.environ.get("COLLABRY_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),

    # FAISS index path prefix (two files will be created: {prefix}.index and {prefix}.meta.json)
    "faiss_index_path": str(ROOT / "memory" / "faiss_index"),

    # Path to the directory containing documents for RAG
    "documents_path": str(ROOT / "documents"),

    # Retrieval options
    "retrieval_top_k": int(os.environ.get("COLLABRY_RETRIEVAL_TOP_K", "3")),
    # How often (in seconds) to checkpoint the vector store to disk. Set 0 to disable.
    "faiss_checkpoint_interval": int(os.environ.get("COLLABRY_FAISS_CHECKPOINT_INTERVAL", "60")),
    # Memory cache size for recent query/result caching
    "memory_cache_size": int(os.environ.get("COLLABRY_MEMORY_CACHE_SIZE", "128")),
    # Memory eviction / retention settings
    # Time-to-live for memory entries (seconds). Default 7 days.
    "memory_ttl_seconds": int(os.environ.get("COLLABRY_MEMORY_TTL_SECONDS", str(7 * 24 * 3600))),
    # Maximum number of persisted memory entries to keep (approx).
    "memory_max_items": int(os.environ.get("COLLABRY_MEMORY_MAX_ITEMS", "5000")),
    # Eviction policy: 'ttl', 'size', or 'hybrid' (both ttl and size)
    "memory_eviction_policy": os.environ.get("COLLABRY_MEMORY_EVICTION_POLICY", "hybrid"),
    # Background pruning interval (seconds). Set 0 to disable background pruning.
    "memory_prune_interval_seconds": int(os.environ.get("COLLABRY_MEMORY_PRUNE_INTERVAL_SECONDS", "3600")),
}

# =====================================================================
# LEGACY CLI SETTINGS (disabled for backend-only mode)
# These settings are preserved for backward compatibility with CLI mode
# but are not used in the FastAPI backend architecture
# =====================================================================

# Wake word detection settings (DISABLED for backend API mode)
# Legacy CLI feature - not used in FastAPI backend
CONFIG["wake_word_enabled"] = os.environ.get("COLLABRY_WAKE_WORD_ENABLED", "false").lower() == "true"
CONFIG["wake_words"] = []  # Empty for backend mode
CONFIG["wake_session_timeout"] = 0  # Not applicable in stateless API
CONFIG["wake_word_strict"] = False  # Deprecated

# Optional mapping of tool name synonyms. The agent will normalize model
# requested tool names using this dict before attempting execution. Add any
# extra aliases here (e.g. 'curl' -> 'web_scrape'). Keys are the variant
# names the model might emit; values are canonical tool names present in the
# tools registry.
CONFIG["tool_synonyms"] = {
    "curl": "web_scrape",
    "curl_get": "web_scrape",
    # Scraper variants
    "web-scraper": "web_scrape",
    "webscraper": "web_scrape",
    "web-scrape": "web_scrape",
    "web_scraper": "web_scrape",
    "scrape": "web_scrape",
    "scraper": "web_scrape",
    # Generic search synonyms to improve dynamic routing
    "search": "web_search",
    "websearch": "web_search",
    "web-search": "web_search",
    "lookup": "web_search",
    "google": "web_search",
    # File operation synonyms
    "save": "write_file",
    "save_file": "write_file",
    "note_down": "write_file",
    "create_file": "write_file",
    "write": "write_file",
    "store": "write_file",
    "read": "read_file",
    "load": "read_file",
    # Document generation synonyms
    "create_doc": "doc_generator",
    "make_doc": "doc_generator",
    "generate_doc": "doc_generator",
    "create_ppt": "ppt_generator",
    "make_ppt": "ppt_generator",
    "generate_ppt": "ppt_generator",
    "presentation": "ppt_generator",
}

# Preferred site mappings for web_search context (study-platform relevant)
CONFIG["preferred_sites"] = {
    "wikipedia": "https://wikipedia.org",
    "github": "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "arxiv": "https://arxiv.org",
    "scholar": "https://scholar.google.com",
}
    
# Allowlist persistence: file to persist allowed hosts/commands the user
# confirmed during interactive sessions. This keeps confirmations from
# prompting on every run. The file will be created under the memory folder.
CONFIG["allowlist_path"] = str(ROOT / "memory" / "allowed_hosts.json")
# Default runtime allowlist values (can be overridden by the persisted file)
CONFIG["allowed_url_hosts"] = []
CONFIG["allowed_commands"] = []


def ensure_paths():
    # Ensure memory directory exists for allowlist
    p = Path(CONFIG["allowlist_path"]).parent
    p.mkdir(parents=True, exist_ok=True)


ensure_paths()
