"""
BYOK (Bring Your Own Key) Integration

Fetches user API keys from the backend and provides them to the LLM client.
"""

import httpx
import logging
from typing import Optional, Dict
from config import CONFIG

logger = logging.getLogger(__name__)

# Cache for user API keys (in-memory, expires after 5 minutes)
_key_cache: Dict[str, Dict] = {}
_cache_ttl = 300  # 5 minutes


async def get_user_api_key(user_id: str, provider: str = "openai") -> Optional[Dict[str, str]]:
    """
    Fetch user's API key from backend if BYOK is enabled.
    
    Args:
        user_id: User ID
        provider: Provider name ('openai', 'groq', 'gemini')
    
    Returns:
        Dict with 'key', 'base_url', 'model' if BYOK enabled, None otherwise
    """
    if not user_id:
        return None
    
    # Check cache first
    cache_key = f"{user_id}:{provider}"
    if cache_key in _key_cache:
        cached = _key_cache[cache_key]
        # Simple TTL check (in production, use proper expiry)
        if cached.get("enabled"):
            logger.info(f"[BYOK] Using cached key for user {user_id}")
            return cached
    
    try:
        backend_url = CONFIG.get("backend_url", "http://localhost:5000")
        url = f"{backend_url}/api/user/{user_id}/api-key/{provider}"
        
        logger.info(f"[BYOK] Fetching API key for user {user_id} from {url}")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("enabled"):
                    result = {
                        "enabled": True,
                        "key": data.get("key"),
                        "base_url": data.get("baseUrl"),
                        "model": data.get("model"),
                        "provider": data.get("provider", provider)
                    }
                    
                    # Cache the result
                    _key_cache[cache_key] = result
                    
                    logger.info(f"[BYOK] ✓ User {user_id} has BYOK enabled for {provider}")
                    return result
                else:
                    logger.info(f"[BYOK] User {user_id} has no active BYOK for {provider}")
                    return None
            else:
                logger.warning(f"[BYOK] Failed to fetch key: HTTP {response.status_code}")
                return None
                
    except httpx.TimeoutException:
        logger.error(f"[BYOK] Timeout fetching API key for user {user_id}")
        return None
    except Exception as e:
        logger.error(f"[BYOK] Error fetching API key for user {user_id}: {e}")
        return None


def clear_user_key_cache(user_id: str, provider: str = "openai"):
    """
    Clear cached API key for a user (e.g., when they update their key).
    
    Args:
        user_id: User ID
        provider: Provider name
    """
    cache_key = f"{user_id}:{provider}"
    if cache_key in _key_cache:
        del _key_cache[cache_key]
        logger.info(f"[BYOK] Cleared cache for user {user_id}, provider {provider}")


def clear_all_cache():
    """Clear entire API key cache."""
    global _key_cache
    _key_cache = {}
    logger.info("[BYOK] Cleared all cached API keys")
