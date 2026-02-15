"""
FastAPI dependencies for JWT authentication and user extraction.
"""
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from typing import Optional, Dict
from config import CONFIG
import logging

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_user_api_key: Optional[str] = Header(None),
    x_user_base_url: Optional[str] = Header(None),
    x_user_provider: Optional[str] = Header(None)
) -> Dict:
    """
    Extract and validate user_id from JWT token.
    Also extract BYOK (Bring Your Own Key) headers if present.
    
    Args:
        credentials: HTTP Bearer credentials from request header
        x_user_api_key: Optional user's own API key (BYOK)
        x_user_base_url: Optional user's custom base URL
        x_user_provider: Optional user's provider (openai/groq/gemini)
        
    Returns:
        dict: {
            'user_id': str - Validated user identifier from JWT claims,
            'byok': dict or None - BYOK info if headers present
        }
        
    Raises:
        HTTPException: 401 if token is invalid or missing user_id
    """
    try:
        # Decode JWT token
        token = credentials.credentials
        logger.info(f"🔑 Received token: {token[:30]}...")
        logger.info(f"🔐 Using secret: {CONFIG['jwt_secret_key'][:20]}...")
        logger.info(f"🔧 Using algorithm: {CONFIG['jwt_algorithm']}")
        
        payload = jwt.decode(
            token,
            CONFIG["jwt_secret_key"],
            algorithms=[CONFIG["jwt_algorithm"]]
        )
        
        logger.info(f"✅ Token decoded successfully. Payload: {payload}")
        
        # Extract user_id from 'sub' or 'id' claim (backend uses 'id', standard is 'sub')
        user_id_raw = payload.get("sub") or payload.get("id")
        
        if user_id_raw is None:
            logger.warning(f"❌ JWT token missing 'sub' or 'id' claim. Payload: {payload}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user identifier",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Convert ObjectId to string if needed
        user_id = str(user_id_raw)
        
        logger.info(f"✅ Authenticated user: '{user_id}' (raw_type={type(user_id_raw).__name__}, str_len={len(user_id)})")
        
        # Build BYOK info if headers present
        byok = None
        if x_user_api_key and x_user_provider:
            byok = {
                "api_key": x_user_api_key,
                "base_url": x_user_base_url,
                "provider": x_user_provider
            }
            logger.info(f"🔑 [BYOK] User {user_id} using own {x_user_provider} key")
        
        return {
            "user_id": user_id,
            "byok": byok
        }
        
    except JWTError as e:
        logger.error(f"❌ JWT validation failed: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"❌ Unexpected error in JWT validation: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[str]:
    """
    Optional JWT authentication for public endpoints.
    
    Returns:
        user_id if token is valid, None otherwise
    """
    if credentials is None:
        return None
    
    try:
        user_info = get_current_user(credentials)
        return user_info["user_id"]
    except HTTPException:
        return None


def get_user_id(user_info: Dict = Depends(get_current_user)) -> str:
    """
    Backward-compatible function that returns just the user_id.
    Use this for endpoints that don't need BYOK support.
    
    Returns:
        user_id: str
    """
    return user_info["user_id"]


def is_admin(user_id: str, token: str) -> bool:
    """
    Check if the user has admin role.
    
    Args:
        user_id: The user ID from token
        token: The JWT token
        
    Returns:
        True if user is admin, False otherwise
    """
    try:
        payload = jwt.decode(
            token,
            CONFIG["jwt_secret_key"],
            algorithms=[CONFIG["jwt_algorithm"]]
        )
        role = payload.get("role", "user")
        return role == "admin"
    except Exception as e:
        logger.debug(f"Error checking admin role: {e}")
        return False
