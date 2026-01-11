"""
FastAPI dependencies for JWT authentication and user extraction.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from typing import Optional
from config import CONFIG
import logging

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Extract and validate user_id from JWT token.
    
    Args:
        credentials: HTTP Bearer credentials from request header
        
    Returns:
        user_id: Validated user identifier from JWT claims
        
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
        user_id: Optional[str] = payload.get("sub") or payload.get("id")
        
        if user_id is None:
            logger.warning(f"❌ JWT token missing 'sub' or 'id' claim. Payload: {payload}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user identifier",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Convert ObjectId to string if needed
        user_id = str(user_id)
        
        logger.info(f"✅ Authenticated user: {user_id}")
        return user_id
        
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
        return get_current_user(credentials)
    except HTTPException:
        return None


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
