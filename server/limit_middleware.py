"""
Usage limit checking middleware.

Enforces subscription tier limits for AI operations.
"""
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from core.usage_tracker import usage_tracker
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class UsageLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to check usage limits before processing AI requests."""
    
    # Subscription tier limits (tokens per day - resets every 24 hours)
    TIER_LIMITS = {
        "free": 10000,
        "basic": 50000,
        "pro": 200000,
        "enterprise": 1000000
    }
    
    # Endpoints that consume tokens
    TOKEN_CONSUMING_ENDPOINTS = [
        "/ai/chat",
        "/ai/summarize",
        "/ai/qa",
        "/ai/mindmap",
        "/ai/sessions"
    ]
    
    async def dispatch(self, request: Request, call_next):
        """Check usage limits before processing request."""
        
        # Skip OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Only check for token-consuming endpoints
        if not any(request.url.path.startswith(endpoint) for endpoint in self.TOKEN_CONSUMING_ENDPOINTS):
            return await call_next(request)
        
        # Skip check for GET requests (they don't consume tokens)
        if request.method == "GET":
            return await call_next(request)
        
        # Extract user_id from token
        user_id = None
        subscription_tier = "free"  # Default
        
        try:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                from jose import jwt
                from config import CONFIG
                
                token = auth_header.replace("Bearer ", "")
                payload = jwt.decode(
                    token,
                    CONFIG["jwt_secret_key"],
                    algorithms=[CONFIG["jwt_algorithm"]]
                )
                user_id = payload.get("sub") or payload.get("id")
                subscription_tier = payload.get("subscriptionTier", "free")
        except Exception as e:
            logger.debug(f"Could not extract user info for limit check: {e}")
        
        # If we have a user_id, check their usage
        if user_id:
            try:
                # Get user's usage for today (aggregated daily_stats is efficient)
                current_tokens = usage_tracker.get_today_tokens(user_id)

                # Get tier limit for this user (default to free)
                tier_limit = self.TIER_LIMITS.get(subscription_tier, self.TIER_LIMITS["free"])

                # Check if user has exceeded daily limit
                if current_tokens >= tier_limit:
                    logger.warning(f"User {user_id} exceeded usage limit: {current_tokens}/{tier_limit}")
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "message": f"Daily token limit exceeded ({tier_limit:,} tokens). Resets in 24 hours.",
                            "current_usage": current_tokens,
                            "limit": tier_limit,
                            "tier": subscription_tier,
                            "suggestion": "Upgrade your plan for higher daily limits or wait for daily reset"
                        }
                    )

                # Warn if approaching limit (>90%)
                usage_percentage = (current_tokens / tier_limit) * 100 if tier_limit > 0 else 0
                if usage_percentage > 90:
                    logger.info(f"User {user_id} approaching limit: {usage_percentage:.1f}%")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error checking usage limits: {e}")
                # Don't block request if limit check fails
        
        # Process request
        return await call_next(request)
