"""
Usage limit checking middleware.

Enforces subscription tier limits for AI operations.
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from core.usage_tracker import usage_tracker
from datetime import datetime, timedelta
import logging
from config import CONFIG

logger = logging.getLogger(__name__)


class UsageLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to check usage limits before processing AI requests."""
    
    # Subscription tier limits (AI questions per day)
    # Synced with backend/src/config/plans.js PLAN_LIMITS.aiQuestionsPerDay
    TIER_LIMITS = {
        "free": 10,        # 10 questions/day
        "starter": 50,     # 50 questions/day
        "basic": 50,       # legacy alias for starter
        "pro": -1,         # unlimited
        "unlimited": -1,   # unlimited
        "enterprise": -1   # legacy alias
    }
    
    # Voice Tutor turn limits per day
    VOICE_LIMITS = {
        "free": 0,         # no voice access
        "starter": 10,     # 10 voice turns/day
        "basic": 10,       # legacy alias
        "pro": 50,         # 50 voice turns/day
        "unlimited": -1,   # unlimited
        "enterprise": -1   # legacy alias
    }
    
    # Endpoints that consume question quota
    QUESTION_CONSUMING_ENDPOINTS = [
        "/ai/chat",
        "/ai/summarize",
        "/ai/qa",
        "/ai/sessions"
    ]

    # POST endpoints that should never count towards limits
    # (they don't call the LLM; e.g. rendering/conversion endpoints)
    NON_CONSUMING_POST_PATHS = {
    }
    
    async def dispatch(self, request: Request, call_next):
        """Check usage limits before processing request."""

        # Allow disabling in local dev (RATE_LIMIT_ENABLED=false)
        try:
            if not bool(CONFIG["rate_limit_enabled"]):
                return await call_next(request)
        except Exception:
            # If config lookup fails for any reason, don't block requests
            return await call_next(request)
        
        # Skip non-consuming POST endpoints
        if request.method != "GET" and request.url.path in self.NON_CONSUMING_POST_PATHS:
            return await call_next(request)

        # Only check for question-consuming endpoints
        if not any(request.url.path.startswith(endpoint) for endpoint in self.QUESTION_CONSUMING_ENDPOINTS):
            return await call_next(request)
        
        # Skip check for GET requests (they don't consume quota)
        if request.method == "GET":
            return await call_next(request)
        
        # Extract user_id from token
        user_id = None
        subscription_tier = "free"  # Default
        
        try:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                from jose import jwt
                
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
                # Get user's question count for today
                current_questions = usage_tracker.get_today_operations(user_id)

                # Get tier limit for this user (default to free)
                tier_limit = self.TIER_LIMITS.get(subscription_tier, self.TIER_LIMITS["free"])

                # -1 means unlimited
                if tier_limit == -1:
                    return await call_next(request)

                # Check if user has exceeded daily question limit
                if current_questions >= tier_limit:
                    logger.warning(f"User {user_id} exceeded question limit: {current_questions}/{tier_limit}")
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "message": f"Daily question limit exceeded ({tier_limit} questions/day). Resets in 24 hours.",
                            "current_usage": current_questions,
                            "limit": tier_limit,
                            "tier": subscription_tier,
                            "suggestion": "Upgrade your plan for higher daily limits or wait for daily reset",
                        },
                        headers={"Retry-After": "86400"},
                    )

                # Warn if approaching limit (>90%)
                usage_percentage = (current_questions / tier_limit) * 100 if tier_limit > 0 else 0
                if usage_percentage > 90:
                    logger.info(f"User {user_id} approaching limit: {usage_percentage:.1f}%")

            except Exception as e:
                logger.error(f"Error checking usage limits: {e}")
                # Don't block request if limit check fails
        
        # Process request
        return await call_next(request)
