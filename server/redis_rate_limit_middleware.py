from datetime import datetime
import logging

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from jose import jwt  # type: ignore

from core.redis_client import get_redis
from config import CONFIG

logger = logging.getLogger(__name__)


class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    """
    PHASE 5 — RATE LIMITING

    Simple per-user rate limiting using Redis:
    - Configurable requests per window per user_id
    - Counter key: rate:{user_id}
    - TTL: configurable window seconds
    - Atomic increment
    """

    def __init__(self, app):
        super().__init__(app)
        # Use config values for rate limiting
        self.MAX_REQUESTS = CONFIG.get("RATE_LIMIT_REQUESTS", 100)
        self.WINDOW_SECONDS = CONFIG.get("RATE_LIMIT_WINDOW", 60)
        self.ENABLED = CONFIG.get("RATE_LIMIT_ENABLED", True)
        logger.info(f"Rate limiting: {self.MAX_REQUESTS} requests per {self.WINDOW_SECONDS}s (enabled: {self.ENABLED})")

    async def dispatch(self, request: Request, call_next):
        # Skip if rate limiting is disabled
        if not self.ENABLED:
            return await call_next(request)
        
        # Only limit AI engine endpoints; health/docs/etc. remain unrestricted.
        if not request.url.path.startswith("/ai/"):
            return await call_next(request)

        # Best-effort Redis client acquisition
        try:
            redis = await get_redis()
        except Exception as e:
            logger.warning(f"Redis rate limiter: failed to acquire client: {e}")
            redis = None

        if redis is None:
            # Fail-open if Redis is not available
            return await call_next(request)

        # Extract user_id from JWT (same logic as other middlewares).
        user_id = None
        try:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.replace("Bearer ", "")
                payload = jwt.decode(
                    token,
                    CONFIG["jwt_secret_key"],
                    algorithms=[CONFIG["jwt_algorithm"]],
                )
                user_id = payload.get("sub") or payload.get("id")
        except Exception as e:
            logger.debug(f"Redis rate limiter: could not extract user info: {e}")

        if not user_id:
            # Anonymous or unauthenticated traffic is not counted here.
            return await call_next(request)

        key = f"rate:{user_id}"

        try:
            # Atomic increment; Redis guarantees this operation is atomic.
            current = await redis.incr(key)
            if current == 1:
                # First request in the window: set TTL
                await redis.expire(key, self.WINDOW_SECONDS)

            if current > self.MAX_REQUESTS:
                logger.warning(f"User {user_id} exceeded rate limit: {current}/{self.MAX_REQUESTS} in {self.WINDOW_SECONDS}s")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "message": "Too many requests. Please slow down.",
                        "limit": self.MAX_REQUESTS,
                        "window_seconds": self.WINDOW_SECONDS,
                        "current": int(current),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    headers={"Retry-After": str(self.WINDOW_SECONDS)},
                )
        except Exception as e:
            # Do not block requests on Redis failures
            logger.warning(f"Redis rate limiter error for user={user_id}: {e}")

        return await call_next(request)

