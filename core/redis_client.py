import logging
import os
from typing import Optional
import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)
_redis_client: Optional[Redis] = None


async def get_redis() -> Optional[Redis]:
    global _redis_client

    if _redis_client:
        return _redis_client

    url = os.getenv("REDIS_URL")
    if not url:
        logger.info("Redis disabled: no REDIS_URL")
        return None

    try:
        _redis_client = redis.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
             health_check_interval=30,
        )
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable ({url}): {e}")
        return None


async def close_redis():
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
