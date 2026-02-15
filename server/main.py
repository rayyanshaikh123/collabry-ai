"""
FastAPI AI Core Server — Production Ready

Multi-user isolated AI backend with JWT authentication.
"""

from fastapi import FastAPI, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
from datetime import datetime
import logging
import os
import requests

from server.schemas import HealthResponse, ErrorResponse
from server.routes import chat, ingest, summarize, qa, mindmap, sessions, usage, studyplan, planning_strategy, verified_knowledge
from server.deps import get_current_user
from server.middleware import UsageTrackingMiddleware
from server.limit_middleware import UsageLimitMiddleware
from server.redis_rate_limit_middleware import RedisRateLimitMiddleware
from core.usage_tracker import usage_tracker
from core.redis_client import get_redis, close_redis
from config import CONFIG


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("collabry.server")


# ---------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Collabry AI Core Server")

    # MongoDB check
    try:
        from pymongo import MongoClient
        client = MongoClient(CONFIG["mongo_uri"], serverSelectionTimeoutMS=3000)
        client.server_info()
        client.close()
        logger.info("✓ MongoDB reachable")
    except Exception:
        logger.warning("MongoDB not reachable at startup")

    # Redis init
    await get_redis()

    yield

    # Shutdown
    await close_redis()
    logger.info("Server shutdown complete")


# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
app = FastAPI(
    title="Collabry AI Core API",
    version="1.0.0",
    description="AI backend with RAG and streaming",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ---------------------------------------------------------------------
# Middleware (ORDER MATTERS)
# ---------------------------------------------------------------------
app.add_middleware(UsageTrackingMiddleware)
app.add_middleware(RedisRateLimitMiddleware)
app.add_middleware(UsageLimitMiddleware)

allowed_origins = CONFIG["cors_origins"]
cors_origin_regex = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error")

    error_obj = ErrorResponse(
        error="Internal server error",
        detail=str(exc),
        timestamp=datetime.utcnow()
    ).dict()

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(error_obj),
    )


# ---------------------------------------------------------------------
# Health & Readiness
# ---------------------------------------------------------------------
@app.get("/health", tags=["health"])
async def health():
    """Fast liveness probe"""
    return {"status": "alive", "timestamp": datetime.utcnow()}


@app.get("/ready", response_model=HealthResponse, tags=["health"])
async def readiness():
    """Readiness probe"""
    components = {}

    # Mongo
    try:
        from pymongo import MongoClient
        client = MongoClient(CONFIG["mongo_uri"], serverSelectionTimeoutMS=1500)
        client.server_info()
        client.close()
        components["mongodb"] = "healthy"
    except Exception:
        components["mongodb"] = "unreachable"

    # Redis
    try:
        r = await get_redis()
        if r:
            await r.ping()
            components["redis"] = "healthy"
        else:
            components["redis"] = "disabled"
    except Exception:
        components["redis"] = "unreachable"

    # Ollama
    try:
        res = requests.get(f"{CONFIG['ollama_host']}/api/tags", timeout=1.5)
        components["ollama"] = "healthy" if res.status_code == 200 else "unreachable"
    except Exception:
        components["ollama"] = "unreachable"

    overall = "healthy" if all(v in ("healthy", "disabled") for v in components.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version="1.0.0",
        components=components,
        usage_stats=None,
        timestamp=datetime.utcnow()
    )


# ---------------------------------------------------------------------
# Auth test
# ---------------------------------------------------------------------
@app.get("/test-auth", tags=["health"])
async def test_auth(user_id: str = Depends(get_current_user)):
    return {"message": "Auth works", "user_id": user_id}


# ---------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(ingest.router)
app.include_router(summarize.router)
app.include_router(qa.router)
app.include_router(mindmap.router)
app.include_router(usage.router)
app.include_router(studyplan.router)
app.include_router(planning_strategy.router)
app.include_router(verified_knowledge.router)


# ---------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------
@app.get("/", tags=["root"])
async def root():
    return {
        "message": "Collabry AI Core API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "timestamp": datetime.utcnow()
    }


# ---------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
