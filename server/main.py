"""
FastAPI AI Core Server

Multi-user isolated AI backend with JWT authentication.

Features:
- JWT-based authentication
- User-isolated conversations (multi-session support)
- RAG document ingestion with background processing
- Summarization, Q&A, and mind map generation
- Streaming and non-streaming endpoints
"""
from fastapi import FastAPI, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from server.schemas import HealthResponse, ErrorResponse
from server.routes import chat, ingest, summarize, qa, mindmap, sessions, usage, studyplan
from server.deps import get_current_user
from server.middleware import UsageTrackingMiddleware
from server.limit_middleware import UsageLimitMiddleware
from core.usage_tracker import usage_tracker
from config import CONFIG
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting Collabry AI Core Server")
    logger.info(f"MongoDB: {CONFIG['mongo_uri']}")
    logger.info(f"LLM Model: {CONFIG['llm_model']}")
    logger.info(f"LLM Backend: {CONFIG['llm_backend']}")
    logger.info(f"JWT Algorithm: {CONFIG['jwt_algorithm']}")
    
    # Verify critical services
    try:
        from pymongo import MongoClient
        client = MongoClient(CONFIG["mongo_uri"], serverSelectionTimeoutMS=5000)
        client.server_info()
        logger.info("✓ MongoDB connection verified")
        client.close()
    except Exception as e:
        logger.error(f"✗ MongoDB connection failed: {e}")
        logger.warning("Server will start but database operations may fail")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Collabry AI Core Server")


# Create FastAPI app
app = FastAPI(
    title="Collabry AI Core API",
    description="Multi-user isolated AI backend with RAG, chat, and document processing",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
# Origins configured via CORS_ORIGINS environment variable
allowed_origins = CONFIG["cors_origins"]
logger.info(f"CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Environment-configurable origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add URL normalization middleware
class URLNormalizationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Normalize path by replacing multiple consecutive slashes with single slash
        import re
        normalized_path = re.sub(r'/+', '/', request.url.path)
        if normalized_path != request.url.path:
            request.scope['path'] = normalized_path
            logger.debug(f"Normalized path: {request.url.path} -> {normalized_path}")
        return await call_next(request)

# Add URL normalization middleware (before other middlewares)
app.add_middleware(URLNormalizationMiddleware)

# Add usage limit checking middleware
app.add_middleware(UsageLimitMiddleware)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler for unhandled errors.
    """
    logger.exception(f"Unhandled exception: {exc}")
    error_obj = ErrorResponse(
        error="Internal server error",
        detail=str(exc),
        timestamp=datetime.utcnow()
    ).dict()

    # Ensure all values are JSON serializable (datetime -> ISO string)
    safe_content = jsonable_encoder(error_obj)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=safe_content
    )


# Health check endpoint (no auth required)
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
    description="Check service health and component status"
)
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        Service status and component health with real-time usage stats
    """
    components = {}
    
    # Check MongoDB
    try:
        from pymongo import MongoClient
        client = MongoClient(CONFIG["mongo_uri"], serverSelectionTimeoutMS=2000)
        client.server_info()
        components["mongodb"] = "healthy"
        client.close()
    except Exception as e:
        components["mongodb"] = f"unhealthy: {str(e)}"
        logger.error(f"MongoDB health check failed: {e}")
    
    # Check Ollama
    try:
        import requests
        response = requests.get(f"{CONFIG['ollama_host']}/api/tags", timeout=2)
        if response.status_code == 200:
            components["ollama"] = "healthy"
        else:
            components["ollama"] = f"unhealthy: status {response.status_code}"
    except Exception as e:
        components["ollama"] = f"unhealthy: {str(e)}"
        logger.error(f"Ollama health check failed: {e}")
    
    # Overall status
    overall_status = "healthy" if all(
        "healthy" in status for status in components.values()
    ) else "degraded"
    
    # Get real-time usage stats
    usage_stats = None
    try:
        usage_stats = usage_tracker.get_realtime_stats()
    except Exception as e:
        logger.warning(f"Failed to get usage stats: {e}")
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        components=components,
        usage_stats=usage_stats,
        timestamp=datetime.utcnow()
    )


# Test auth endpoint
@app.get("/test-auth", tags=["health"])
async def test_auth(user_id: str = Depends(get_current_user)):
    """Test endpoint to verify JWT authentication is working"""
    logger.info(f"🎯 Test auth endpoint hit! User ID: {user_id}")
    return {"message": "Auth works!", "user_id": user_id}


# Include routers
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(ingest.router)
app.include_router(summarize.router)
app.include_router(qa.router)
app.include_router(mindmap.router)
app.include_router(usage.router)
app.include_router(studyplan.router)


# Root endpoint
@app.get(
    "/",
    tags=["root"],
    summary="API root",
    description="Welcome message and API information"
)
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Collabry AI Core API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "POST /ai/chat - Chat with AI assistant",
            "chat_stream": "POST /ai/chat/stream - Streaming chat (SSE)",
            "upload": "POST /ai/upload - Upload document for RAG",
            "summarize": "POST /ai/summarize - Summarize text",
            "summarize_stream": "POST /ai/summarize/stream - Streaming summarization (SSE)",
            "qa": "POST /ai/qa - Question answering with RAG",
            "qa_stream": "POST /ai/qa/stream - Streaming QA (SSE)",
            "qa_file": "POST /ai/qa/file - QA with file upload (PDF/TXT/MD, max 10MB)",
            "qa_file_stream": "POST /ai/qa/file/stream - Streaming QA with file (SSE)",
            "mindmap": "POST /ai/mindmap - Generate mind map",
            "sessions": "GET /ai/sessions - List user sessions",
            "create_session": "POST /ai/sessions - Create new session",
            "usage_stats": "GET /ai/usage/stats?days=7 - Public usage statistics (no auth)",
            "my_usage": "GET /ai/usage/me - Get my usage statistics",
            "global_usage": "GET /ai/usage/global?days=7 - Get global usage (admin, requires auth)",
            "realtime_stats": "GET /ai/usage/realtime - Get realtime stats (admin, requires auth)"
        },
        "authentication": "JWT Bearer token required (except /health and /)",
        "timestamp": datetime.utcnow()
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
