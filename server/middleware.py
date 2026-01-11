"""
Middleware for automatic usage tracking.

Tracks AI operations and logs usage metrics automatically.
Each operation consumes ~100 tokens from the user's daily limit.
Daily limits reset every 24 hours based on UTC timezone.
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from core.usage_tracker import usage_tracker
from datetime import datetime
import time
import logging
import re

logger = logging.getLogger(__name__)


class UsageTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically track AI usage."""
    
    # Endpoints to track
    TRACKED_ENDPOINTS = {
        "/ai/chat": "chat",
        "/ai/chat/stream": "chat_stream",
        "/ai/summarize": "summarize",
        "/ai/summarize/stream": "summarize_stream",
        "/ai/qa": "qa",
        "/ai/qa/stream": "qa_stream",
        "/ai/qa/file": "qa_file",
        "/ai/qa/file/stream": "qa_file_stream",
        "/ai/mindmap": "mindmap",
        "/ai/upload": "upload"
    }
    
    # Regex patterns for dynamic endpoints
    SESSION_MESSAGE_PATTERN = re.compile(r'^/ai/sessions/[^/]+/messages$')
    
    async def dispatch(self, request: Request, call_next):
        """Process request and track usage."""
        start_time = time.time()
        path = request.url.path
        
        # Check if this endpoint should be tracked
        operation_type = None
        for endpoint, op_type in self.TRACKED_ENDPOINTS.items():
            if path.startswith(endpoint):
                operation_type = op_type
                break
        
        # Check for session messages endpoint (POST only for sending messages)
        if not operation_type and request.method == "POST" and self.SESSION_MESSAGE_PATTERN.match(path):
            operation_type = "session_chat"
        
        # If not a tracked endpoint, pass through
        if not operation_type:
            return await call_next(request)
        
        # Get user ID from request state (set by auth dependency)
        user_id = None
        try:
            # Extract JWT token from Authorization header
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
                # Extract user_id from 'sub' or 'id' claim
                user_id = payload.get("sub") or payload.get("id")
                logger.debug(f"Extracted user_id from token: {user_id}")
        except Exception as e:
            logger.debug(f"Could not extract user_id: {e}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Determine success based on status code
        success = 200 <= response.status_code < 300
        
        # Estimate tokens used based on operation type
        tokens_used = self._estimate_tokens(operation_type, request, response)
        
        # Log usage if we have a user_id
        if user_id:
            try:
                usage_tracker.log_operation(
                    user_id=user_id,
                    endpoint=path,
                    operation_type=operation_type,
                    tokens_used=tokens_used,
                    success=success,
                    response_time_ms=response_time_ms,
                    metadata={
                        "method": request.method,
                        "status_code": response.status_code
                    }
                )
            except Exception as e:
                logger.error(f"Failed to log usage: {e}")
        
        return response
    
    def _estimate_tokens(self, operation_type: str, request: Request, response) -> int:
        """
        Estimate tokens used based on operation type.
        
        This is a rough estimation. In production, you'd want to:
        1. Extract actual token counts from the LLM response
        2. Use tiktoken or similar for accurate counting
        3. Store token counts in response headers
        
        Args:
            operation_type: Type of operation
            request: Request object
            response: Response object
            
        Returns:
            Estimated token count
        """
        # Base token estimates by operation type
        # Set to ~100 tokens per operation as requested
        token_estimates = {
            "chat": 100,
            "chat_stream": 100,
            "session_chat": 100,  # Session-based chat
            "summarize": 100,
            "summarize_stream": 100,
            "qa": 100,
            "qa_stream": 100,
            "qa_file": 100,
            "qa_file_stream": 100,
            "mindmap": 100,
            "upload": 100
        }
        
        return token_estimates.get(operation_type, 100)
