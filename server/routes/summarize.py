"""
Summarization endpoint for text content.

Handles:
- Text summarization using LLM
- Configurable summary styles
- Length control
- Streaming and non-streaming responses
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from server.deps import get_current_user, get_user_id
from server.schemas import SummarizeRequest, SummarizeResponse, ErrorResponse
from core.agent import run_agent, chat
from config import CONFIG
import json
import logging
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["summarize"])


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Summarize text",
    description="Generate summary of provided text using AI"
)
async def summarize_text(
    request: SummarizeRequest,
    user_id: str = Depends(get_user_id)
) -> SummarizeResponse:
    """
    Summarize text content.
    
    - Extracts user_id from JWT token
    - Generates summary with configurable style
    - Returns summary with length statistics
    """
    try:
        logger.info(f"Summarization request from user={user_id}, style={request.style}")
        
        # Build summarization prompt based on style
        style_instructions = {
            "concise": "Provide a concise 2-3 sentence summary.",
            "detailed": "Provide a detailed summary covering main points (3-5 paragraphs).",
            "bullet": "Provide a bullet-point summary of key points."
        }
        
        instruction = style_instructions.get(request.style, style_instructions["concise"])
        
        # Add length constraint if specified
        if request.max_length:
            instruction += f" Keep summary under {request.max_length} words."
        
        # Build prompt
        prompt = f"""Summarize the following text.

Instructions: {instruction}

Text to summarize:
{request.text}

Summary:"""
        
        # Execute agent directly with new API
        session_id = str(uuid4())  # Temporary session for stateless operation
        summary = await chat(
            user_id=user_id,
            session_id=session_id,
            message=prompt,
            notebook_id=None
        )
        
        summary = summary.strip()
        
        logger.info(f"Summary generated: {len(summary)} chars from {len(request.text)} chars")
        
        return SummarizeResponse(
            summary=summary,
            original_length=len(request.text),
            summary_length=len(summary),
            user_id=user_id,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.exception(f"Summarization error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate summary: {str(e)}"
        )


@router.post(
    "/summarize/stream",
    summary="Streaming summarization endpoint",
    description="Summarize text using Server-Sent Events (SSE) streaming",
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def summarize_stream(
    request: SummarizeRequest,
    user_id: str = Depends(get_user_id)
):
    """
    Streaming summarization endpoint using SSE.
    
    Returns:
        StreamingResponse with Server-Sent Events
    """
    try:
        logger.info(f"Streaming summarization request from user={user_id}, style={request.style}")
        
        # Build summarization prompt based on style
        style_instructions = {
            "concise": "Provide a concise 2-3 sentence summary.",
            "detailed": "Provide a detailed summary covering main points (3-5 paragraphs).",
            "bullet": "Provide a bullet-point summary of key points."
        }
        
        instruction = style_instructions.get(request.style, style_instructions["concise"])
        
        # Add length constraint if specified
        if request.max_length:
            instruction += f" Keep summary under {request.max_length} words."
        
        # Build prompt
        prompt = f"""Summarize the following text.

Instructions: {instruction}

Text to summarize:
{request.text}

Summary:"""
        
        async def event_generator():
            """Generate SSE events that stream in real-time."""
            
            # Track if we've sent any data
            has_data = False
            session_id = str(uuid4())  # Temporary session for stateless operation
            
            # Stream the response using new agent API
            async for chunk in run_agent(
                user_id=user_id,
                session_id=session_id,
                message=prompt,
                notebook_id=None,
                stream=True
            ):
                if chunk.strip():
                    has_data = True
                    yield f"data: {chunk}\n\n"
            
            # Send completion event
            if has_data:
                yield f"event: done\ndata: \n\n"
            else:
                yield f"data: No summary generated\n\n"
                yield f"event: done\ndata: \n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logger.exception(f"Streaming summarization error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stream summary: {str(e)}"
        )
