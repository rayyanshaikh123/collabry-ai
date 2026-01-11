"""
Chat endpoint for conversational AI interactions.

Handles:
- Multi-user isolated conversations
- Session management
- Streaming and non-streaming responses
- Tool invocation tracking
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from server.deps import get_current_user
from server.schemas import ChatRequest, ChatResponse, ErrorResponse
from core.agent import create_agent
from config import CONFIG
import logging
from uuid import uuid4
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Chat with AI assistant",
    description="Send a message to the AI assistant with conversation continuity via session_id"
)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_current_user)
) -> ChatResponse:
    """
    Chat endpoint with user-isolated conversation memory.
    
    - Extracts user_id from JWT token
    - Maintains conversation context via session_id
    - Supports tool invocation (web search, document generation, etc.)
    """
    try:
        # Generate session_id if not provided
        session_id = request.session_id or str(uuid4())
        
        logger.info(f"Chat request from user={user_id}, session={session_id}")
        
        # Create user-isolated agent
        agent, _, _, memory = create_agent(
            user_id=user_id,
            session_id=session_id,
            config=CONFIG
        )
        
        # Collect response chunks
        response_chunks = []
        tool_used = None
        
        def collect_chunk(chunk: str):
            response_chunks.append(chunk)
        
        # Execute agent with streaming collection
        agent.handle_user_input_stream(request.message, collect_chunk)
        
        # Check if tool was used
        if hasattr(agent, 'last_tool_called') and agent.last_tool_called:
            tool_used = agent.last_tool_called
        
        # Combine chunks into full response
        full_response = "".join(response_chunks)
        
        logger.info(f"Chat response generated: {len(full_response)} chars, tool={tool_used}")
        
        return ChatResponse(
            response=full_response,
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            tool_used=tool_used
        )
        
    except Exception as e:
        logger.exception(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )


@router.post(
    "/chat/stream",
    summary="Streaming chat endpoint",
    description="Chat with AI assistant using Server-Sent Events (SSE) streaming",
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def chat_stream(
    request: ChatRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Streaming chat endpoint using SSE.
    
    Returns:
        StreamingResponse with Server-Sent Events
    """
    try:
        session_id = request.session_id or str(uuid4())
        
        logger.info(f"Streaming chat request from user={user_id}, session={session_id}")
        
        # Create user-isolated agent
        agent, _, _, _ = create_agent(
            user_id=user_id,
            session_id=session_id,
            config=CONFIG
        )
        
        async def event_generator():
            """Generate SSE events that stream in real-time."""
            
            def stream_chunk(chunk: str):
                """Callback that yields chunks immediately."""
                if chunk.strip():
                    # Yield each chunk as it's generated
                    return f"data: {chunk}\n\n"
                return None
            
            # Track if we've sent any data
            has_data = False
            
            # Execute agent with immediate streaming
            chunks_buffer = []
            def collect_chunk(chunk: str):
                chunks_buffer.append(chunk)
            
            agent.handle_user_input_stream(request.message, collect_chunk)
            
            # Stream the collected chunks
            for chunk in chunks_buffer:
                if chunk.strip():
                    has_data = True
                    yield f"data: {chunk}\n\n"
            
            # Send completion event (without session_id in data)
            if has_data:
                yield f"event: done\ndata: \n\n"
            else:
                yield f"data: No response generated\n\n"
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
        logger.exception(f"Streaming chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stream chat response: {str(e)}"
        )
