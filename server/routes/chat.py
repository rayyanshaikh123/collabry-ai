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
from core.agent import run_agent, chat as agent_chat
from core.artifact_prompts import build_artifact_prompt
from config import CONFIG
import logging
from uuid import uuid4
from datetime import datetime
import json

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
    user_info: dict = Depends(get_current_user)
) -> ChatResponse:
    """
    Chat endpoint with user-isolated conversation memory.
    
    - Extracts user_id from JWT token
    - Maintains conversation context via session_id
    - Supports tool invocation (web search, document generation, etc.)
    - Supports BYOK (Bring Your Own Key) if user provides API key
    """
    try:
        user_id = user_info["user_id"]
        byok = user_info.get("byok")
        
        # Generate session_id if not provided
        session_id = request.session_id or str(uuid4())
        
        logger.info(f"Chat request from user={user_id}, session={session_id}")

        is_artifact = (request.type or "").lower() == "artifact_request"
        artifact_type = (request.artifact or "").lower() if is_artifact else None
        topic = (request.topic or "").strip() if is_artifact else None

        if is_artifact:
            if not artifact_type or not topic:
                raise HTTPException(status_code=400, detail="Artifact request requires 'artifact' and 'topic'")
            internal_prompt = build_artifact_prompt(
                artifact_type,
                topic,
                (request.artifact_params or {}) if isinstance(request.artifact_params, dict) else {}
            )
            agent_message = internal_prompt
        else:
            if not request.message or not request.message.strip():
                raise HTTPException(status_code=400, detail="message is required for chat requests")
            agent_message = request.message
        
        # Use new agent architecture with BYOK support
        response = await agent_chat(
            user_id=user_id,
            session_id=session_id,
            message=agent_message,
            notebook_id=request.notebook_id,
            source_ids=request.source_ids,
            verified_mode=request.verified_mode
        )
        
        logger.info(f"Chat response generated: {len(response)} chars")
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            tool_used=None  # New agent doesn't expose this directly
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
    user_info: dict = Depends(get_current_user)
):
    """
    Streaming chat endpoint using SSE.
    Supports BYOK (Bring Your Own Key) if user provides API key.
    
    Returns:
        StreamingResponse with Server-Sent Events
    """
    try:
        user_id = user_info["user_id"]
        byok = user_info.get("byok")
        
        session_id = request.session_id or str(uuid4())
        
        logger.info(f"Streaming chat request from user={user_id}, session={session_id}")

        async def event_generator():
            """Generate SSE events that stream tool calls and responses from ReAct agent."""
            has_data = False
            
            try:
                # Use new agent with streaming (returns event dicts)
                async for event in run_agent(
                    user_id=user_id,
                    session_id=session_id,
                    message=request.message,
                    notebook_id=request.notebook_id,
                    use_rag=bool(request.use_rag) or bool(request.source_ids),
                    source_ids=request.source_ids,
                    stream=True,
                    verified_mode=request.verified_mode
                ):
                    if event:
                        has_data = True
                        event_type = event.get("type")
                        
                        if event_type == "tool_start":
                            # Tool invocation started
                            yield f"data: {json.dumps(event)}\n\n"
                        
                        elif event_type == "tool_end":
                            # Tool completed
                            yield f"data: {json.dumps(event)}\n\n"
                        
                        elif event_type == "token":
                            # Content token (backward compatible with old format)
                            yield f"data: {json.dumps({'content': event['content']})}\n\n"
                        
                        elif event_type == "complete":
                            # Final completion
                            yield f"data: {json.dumps({'done': True, 'message': event['message']})}\n\n"
                        
                        elif event_type == "error":
                            # Error occurred
                            yield f"data: {json.dumps(event)}\n\n"
                
                # Send completion event if no explicit complete was sent
                if has_data:
                    yield f"data: {json.dumps({'done': True})}\n\n"
                else:
                    yield f"data: {json.dumps({'content': 'No response generated'})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
            
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
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
