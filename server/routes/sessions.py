"""
Chat Sessions Management
Handles session creation, retrieval, and rate limiting
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
from server.deps import get_current_user, security
from server.schemas import ErrorResponse, ChatRequest
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from config import CONFIG
from core.agent import run_agent
from core.session_state import delete_session_state
from core.artifact_prompts import build_artifact_prompt
from core.redis_client import get_redis
import logging
import json

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai/sessions", tags=["sessions"])

# MongoDB connection
mongo_client = MongoClient(CONFIG["mongo_uri"])
db = mongo_client[CONFIG["mongo_db"]]
sessions_collection = db["chat_sessions"]
messages_collection = db["chat_messages"]
notebooks_collection = db["notebooks"]

# Create indexes
sessions_collection.create_index([("user_id", 1), ("created_at", DESCENDING)])
sessions_collection.create_index([("notebook_id", 1)])
messages_collection.create_index([("session_id", 1), ("timestamp", 1)])


def verify_session_access(session_id: str, user_id: str) -> dict:
    """
    Verify if a user has access to a session.
    Access is granted if:
    1. The user is the owner of the session.
    2. The session is linked to a notebook where the user is a collaborator/owner.
    """
    try:
        session_obj_id = ObjectId(session_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    session = sessions_collection.find_one({"_id": session_obj_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 1. Direct ownership check
    if session.get("user_id") == user_id:
        return session

    # 2. Collaborative notebook check
    notebook_id = session.get("notebook_id")
    
    # 2a. Legacy recovery: If session has no notebook_id, search notebooks by aiSessionId
    if not notebook_id:
        logger.info(f"🔄 Recovering notebook_id for legacy session {session_id}")
        notebook_doc = notebooks_collection.find_one({"aiSessionId": session_id})
        if notebook_doc:
            notebook_id = str(notebook_doc["_id"])
            # Update session object in memory for current request
            session["notebook_id"] = notebook_id
            # Backfill for next time
            sessions_collection.update_one(
                {"_id": session_obj_id},
                {"$set": {"notebook_id": notebook_id}}
            )
            logger.info(f"✅ Backfilled notebook_id {notebook_id} to session {session_id}")
        else:
            logger.warning(f"⚠️ No notebook found for legacy session {session_id}")

    if notebook_id:
        try:
            # Backend stores IDs as ObjectIds
            nb_obj_id = ObjectId(notebook_id)
            user_obj_id = ObjectId(user_id)
        except Exception:
            # Fallback if IDs are not valid ObjectIds
            nb_obj_id = notebook_id
            user_obj_id = user_id

        # Try both ObjectId and string matches for robustness
        query = {
            "_id": nb_obj_id,
            "$or": [
                {"userId": user_obj_id},
                {"userId": str(user_obj_id)},
                {
                    "collaborators": {
                        "$elemMatch": {
                            "userId": {"$in": [user_obj_id, str(user_obj_id)]},
                            "status": {"$in": ["accepted", None]}
                        }
                    }
                }
            ]
        }
        
        notebook = notebooks_collection.find_one(query)
        if notebook:
            return session

    raise HTTPException(status_code=403, detail="Access denied to this session")


class Message(BaseModel):
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    timestamp: str
    user_id: Optional[str] = None


class SessionCreate(BaseModel):
    title: str = Field(default="New Chat Session")
    notebook_id: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    user_id: str
    notebook_id: Optional[str] = None
    title: str
    last_message: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class SessionsListResponse(BaseModel):
    sessions: List[SessionResponse]
    total: int
    limit_reached: bool
    max_sessions: int


@router.get(
    "",
    response_model=SessionsListResponse,
    responses={401: {"model": ErrorResponse}},
    summary="Get user's chat sessions",
    description="Retrieve all chat sessions for the authenticated user"
)
async def get_sessions(current_user: dict = Depends(get_current_user)):
    """
    Get all chat sessions for a user with rate limiting info.
    """
    user_id = current_user["user_id"]  # Extract user_id from dict
    try:
        # Get user role from user_id (you might want to fetch this from backend)
        # For now, assume 'student' role means free user
        max_sessions = 3  # Free users limited to 3 sessions
        
        # Fetch sessions from MongoDB
        sessions_cursor = sessions_collection.find(
            {"user_id": user_id}
        ).sort("created_at", DESCENDING)
        
        sessions = []
        for session_doc in sessions_cursor:
            # Get message count
            msg_count = messages_collection.count_documents({"session_id": str(session_doc["_id"])})
            
            sessions.append(SessionResponse(
                id=str(session_doc["_id"]),
                user_id=user_id,
                notebook_id=session_doc.get("notebook_id"),
                title=session_doc.get("title", "New Chat Session"),
                last_message=session_doc.get("last_message", "Start chatting..."),
                message_count=msg_count,
                created_at=session_doc.get("created_at", datetime.utcnow()),
                updated_at=session_doc.get("updated_at", datetime.utcnow())
            ))
        
        return SessionsListResponse(
            sessions=sessions,
            total=len(sessions),
            limit_reached=len(sessions) >= max_sessions,
            max_sessions=max_sessions
        )
        
    except Exception as e:
        logger.exception(f"Failed to fetch sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch sessions")


@router.post(
    "",
    response_model=SessionResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse}
    },
    summary="Create new chat session",
    description="Create a new chat session (rate limited for free users)"
)
async def create_session(
    session_data: SessionCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new chat session with rate limiting.
    """
    user_id = current_user["user_id"]
    try:
        # Check session limit for free users
        max_sessions = 50  # Increased limit for development
        existing_count = sessions_collection.count_documents({"user_id": user_id})
        
        if existing_count >= max_sessions:
            raise HTTPException(
                status_code=403,
                detail=f"Session limit reached ({max_sessions} sessions). Please delete an old session."
            )
        
        # Create new session
        new_session = {
            "user_id": user_id,
            "notebook_id": session_data.notebook_id,
            "title": session_data.title,
            "last_message": "Start chatting...",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = sessions_collection.insert_one(new_session)
        session_id = str(result.inserted_id)
        
        logger.info(f"Created new session {session_id} for user {user_id}")
        
        return SessionResponse(
            id=session_id,
            user_id=user_id,
            notebook_id=new_session["notebook_id"],
            title=new_session["title"],
            last_message=new_session["last_message"],
            message_count=0,
            created_at=new_session["created_at"],
            updated_at=new_session["updated_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.get(
    "/{session_id}/messages",
    response_model=List[Message],
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse}
    },
    summary="Get session messages",
    description="Retrieve all messages for a specific session"
)
async def get_session_messages(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get all messages for a specific session.
    """
    user_id = current_user["user_id"]
    try:
        # Verify session access
        session = verify_session_access(session_id, user_id)
        
        # Fetch messages
        logger.info(f"[PERSISTENCE DEBUG] Fetching messages for session_id: {session_id}")
        messages_cursor = messages_collection.find(
            {"session_id": session_id}
        ).sort("timestamp", 1)
        
        messages = []
        for msg_doc in messages_cursor:
            messages.append(Message(
                role=msg_doc.get("role", msg_doc.get("sender", "assistant")),  # Support both old and new format
                content=msg_doc.get("content", msg_doc.get("text", "")),
                timestamp=msg_doc["timestamp"],
                user_id=msg_doc.get("user_id")
            ))
        
        logger.info(f"[PERSISTENCE DEBUG] Found {len(messages)} messages for session {session_id}")
        return messages
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to fetch messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch messages")


@router.post(
    "/{session_id}/messages",
    response_model=Message,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse}
    },
    summary="Save message to session",
    description="Save a new message to a chat session"
)
async def save_message(
    session_id: str,
    message: Message,
    current_user: dict = Depends(get_current_user)
):
    """
    Save a message to a session.
    """
    user_id = current_user["user_id"]
    try:
        # Verify session access
        session = verify_session_access(session_id, user_id)
        session_obj_id = ObjectId(session_id)
        
        # Save message
        message_doc = {
            "session_id": session_id,
            "user_id": user_id,
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp,
            "created_at": datetime.utcnow()
        }
        
        messages_collection.insert_one(message_doc)
        
        # Update session's last message
        sessions_collection.update_one(
            {"_id": session_obj_id},
            {
                "$set": {
                    "last_message": message.content[:50],
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return message
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to save message: {e}")
        raise HTTPException(status_code=500, detail="Failed to save message")


@router.delete(
    "/{session_id}",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse}
    },
    summary="Delete chat session",
    description="Delete a chat session and all its messages"
)
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a session and all its messages.
    """
    user_id = current_user["user_id"]
    try:
        # Verify session access
        session = verify_session_access(session_id, user_id)
        session_obj_id = ObjectId(session_id)
        result = sessions_collection.delete_one({"_id": session_obj_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete all messages in this session
        messages_collection.delete_many({"session_id": session_id})

        # Best-effort: clear any cached SessionTaskState for this session
        try:
            composite_id = f"{user_id}:{session_id}"
            await delete_session_state(composite_id)
        except Exception as e:
            logger.warning(f"Failed to delete Redis session state for {session_id}: {e}")
        
        logger.info(f"Deleted session {session_id} for user {user_id}")
        
        return {"message": "Session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


@router.delete(
    "/{session_id}/messages",
    summary="Clear session messages",
    description="Delete all messages in a session without deleting the session itself",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse}
    }
)
async def clear_session_messages(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Clear all messages in a session.
    """
    user_id = current_user["user_id"]
    try:
        # Verify session access
        session = verify_session_access(session_id, user_id)
        session_obj_id = ObjectId(session_id)
        
        # Delete all messages in this session
        result = messages_collection.delete_many({"session_id": session_id})
        
        logger.info(f"Cleared {result.deleted_count} messages from session {session_id} for user {user_id}")
        
        return {"message": "Messages cleared successfully", "deleted_count": result.deleted_count}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to clear messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear messages")


@router.get(
    "/{session_id}/chat/stream",
    summary="Streaming chat for session (GET)",
    description="Send a message via query params and get streaming AI response for a specific session",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def session_chat_stream_get(
    session_id: str,
    message: str,
    use_rag: bool = False,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    """
    Streaming chat endpoint using GET with query parameters (for SSE compatibility).
    Automatically saves messages to the session.
    
    Args:
        session_id: The session ID
        message: The user's message (query param)
        use_rag: Whether to use RAG retrieval (query param)
        
    Returns:
        StreamingResponse with Server-Sent Events
    """
    # Create a ChatRequest object from query params
    request = ChatRequest(message=message, use_rag=use_rag)
    
    # Reuse the POST handler logic
    return await session_chat_stream(session_id, request, user_id)


@router.post(
    "/{session_id}/chat/stream",
    summary="Streaming chat for session (POST)",
    description="Send a message and get streaming AI response for a specific session",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def session_chat_stream(
    session_id: str,
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    user_id = current_user["user_id"]
    """
    Streaming chat endpoint for a specific session using SSE.
    Automatically saves messages to the session.
    
    Returns:
        StreamingResponse with Server-Sent Events
    """
    try:
        # Verify session access
        session = verify_session_access(session_id, user_id)
        session_obj_id = ObjectId(session_id)
        
        # Get notebook context for agent
        notebook_id = session.get("notebook_id")
        
        logger.info(f"Streaming chat request for session={session_id}, user={user_id}, notebook={notebook_id}")

        # PHASE 6 — STREAM BUFFER (optional safety)
        # Generate a request-scoped stream ID and best-effort Redis client.
        from uuid import uuid4
        stream_id = str(uuid4())
        redis = None
        try:
            redis = await get_redis()
        except Exception as e:
            logger.warning(f"Redis stream buffer: failed to acquire client: {e}")
            redis = None

        # Resolve how this request should be interpreted by the agent.
        # For normal chat: agent_message == user_message, role == 'user'.
        # For artifact_request: agent_message is the internal prompt built on the server,
        # while user_message is a short, user-facing system message.
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
            # Short, user-visible system messages mapped per artifact type.
            artifact_labels = {
                "quiz": "Quiz generation requested",
                "flashcards": "Preparing flashcards",
                "mindmap": "Creating concept map",
                "summary": "Generating summary",
            }
            user_message = artifact_labels.get(artifact_type, "Generating study artifact")
            user_role = "system"
        else:
            if not request.message or not request.message.strip():
                raise HTTPException(status_code=400, detail="message is required for chat requests")
            internal_prompt = request.message
            user_message = request.message
            user_role = "user"
        
        async def event_generator():
            """Generate SSE events that stream in real-time from agent."""
            full_response = ""
            max_tool_output_chars = 4000
            
            try:
                # Stream from new agent architecture
                async for event in run_agent(
                    user_id=user_id,
                    session_id=session_id,
                    message=internal_prompt,
                    notebook_id=notebook_id,
                    use_rag=bool(request.use_rag) or bool(request.source_ids),
                    source_ids=request.source_ids,
                    verified_mode=request.verified_mode,
                    token=credentials.credentials if credentials else None,
                    stream=True
                ):
                    if not event:
                        continue
                    
                    event_type = event.get("type")
                    
                    # Handle token events (streaming text)
                    if event_type == "token":
                        content = event.get("content", "")
                        full_response += content
                        payload = json.dumps(
                            {"type": "token", "content": content},
                            ensure_ascii=False,
                        )

                        # Optional: append token to Redis buffer for potential resume.
                        if redis is not None:
                            try:
                                stream_key = f"stream:{stream_id}"
                                await redis.append(stream_key, content)
                                # Ensure a short TTL so buffers don't accumulate
                                await redis.expire(stream_key, 5 * 60)
                            except Exception as e:
                                logger.warning(f"Redis stream buffer error: {e}")

                        yield f"data: {payload}\n\n"
                    
                    # Handle tool execution events
                    elif event_type == "tool_start":
                        tool_info = json.dumps({
                            "type": "tool_start",
                            "tool": event.get("tool"),
                            "inputs": event.get("inputs")
                        })
                        yield f"data: {tool_info}\n\n"
                    
                    elif event_type == "tool_end":
                        output = event.get("output")
                        output_text = "" if output is None else str(output)
                        output_preview = output_text[:max_tool_output_chars]
                        truncated = len(output_text) > max_tool_output_chars

                        tool_info = json.dumps({
                            "type": "tool_end",
                            "tool": event.get("tool"),
                            "output": output_preview,
                            "output_truncated": truncated,
                            "output_len": len(output_text),
                        })
                        yield f"data: {tool_info}\n\n"
                    
                    # Handle completion
                    elif event_type == "complete":
                        full_response = event.get("message", full_response)
                    
                    # Handle errors
                    elif event_type == "error":
                        error_payload = json.dumps({
                            "type": "error",
                            "message": event.get("message"),
                            "details": event.get("details")
                        })
                        yield f"data: {error_payload}\n\n"
                        full_response = event.get("message", "An error occurred.")
                
                # Send completion event
                yield f"event: done\ndata: \n\n"
                
                # Save messages after streaming completes
                try:
                    # Save user-visible message (never the internal artifact prompt)
                    logger.info(f"[PERSISTENCE DEBUG] Saving messages for session_id: {session_id}")
                    user_msg = {
                        "session_id": session_id,
                        "user_id": user_id,
                        "role": user_role,
                        "content": user_message,
                        "timestamp": datetime.utcnow().isoformat(),
                        "created_at": datetime.utcnow()
                    }
                    messages_collection.insert_one(user_msg)
                    logger.info(f"[PERSISTENCE DEBUG] Saved user message: {user_message[:50]}...")
                    
                    # Save assistant response
                    assistant_msg = {
                        "session_id": session_id,
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.utcnow().isoformat(),
                        "created_at": datetime.utcnow()
                    }
                    messages_collection.insert_one(assistant_msg)
                    logger.info(f"[PERSISTENCE DEBUG] Saved assistant response ({len(full_response)} chars)")
                    
                    # Update session's last message
                    sessions_collection.update_one(
                        {"_id": session_obj_id},
                        {
                            "$set": {
                                "last_message": request.message[:50],
                                "updated_at": datetime.utcnow()
                            }
                        }
                    )
                    
                    logger.info(f"Saved messages for session {session_id}")
                except Exception as save_error:
                    logger.error(f"Failed to save messages: {save_error}")
                    
            except Exception as e:
                logger.exception(f"Error in agent execution: {e}")
                yield f"data: ❌ Error: {str(e)}\n\n"
                yield f"event: done\ndata: \n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Stream-Id": stream_id,
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Streaming chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stream chat response: {str(e)}"
        )
