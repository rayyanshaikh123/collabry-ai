"""
Chat Sessions Management
Handles session creation, retrieval, and rate limiting
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from server.deps import get_current_user
from server.schemas import ErrorResponse, ChatRequest
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from config import CONFIG
from core.agent import create_agent
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai/sessions", tags=["sessions"])

# MongoDB connection
mongo_client = MongoClient(CONFIG["mongo_uri"])
db = mongo_client[CONFIG["mongo_db"]]
sessions_collection = db["chat_sessions"]
messages_collection = db["chat_messages"]

# Create indexes
sessions_collection.create_index([("user_id", 1), ("created_at", DESCENDING)])
messages_collection.create_index([("session_id", 1), ("timestamp", 1)])


class Message(BaseModel):
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    timestamp: str


class SessionCreate(BaseModel):
    title: str = Field(default="New Chat Session")


class SessionResponse(BaseModel):
    id: str
    user_id: str
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
async def get_sessions(user_id: str = Depends(get_current_user)):
    """
    Get all chat sessions for a user with rate limiting info.
    """
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
    user_id: str = Depends(get_current_user)
):
    """
    Create a new chat session with rate limiting.
    """
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
    user_id: str = Depends(get_current_user)
):
    """
    Get all messages for a specific session.
    """
    try:
        # Verify session belongs to user
        try:
            session_obj_id = ObjectId(session_id)
        except:
            raise HTTPException(status_code=404, detail="Invalid session ID")
            
        session = sessions_collection.find_one({"_id": session_obj_id, "user_id": user_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Fetch messages
        messages_cursor = messages_collection.find(
            {"session_id": session_id}
        ).sort("timestamp", 1)
        
        messages = []
        for msg_doc in messages_cursor:
            messages.append(Message(
                role=msg_doc.get("role", msg_doc.get("sender", "assistant")),  # Support both old and new format
                content=msg_doc.get("content", msg_doc.get("text", "")),
                timestamp=msg_doc["timestamp"]
            ))
        
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
    user_id: str = Depends(get_current_user)
):
    """
    Save a message to a session.
    """
    try:
        # Verify session belongs to user
        try:
            session_obj_id = ObjectId(session_id)
        except:
            raise HTTPException(status_code=404, detail="Invalid session ID")
            
        session = sessions_collection.find_one({"_id": session_obj_id, "user_id": user_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Save message
        message_doc = {
            "session_id": session_id,
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
    user_id: str = Depends(get_current_user)
):
    """
    Delete a session and all its messages.
    """
    try:
        # Verify session belongs to user
        try:
            session_obj_id = ObjectId(session_id)
        except:
            raise HTTPException(status_code=404, detail="Invalid session ID")
            
        result = sessions_collection.delete_one({"_id": session_obj_id, "user_id": user_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete all messages in this session
        messages_collection.delete_many({"session_id": session_id})
        
        logger.info(f"Deleted session {session_id} for user {user_id}")
        
        return {"message": "Session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


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
    user_id: str = Depends(get_current_user)
):
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
    user_id: str = Depends(get_current_user)
):
    """
    Streaming chat endpoint for a specific session using SSE.
    Automatically saves messages to the session.
    
    Returns:
        StreamingResponse with Server-Sent Events
    """
    try:
        # Verify session exists and belongs to user
        try:
            session_obj_id = ObjectId(session_id)
        except:
            raise HTTPException(status_code=404, detail="Invalid session ID")
            
        session = sessions_collection.find_one({"_id": session_obj_id, "user_id": user_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"Streaming chat request for session={session_id}, user={user_id}")
        
        # Create user-isolated agent
        agent, _, _, _ = create_agent(
            user_id=user_id,
            session_id=session_id,
            config=CONFIG
        )
        
        async def event_generator():
            """Generate SSE events that stream in real-time from agent."""
            full_response = ""
            
            try:
                # Execute agent with streaming callback
                def stream_chunk(chunk: str):
                    """Callback for each chunk from agent."""
                    nonlocal full_response
                    if chunk.strip():
                        full_response += chunk
                        return chunk
                    return None
                
                # Execute agent - this will call stream_chunk for each piece of output
                # Pass source_ids to filter RAG by selected sources only
                agent.handle_user_input_stream(
                    request.message, 
                    stream_chunk,
                    source_ids=request.source_ids
                )
                
                # Now stream the collected response character by character to preserve markdown
                if full_response:
                    import asyncio
                    # Stream character by character to preserve newlines and markdown
                    for char in full_response:
                        # Escape special characters for SSE
                        if char == '\n':
                            # Send newline as literal \n for markdown to process
                            yield f"data: \\n\n\n"
                        else:
                            yield f"data: {char}\n\n"
                        await asyncio.sleep(0.008)  # 8ms delay for smooth streaming
                
                # Send completion event
                yield f"event: done\ndata: \n\n"
                
                # Save messages after streaming completes
                try:
                    # Save user message
                    user_msg = {
                        "session_id": session_id,
                        "role": "user",
                        "content": request.message,
                        "timestamp": datetime.utcnow().isoformat(),
                        "created_at": datetime.utcnow()
                    }
                    messages_collection.insert_one(user_msg)
                    
                    # Save assistant response
                    assistant_msg = {
                        "session_id": session_id,
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.utcnow().isoformat(),
                        "created_at": datetime.utcnow()
                    }
                    messages_collection.insert_one(assistant_msg)
                    
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
