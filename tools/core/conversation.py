"""
MongoDB Conversation History Manager.

Handles CRUD operations for multi-user, multi-session chat history.
No LangChain memory classes - just plain MongoDB queries.

Schema:
{
    user_id: str,
    session_id: str,
    notebook_id: str | null,
    messages: [
        {role: "user"|"assistant", content: str, timestamp: datetime},
        ...
    ],
    metadata: dict,
    created_at: datetime,
    updated_at: datetime
}
"""

import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database


class ConversationManager:
    """Manages conversation history in MongoDB."""
    
    def __init__(self, mongodb_uri: Optional[str] = None, db_name: Optional[str] = None):
        """
        Initialize conversation manager.
        
        Args:
            mongodb_uri: MongoDB connection URI (defaults to env MONGODB_URI)
            db_name: Database name (defaults to env MONGODB_DB or 'study_assistant')
        """
        # Accept common env var aliases used across deployments.
        env_uri = (
            (os.getenv("MONGODB_URI") or "").strip()
            or (os.getenv("MONGO_URI") or "").strip()
            or (os.getenv("MONGO_URL") or "").strip()
            or (os.getenv("MONGODB_URL") or "").strip()
            or (os.getenv("DATABASE_URL") or "").strip()
        )
        if env_uri and not env_uri.lower().startswith("mongodb"):
            env_uri = ""

        self.uri = mongodb_uri or env_uri or "mongodb://localhost:27017"
        self.db_name = db_name or (os.getenv("MONGODB_DB") or os.getenv("MONGO_DB") or "study_assistant")
        
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.conversations: Optional[Collection] = None
        
        self._connect()
    
    def _connect(self):
        """Establish MongoDB connection and create indexes."""
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.conversations = self.db["conversations"]
        
        # Create indexes for efficient queries
        self.conversations.create_index([("user_id", ASCENDING), ("session_id", ASCENDING)])
        self.conversations.create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)])
        self.conversations.create_index([("notebook_id", ASCENDING)])
    
    def get_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum number of messages to retrieve (most recent)
        
        Returns:
            List of message dicts: [{role, content, sender_id?, sender_name?}, ...]
        """
        conversation = self.conversations.find_one({
            "user_id": user_id,
            "session_id": session_id
        })
        
        if not conversation or "messages" not in conversation:
            return []
        
        messages = conversation["messages"]
        
        # Return last N messages
        if limit and len(messages) > limit:
            messages = messages[-limit:]
        
        # Return in LangChain format with optional sender attribution
        result = []
        for msg in messages:
            entry = {"role": msg["role"], "content": msg["content"]}
            if "sender_id" in msg:
                entry["sender_id"] = msg["sender_id"]
            if "sender_name" in msg:
                entry["sender_name"] = msg["sender_name"]
            result.append(entry)
        return result
    
    def save_turn(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        notebook_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sender_id: Optional[str] = None,
        sender_name: Optional[str] = None
    ):
        """
        Save a conversation turn (user message + assistant response).
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            user_message: User's message
            assistant_message: Assistant's response
            notebook_id: Optional notebook context
            metadata: Optional metadata (tool_calls, sources, etc.)
            sender_id: Optional sender user ID (for collaborative sessions)
            sender_name: Optional sender display name (for collaborative sessions)
        """
        now = datetime.utcnow()
        
        # Create turn messages with optional sender attribution
        user_msg = {
            "role": "user",
            "content": user_message,
            "timestamp": now,
            "sender_id": sender_id or user_id,
            "sender_name": sender_name or "Unknown",
        }
        
        assistant_msg = {
            "role": "assistant",
            "content": assistant_message,
            "timestamp": now,
            "metadata": metadata or {}
        }
        
        turn_messages = [user_msg, assistant_msg]
        
        # Upsert conversation document
        self.conversations.update_one(
            {"user_id": user_id, "session_id": session_id},
            {
                "$push": {"messages": {"$each": turn_messages}},
                "$set": {
                    "updated_at": now,
                    "notebook_id": notebook_id
                },
                "$setOnInsert": {
                    "created_at": now
                }
            },
            upsert=True
        )
    
    def create_session(
        self,
        user_id: str,
        session_id: str,
        notebook_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new conversation session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            notebook_id: Optional notebook context
        
        Returns:
            Created session document
        """
        now = datetime.utcnow()
        
        session = {
            "user_id": user_id,
            "session_id": session_id,
            "notebook_id": notebook_id,
            "messages": [],
            "created_at": now,
            "updated_at": now
        }
        
        self.conversations.insert_one(session)
        return session
    
    def list_sessions(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        List user's conversation sessions.
        
        Args:
            user_id: User identifier
            limit: Maximum number of sessions to return
        
        Returns:
            List of session summaries with metadata
        """
        sessions = self.conversations.find(
            {"user_id": user_id},
            {
                "session_id": 1,
                "notebook_id": 1,
                "created_at": 1,
                "updated_at": 1,
                "messages": {"$slice": -1}  # Get last message for preview
            }
        ).sort("updated_at", DESCENDING).limit(limit)
        
        result = []
        for session in sessions:
            last_message = session.get("messages", [{}])[-1] if session.get("messages") else {}
            
            result.append({
                "session_id": session["session_id"],
                "notebook_id": session.get("notebook_id"),
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "last_message_preview": last_message.get("content", "")[:100]
            })
        
        return result
    
    def delete_session(self, user_id: str, session_id: str) -> bool:
        """
        Delete a conversation session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
        
        Returns:
            True if session was deleted, False if not found
        """
        result = self.conversations.delete_one({
            "user_id": user_id,
            "session_id": session_id
        })
        return result.deleted_count > 0
    
    def get_session_metadata(
        self,
        user_id: str,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get session metadata without full message history.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
        
        Returns:
            Session metadata or None if not found
        """
        session = self.conversations.find_one(
            {"user_id": user_id, "session_id": session_id},
            {"messages": 0}  # Exclude messages
        )
        return session
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()


# Singleton instance
_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get or create conversation manager singleton."""
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager


def reset_manager():
    """Reset manager singleton. Useful for testing."""
    global _manager
    if _manager:
        _manager.close()
    _manager = None
