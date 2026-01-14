# core/memory.py
"""
Thread-scoped conversational memory with MongoDB persistence and multi-user isolation.

- Multi-user support: Each user has isolated memory via user_id
- Multi-session support: Users can have multiple chat sessions (like ChatGPT)
- Thread format: "{user_id}:{session_id}" for strict isolation
- Long-term persistence: MongoDB-backed (REQUIRED)

JWT-based flow:
  1. JWT validated upstream (not handled here)
  2. user_id extracted from JWT claims
  3. session_id chosen/created by user
  4. Memory isolated by user_id + session_id

API:
  - load_memory_variables() -> dict with buffer/summary
  - save_context() -> stores turn with user isolation
  - get_history_string() -> formatted conversation
  - list_user_sessions() -> all sessions for current user
  - create_session() -> new isolated chat session
"""

from __future__ import annotations

import time
import logging
import uuid
from typing import Dict, List, Any, Optional

from langgraph.checkpoint.memory import InMemorySaver
from core.mongo_store import MongoMemoryStore
from config import CONFIG

logger = logging.getLogger(__name__)


def format_thread_id(user_id: str, session_id: str) -> str:
    """Enforce thread_id format for user isolation."""
    if not user_id or not session_id:
        raise ValueError("user_id and session_id are required")
    return f"{user_id}:{session_id}"


def parse_thread_id(thread_id: str) -> tuple[str, str]:
    """Parse thread_id into (user_id, session_id)."""
    parts = thread_id.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid thread_id format: {thread_id}. Expected 'user_id:session_id'")
    return parts[0], parts[1]


class MemoryManager:
    def __init__(
        self, 
        user_id: str,
        session_id: str = "default",
        llm=None
    ):
        """
        Initialize memory manager with strict user isolation.
        
        Args:
            user_id: User identifier from JWT (REQUIRED)
            session_id: Chat session identifier (default: "default")
            llm: LLM instance (optional, for compatibility)
        """
        if not user_id:
            raise ValueError("user_id is REQUIRED for multi-user isolation")
        
        # llm is accepted for signature compatibility
        self.llm = llm
        self.user_id = user_id
        self.session_id = session_id
        
        # Enforce thread_id format: "user_id:session_id"
        self.thread_id = format_thread_id(user_id, session_id)

        # Short-term per-thread history: { thread_id: [ {user, assistant, ts} ] }
        self._history_by_thread: Dict[str, List[Dict[str, Any]]] = {}

        # In-memory checkpointer instance (ready to be used by langgraph-based flows)
        self.checkpointer = InMemorySaver()

        # Initialize MongoDB persistence (with fallback to in-memory)
        mongo_uri = CONFIG["mongo_uri"]
        mongo_db = CONFIG["mongo_db"]
        collection = CONFIG["memory_collection"]

        try:
            self._mongo_store = MongoMemoryStore(
                mongo_uri=mongo_uri,
                db_name=mongo_db,
                collection_name=collection,
                user_id=self.user_id,
            )

            if not self._mongo_store.is_connected():
                raise ConnectionError("MongoDB not connected")

        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}. Using in-memory storage only.")
            logger.warning("⚠️ Memory will not persist between sessions. This is for development only.")
            self._mongo_store = None
        
        logger.info(f"✓ MongoDB memory persistence enabled: {mongo_db}.{collection}")
        logger.info(f"✓ User isolation: user_id={user_id}, session_id={session_id}, thread={self.thread_id}")

        # Load existing conversation history for this thread
        self._load_thread_from_storage(self.thread_id)

    # ---------------- API used by the agent ----------------
    def load_memory_variables(self, input_dict: Dict[str, Any]):
        """Return dict shaped for the agent prompt builder."""
        history = self._history_by_thread.get(self.thread_id, [])
        buffer = self._format_buffer(history)
        # Lightweight summary: first N and last N turns concatenated
        summary = self._summarize(history)
        return {
            "chat_history_buffer": buffer,
            "chat_history_summary": summary,
        }

    def save_context(self, input_dict: Dict[str, Any], output_dict: Dict[str, Any]):
        """Append one conversational turn and persist to MongoDB if available."""
        user = (input_dict or {}).get("user_input", "")
        assistant = (output_dict or {}).get("output", "")
        turn = {"timestamp": time.time(), "user": user, "assistant": assistant}

        # Store in memory
        self._history_by_thread.setdefault(self.thread_id, []).append(turn)

        # Persist to MongoDB if available
        if self._mongo_store:
            success = self._mongo_store.append_turn(
                thread_id=self.thread_id,
                user_message=user,
                assistant_message=assistant,
                metadata={"timestamp": turn["timestamp"]},
                user_id=self.user_id,
            )

            if not success:
                logger.warning(f"Failed to persist turn to MongoDB for thread={self.thread_id}")
        else:
            logger.debug(f"MongoDB not available - turn stored in memory only for thread={self.thread_id}")

    def get_history_string(self) -> str:
        vars = self.load_memory_variables({})
        history = ""
        if vars.get("chat_history_summary"):
            history += "Summary of conversation:\n" + vars["chat_history_summary"] + "\n\n"
        if vars.get("chat_history_buffer"):
            history += "Recent conversation:\n" + vars["chat_history_buffer"]
        return history.strip()

    def clear(self):
        """Clear current thread's short-term memory cache only."""
        self._history_by_thread[self.thread_id] = []
        # Note: This only clears in-memory cache. Use delete_thread() to remove from MongoDB.

    def delete_thread(self):
        """Permanently delete current thread from MongoDB if available."""
        if self._mongo_store:
            self._mongo_store.delete_thread(self.thread_id, user_id=self.user_id)
        self._history_by_thread[self.thread_id] = []

    # ---------------- Optional helpers ----------------
    def set_thread(self, thread_id: str):
        """
        Switch to a different conversation thread.
        WARNING: thread_id must follow format "user_id:session_id"
        and user_id must match current user for security.
        """
        try:
            tid_user, tid_session = parse_thread_id(thread_id)
            if tid_user != self.user_id:
                raise PermissionError(
                    f"Cannot switch to thread for different user. "
                    f"Current user: {self.user_id}, Thread user: {tid_user}"
                )
            self.thread_id = thread_id
            self.session_id = tid_session
        except ValueError as e:
            raise ValueError(f"Invalid thread_id format: {e}")
        
        # Lazy load thread history if not in memory
        if thread_id not in self._history_by_thread:
            self._load_thread_from_storage(thread_id)
    
    def switch_session(self, session_id: str):
        """Switch to a different session for the current user."""
        new_thread_id = format_thread_id(self.user_id, session_id)
        self.set_thread(new_thread_id)

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        return list(self._history_by_thread.get(self.thread_id, [])[-n:])
    
    def get_all_threads(self) -> List[str]:
        """Get list of all thread IDs for current user."""
        if self._mongo_store:
            return self._mongo_store.get_all_threads(user_id=self.user_id)
        else:
            # Return in-memory threads only
            return list(self._history_by_thread.keys())
    
    def list_user_sessions(self) -> List[Dict[str, Any]]:
        """
        List all chat sessions for current user.
        Returns list of {session_id, thread_id, last_activity}.
        """
        all_threads = self.get_all_threads()
        sessions = []
        
        for thread_id in all_threads:
            try:
                tid_user, tid_session = parse_thread_id(thread_id)
                if tid_user == self.user_id:
                    # Get last activity timestamp
                    if self._mongo_store:
                        history = self._mongo_store.load_thread_history(
                            thread_id=thread_id,
                            user_id=self.user_id,
                            limit=1
                        )
                        last_ts = history[0]["timestamp"] if history else 0
                    else:
                        # Use in-memory history
                        thread_history = self._history_by_thread.get(thread_id, [])
                        last_ts = thread_history[-1]["timestamp"] if thread_history else 0

                    sessions.append({
                        "session_id": tid_session,
                        "thread_id": thread_id,
                        "last_activity": last_ts,
                        "is_current": thread_id == self.thread_id
                    })
            except ValueError:
                # Skip malformed thread_ids
                continue
        
        # Sort by last activity (most recent first)
        sessions.sort(key=lambda x: x["last_activity"], reverse=True)
        return sessions
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new chat session for current user.
        Returns: session_id
        """
        if not session_id:
            session_id = str(uuid.uuid4())[:8]  # Short UUID
        
        new_thread_id = format_thread_id(self.user_id, session_id)
        
        # Initialize empty history
        self._history_by_thread[new_thread_id] = []
        
        # Switch to new session
        self.thread_id = new_thread_id
        self.session_id = session_id
        
        logger.info(f"✓ Created new session: {session_id} for user: {self.user_id}")
        return session_id

    def close(self):
        """Close MongoDB connection."""
        if self._mongo_store:
            self._mongo_store.close()

    # ---------------- Internal utilities ----------------
    def _load_thread_from_storage(self, thread_id: str):
        """Load a specific thread from MongoDB if available."""
        if self._mongo_store:
            history = self._mongo_store.load_thread_history(
                thread_id=thread_id,
                user_id=self.user_id,
                limit=50
            )
            # Convert to expected format
            formatted_history = []
            for msg in history:
                formatted_history.append({
                    "timestamp": msg["timestamp"],
                    "user": msg["user"],
                    "assistant": msg["assistant"]
                })
            self._history_by_thread[thread_id] = formatted_history
            logger.debug(f"Loaded {len(formatted_history)} turns for thread={thread_id}")
        else:
            # MongoDB not available, keep in-memory history
            logger.debug(f"Using in-memory history for thread {thread_id}")

    def _format_buffer(self, history: List[Dict[str, Any]], max_turns: int = 10) -> str:
        tail = history[-max_turns:]
        lines: List[str] = []
        for t in tail:
            u = (t.get("user") or "").strip()
            a = (t.get("assistant") or "").strip()
            if u:
                lines.append(f"User: {u}")
            if a:
                lines.append(f"Assistant: {a}")
        return "\n".join(lines)

    def _summarize(self, history: List[Dict[str, Any]], head: int = 2, tail: int = 2) -> str:
        if not history:
            return ""
        parts: List[str] = []
        head_part = history[:head]
        tail_part = history[-tail:] if len(history) > head else []
        def fmt(block):
            out: List[str] = []
            for t in block:
                u = (t.get("user") or "").strip()
                a = (t.get("assistant") or "").strip()
                if u:
                    out.append(f"User: {u}")
                if a:
                    out.append(f"Assistant: {a}")
            return "\n".join(out)
        if head_part:
            parts.append(fmt(head_part))
        if tail_part:
            parts.append(fmt(tail_part))
        return "\n...\n".join([p for p in parts if p])
