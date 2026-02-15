from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import os
import threading
from contextlib import contextmanager
import logging

from core.redis_client import get_redis

logger = logging.getLogger(__name__)

SESSION_STATE_TTL_SECONDS = int(os.getenv("SESSION_STATE_TTL", "3600"))


class SessionTaskState:
    """
    SECURITY FIX - Phase 2: Enhanced with proper isolation and locking.
    Tracks the current task state for a user session to handle follow-up mutations.
    """
    
    def __init__(self, session_id: str, notebook_id: Optional[str] = None):
        self.session_id = session_id
        self.notebook_id = notebook_id
        self.active_users: Dict[str, datetime] = {}  # user_id -> last_active
        self.active_task: Optional[str] = None  # e.g., "quiz", "flashcards", "search", "web", "mindmap"
        self.last_tool: Optional[str] = None     # e.g., "generate_quiz", "search_web"
        self.task_params: Dict[str, Any] = {}    # Current parameters (topic, num, etc.)
        self.last_update: datetime = datetime.now()
        self.history: List[Dict[str, str]] = []  # Brief history for current task context
        
        # SECURITY FIX - Phase 2: Add artifact storage and thread safety
        self.artifacts: Dict[str, Any] = {}  # Store generated content for follow-up questions
        self.artifact_history: List[Dict[str, Any]] = []  # Timeline of artifacts
        self._lock = threading.RLock()  # Thread safety for concurrent requests
        # Track the user's preferred language for this session (auto-detected).
        self.session_language: Optional[str] = None
    
    def register_user(self, user_id: str):
        """Track a user as active in this session."""
        with self._lock:
            self.active_users[user_id] = datetime.now()
    
    def get_active_user_count(self) -> int:
        """Return count of active users in this session."""
        with self._lock:
            return len(self.active_users)
    
    def set_task(self, task: str, tool: str, params: Dict[str, Any]):
        """
        SECURITY FIX - Phase 2: Properly clear previous state to prevent bleed.
        Set a new active task and reset parameters.
        """
        with self._lock:
            # Clear ALL previous state to prevent parameter bleeding 
            old_task = self.active_task
            self.active_task = task
            self.last_tool = tool
            self.task_params = params.copy()  # Fresh copy, not update
            self.last_update = datetime.now()
            
            # Clear stale artifacts if task type changes
            if old_task and old_task != task:
                logger.info(f"🧹 Clearing artifacts from {old_task} -> {task}")
                self.artifacts.clear()
                
            logger.info(f"🔄 Session {self.session_id}: NEW task context: {task} ({tool}) - Previous state cleared")
    
    def update_params(self, updates: Dict[str, Any]):
        """Update existing task parameters safely."""
        with self._lock:
            self.task_params.update(updates)
            self.last_update = datetime.now()
            logger.info(f"🔄 Session {self.session_id}: Updated params for {self.active_task}: {updates}")
    
    def store_artifact(self, artifact_type: str, content: Any, metadata: Dict[str, Any] = None):
        """SECURITY FIX - Phase 2: Store generated artifacts for follow-up questions."""
        with self._lock:
            self.artifacts = {
                'type': artifact_type,
                'content': content,
                'metadata': metadata or {},
                'task_type': self.active_task,
                'created_at': datetime.now()
            }
            
            # Keep history for debugging
            self.artifact_history.append({
                'type': artifact_type,
                'task_type': self.active_task,
                'created_at': datetime.now(),
                'content_preview': str(content)[:100] + "..." if len(str(content)) > 100 else str(content)
            })
            
            logger.info(f"📦 Stored {artifact_type} artifact for session {self.session_id}")
            
    def get_artifact_context(self) -> Optional[Dict[str, Any]]:
        """SECURITY FIX - Phase 2: Retrieve current artifact for context-aware responses."""
        with self._lock:
            if self.artifacts:
                return self.artifacts.copy()
            return None
    
    def clear(self):
        """Reset the session task state."""
        with self._lock:
            self.active_task = None
            self.last_tool = None
            self.task_params = {}
            self.last_update = datetime.now()
            self.artifacts.clear()
            logger.info(f"🧹 Cleared session state for {self.session_id}")

    # --------- (De)serialization helpers for Redis-backed persistence ---------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the minimal state needed for follow-up mutations.
        """
        with self._lock:
            return {
                "session_id": self.session_id,
                "notebook_id": self.notebook_id,
                "active_task": self.active_task,
                "last_tool": self.last_tool,
                "task_params": self.task_params,
                "artifacts": self.artifacts,
                "session_language": self.session_language,
                "last_update": self.last_update.isoformat(),
            }

    def apply_dict(self, data: Dict[str, Any]) -> None:
        """
        Hydrate this instance from a previously serialized dict.
        """
        if not data:
            return
        with self._lock:
            self.session_id = data.get("session_id", self.session_id)
            self.notebook_id = data.get("notebook_id", self.notebook_id)
            self.active_task = data.get("active_task", self.active_task)
            self.last_tool = data.get("last_tool", self.last_tool)
            self.task_params = data.get("task_params", self.task_params) or {}
            self.artifacts = data.get("artifacts", self.artifacts) or {}
            self.session_language = data.get("session_language", self.session_language)
            try:
                ts = data.get("last_update")
                if ts:
                    self.last_update = datetime.fromisoformat(ts)
            except Exception:
                # Fallback to "now" if parsing fails
                self.last_update = datetime.now()

# SECURITY FIX - Phase 2: Enhanced global state management with proper isolation
_session_states: Dict[str, SessionTaskState] = {}
_session_locks: Dict[str, threading.RLock] = {}

@contextmanager
def session_state_lock(session_id: str):
    """SECURITY FIX - Phase 2: Request-scoped session locking."""
    if session_id not in _session_locks:
        _session_locks[session_id] = threading.RLock()
    
    with _session_locks[session_id]:
        yield

def get_session_state(session_id: str, user_id: str = None) -> SessionTaskState:
    """
    SECURITY FIX - Phase 2: Enhanced isolation with user validation.
    Get or create session task state for a session ID with proper user isolation.
    """
    # Create composite key for proper user isolation
    if user_id:
        composite_key = f"{user_id}:{session_id}"
    else:
        composite_key = session_id
        
    with session_state_lock(composite_key):
        if composite_key not in _session_states:
            _session_states[composite_key] = SessionTaskState(session_id)
            logger.info(f"🆕 Created new session state for {composite_key}")
        return _session_states[composite_key]


# --------- Redis-backed persistence helpers (with in-memory fallback) ---------

def _session_state_key(composite_id: str) -> str:
    """
    Build the Redis key for a session task state.

    We deliberately accept a composite identifier (e.g. "user_id:session_id")
    to maintain the same isolation semantics as the in-memory store.
    """
    return f"session:{composite_id}"


async def save_session_state(session_composite_id: str, state_dict: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
    """
    Persist SessionTaskState into Redis as JSON.

    Failure-safe: if Redis is unavailable, this is a no-op and the system
    continues using the in-memory state.
    """
    try:
        redis = await get_redis()
        if not redis:
            return
        key = _session_state_key(session_composite_id)
        ttl = ttl_seconds or SESSION_STATE_TTL_SECONDS
        payload = json.dumps(state_dict, ensure_ascii=False, default=str)
        await redis.set(key, payload, ex=ttl)
    except Exception as e:
        logger.warning(f"Redis save_session_state failed for {session_composite_id}: {e}")


async def load_session_state(session_composite_id: str) -> Optional[Dict[str, Any]]:
    """
    Load SessionTaskState from Redis.

    Returns the decoded dict or None if not present or on error.
    Also refreshes TTL on access.
    """
    try:
        redis = await get_redis()
        if not redis:
            return None
        key = _session_state_key(session_composite_id)
        raw = await redis.get(key)
        if not raw:
            return None
        try:
            await redis.expire(key, SESSION_STATE_TTL_SECONDS)
        except Exception:
            # TTL refresh is best-effort
            pass
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Redis load_session_state failed for {session_composite_id}: {e}")
        return None


async def delete_session_state(session_composite_id: str) -> None:
    """
    Delete a SessionTaskState from Redis.
    """
    try:
        redis = await get_redis()
        if not redis:
            return
        key = _session_state_key(session_composite_id)
        await redis.delete(key)
    except Exception as e:
        logger.warning(f"Redis delete_session_state failed for {session_composite_id}: {e}")

