# core/mongo_store.py
"""
MongoDB persistence adapter for conversational memory.

Provides append-only storage for thread-scoped conversation history.
Schema: { user_id, thread_id, role, content, metadata, timestamp }

Thread safety: Uses connection pooling; safe for concurrent writes.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Any, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
from gridfs import GridFS
import io
import tarfile

logger = logging.getLogger(__name__)


class MongoMemoryStore:
    """MongoDB-backed memory persistence layer."""

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str = "conversations",
        user_id: str = "default_user",
    ):
        """
        Initialize MongoDB connection.

        Args:
            mongo_uri: MongoDB connection string (e.g., "mongodb://localhost:27017")
            db_name: Database name
            collection_name: Collection for conversation history
            user_id: Default user identifier (for multi-user support)
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.user_id = user_id

        self._client: Optional[MongoClient] = None
        self._db = None
        self._collection = None
        self._connected = False

        self._connect()

    def _connect(self):
        """Establish MongoDB connection and create indexes."""
        try:
            self._client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
            )
            # Test connection
            self._client.admin.command("ping")
            
            self._db = self._client[self.db_name]
            self._collection = self._db[self.collection_name]

            # GridFS for storing binary artifacts (FAISS index archives, etc.)
            try:
                self._fs = GridFS(self._db)
            except Exception:
                self._fs = None

            # Create indexes for efficient queries
            self._collection.create_index(
                [("user_id", ASCENDING), ("thread_id", ASCENDING), ("timestamp", ASCENDING)]
            )
            self._collection.create_index([("timestamp", DESCENDING)])

            self._connected = True
            logger.info(f"✓ MongoDB connected: {self.db_name}.{self.collection_name}")

        except (ConnectionFailure, OperationFailure) as e:
            logger.warning(f"MongoDB connection failed: {e}. Using fallback in-memory mode.")
            self._connected = False

    def is_connected(self) -> bool:
        """Check if MongoDB connection is active."""
        return self._connected and self._client is not None

    def append_turn(
        self,
        thread_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Append a conversational turn to MongoDB.

        Args:
            thread_id: Conversation thread identifier
            user_message: User's input message
            assistant_message: Assistant's response
            metadata: Optional metadata dict
            user_id: Override default user_id

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            logger.debug("MongoDB not connected, skipping append")
            return False

        try:
            timestamp = time.time()
            uid = user_id or self.user_id

            # Store as a single turn document with both messages
            document = {
                "user_id": uid,
                "thread_id": thread_id,
                "timestamp": timestamp,
                "user": user_message,
                "assistant": assistant_message,
                "metadata": metadata or {},
            }

            self._collection.insert_one(document)
            logger.debug(f"Stored turn for thread={thread_id}, user={uid}")
            return True

        except Exception as e:
            logger.error(f"Failed to append turn to MongoDB: {e}")
            return False

    def load_thread_history(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Load conversation history for a specific thread.

        Args:
            thread_id: Thread identifier
            user_id: Override default user_id
            limit: Maximum number of recent turns to retrieve

        Returns:
            List of turn dicts: [{timestamp, user, assistant, metadata}, ...]
        """
        if not self.is_connected():
            logger.debug("MongoDB not connected, returning empty history")
            return []

        try:
            uid = user_id or self.user_id
            
            cursor = self._collection.find(
                {"user_id": uid, "thread_id": thread_id}
            ).sort("timestamp", ASCENDING).limit(limit)

            history = []
            for doc in cursor:
                history.append({
                    "timestamp": doc.get("timestamp", 0),
                    "user": doc.get("user", ""),
                    "assistant": doc.get("assistant", ""),
                    "metadata": doc.get("metadata", {}),
                })

            logger.debug(f"Loaded {len(history)} turns for thread={thread_id}")
            return history

        except Exception as e:
            logger.error(f"Failed to load thread history: {e}")
            return []

    def get_all_threads(self, user_id: Optional[str] = None) -> List[str]:
        """
        Get list of all thread IDs for a user.

        Args:
            user_id: Override default user_id

        Returns:
            List of thread_id strings
        """
        if not self.is_connected():
            return []

        try:
            uid = user_id or self.user_id
            thread_ids = self._collection.distinct("thread_id", {"user_id": uid})
            return sorted(thread_ids)

        except Exception as e:
            logger.error(f"Failed to get thread list: {e}")
            return []

    def delete_thread(self, thread_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete all history for a specific thread.

        Args:
            thread_id: Thread to delete
            user_id: Override default user_id

        Returns:
            True if successful
        """
        if not self.is_connected():
            return False

        try:
            uid = user_id or self.user_id
            result = self._collection.delete_many({"user_id": uid, "thread_id": thread_id})
            logger.info(f"Deleted {result.deleted_count} turns for thread={thread_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete thread: {e}")
            return False

    def close(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._connected = False
            logger.info("MongoDB connection closed")

    # -----------------------------
    # GridFS helpers for FAISS index
    # -----------------------------
    def save_faiss_index_archive(self, name: str, archive_bytes: bytes) -> bool:
        """
        Save a FAISS index archive (tar.gz bytes) into GridFS under filename `name`.
        Overwrites existing files with the same filename.
        """
        if not self.is_connected() or not getattr(self, '_fs', None):
            logger.debug("GridFS not available, cannot save FAISS archive")
            return False

        try:
            # Remove existing files with same filename
            for f in self._db.fs.files.find({'filename': name}):
                self._fs.delete(f['_id'])

            self._fs.put(archive_bytes, filename=name)
            logger.info(f"✓ Saved FAISS archive to GridFS: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save FAISS archive to GridFS: {e}")
            return False

    def load_faiss_index_archive(self, name: str) -> Optional[bytes]:
        """
        Load a FAISS archive by filename from GridFS. Returns bytes or None.
        """
        if not self.is_connected() or not getattr(self, '_fs', None):
            logger.debug("GridFS not available, cannot load FAISS archive")
            return None

        try:
            # Find latest file with this filename
            files = list(self._db.fs.files.find({'filename': name}).sort('uploadDate', -1).limit(1))
            if not files:
                return None
            grid_id = files[0]['_id']
            data = self._fs.get(grid_id).read()
            logger.info(f"✓ Loaded FAISS archive from GridFS: {name}")
            return data
        except Exception as e:
            logger.error(f"Failed to load FAISS archive from GridFS: {e}")
            return None
