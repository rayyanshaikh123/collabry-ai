"""
Usage tracking system for monitoring AI operations and token usage.

Tracks:
- Token usage per user
- API endpoint usage
- Success/failure rates
- Response times
- Document processing stats
"""
from pymongo import MongoClient, DESCENDING
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from config import CONFIG
import logging

logger = logging.getLogger(__name__)


class UsageTracker:
    """Track and store AI usage metrics."""
    
    def __init__(self):
        """Initialize usage tracker with MongoDB connection."""
        self.client = MongoClient(CONFIG["mongo_uri"])
        self.db = self.client[CONFIG["mongo_db"]]
        self.usage_collection = self.db["usage_logs"]
        self.daily_stats_collection = self.db["daily_stats"]
        
        # Create indexes for efficient queries
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes for performance."""
        try:
            # Index for user queries
            self.usage_collection.create_index([("user_id", 1), ("timestamp", DESCENDING)])
            # Index for admin queries
            self.usage_collection.create_index([("timestamp", DESCENDING)])
            # Index for endpoint analytics
            self.usage_collection.create_index([("endpoint", 1), ("timestamp", DESCENDING)])
            # Index for daily stats
            self.daily_stats_collection.create_index([("date", DESCENDING), ("user_id", 1)])
            
            logger.info("✓ Usage tracking indexes created")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    def log_operation(
        self,
        user_id: str,
        endpoint: str,
        operation_type: str,
        tokens_used: int = 0,
        success: bool = True,
        response_time_ms: float = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an AI operation.
        
        Args:
            user_id: User performing the operation
            endpoint: API endpoint called
            operation_type: Type of operation (chat, summarize, qa, etc.)
            tokens_used: Estimated tokens used
            success: Whether operation succeeded
            response_time_ms: Response time in milliseconds
            metadata: Additional metadata (model used, document count, etc.)
        """
        try:
            log_entry = {
                "user_id": user_id,
                "endpoint": endpoint,
                "operation_type": operation_type,
                "tokens_used": tokens_used,
                "success": success,
                "response_time_ms": response_time_ms,
                "timestamp": datetime.utcnow(),
                "metadata": metadata or {}
            }
            
            self.usage_collection.insert_one(log_entry)
            
            # Update daily stats
            self._update_daily_stats(user_id, tokens_used, success)
            
        except Exception as e:
            logger.error(f"Failed to log usage: {e}")
    
    def _update_daily_stats(self, user_id: str, tokens: int, success: bool):
        """Update aggregated daily statistics."""
        try:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            
            self.daily_stats_collection.update_one(
                {"user_id": user_id, "date": today},
                {
                    "$inc": {
                        "total_operations": 1,
                        "total_tokens": tokens,
                        "successful_operations": 1 if success else 0,
                        "failed_operations": 0 if success else 1
                    },
                    "$setOnInsert": {"date": today}
                },
                upsert=True
            )
            
            # Also update global daily stats
            self.daily_stats_collection.update_one(
                {"user_id": "global", "date": today},
                {
                    "$inc": {
                        "total_operations": 1,
                        "total_tokens": tokens,
                        "successful_operations": 1 if success else 0,
                        "failed_operations": 0 if success else 1
                    },
                    "$setOnInsert": {"date": today}
                },
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to update daily stats: {e}")
    
    def get_user_usage(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get usage statistics for a specific user.
        
        Args:
            user_id: User ID to get stats for
            days: Number of days to look back
            
        Returns:
            Dictionary with usage statistics
        """
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Get logs for this user
            logs = list(self.usage_collection.find({
                "user_id": user_id,
                "timestamp": {"$gte": start_date}
            }).sort("timestamp", DESCENDING))
            
            # Calculate statistics
            total_operations = len(logs)
            successful = sum(1 for log in logs if log.get("success", True))
            failed = total_operations - successful
            total_tokens = sum(log.get("tokens_used", 0) for log in logs)
            avg_response_time = (
                sum(log.get("response_time_ms", 0) for log in logs) / total_operations
                if total_operations > 0 else 0
            )
            
            # Operations by type
            operations_by_type = {}
            for log in logs:
                op_type = log.get("operation_type", "unknown")
                operations_by_type[op_type] = operations_by_type.get(op_type, 0) + 1
            
            # Daily breakdown
            daily_usage = {}
            for log in logs:
                date_key = log["timestamp"].strftime("%Y-%m-%d")
                if date_key not in daily_usage:
                    daily_usage[date_key] = {"operations": 0, "tokens": 0}
                daily_usage[date_key]["operations"] += 1
                daily_usage[date_key]["tokens"] += log.get("tokens_used", 0)
            
            return {
                "user_id": user_id,
                "period_days": days,
                "total_operations": total_operations,
                "successful_operations": successful,
                "failed_operations": failed,
                "total_tokens": total_tokens,
                "avg_response_time_ms": round(avg_response_time, 2),
                "success_rate": round((successful / total_operations * 100) if total_operations > 0 else 0, 2),
                "operations_by_type": operations_by_type,
                "daily_usage": daily_usage,
                "most_recent_activity": logs[0]["timestamp"] if logs else None
            }
        except Exception as e:
            logger.error(f"Failed to get user usage: {e}")
            return {"error": str(e)}
    
    def get_global_usage(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get global usage statistics across all users.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with global usage statistics
        """
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Get all logs in period
            logs = list(self.usage_collection.find({
                "timestamp": {"$gte": start_date}
            }))
            
            # Calculate statistics
            total_operations = len(logs)
            successful = sum(1 for log in logs if log.get("success", True))
            failed = total_operations - successful
            total_tokens = sum(log.get("tokens_used", 0) for log in logs)
            unique_users = len(set(log.get("user_id") for log in logs))
            avg_response_time = (
                sum(log.get("response_time_ms", 0) for log in logs) / total_operations
                if total_operations > 0 else 0
            )
            
            # Operations by type
            operations_by_type = {}
            tokens_by_type = {}
            for log in logs:
                op_type = log.get("operation_type", "unknown")
                operations_by_type[op_type] = operations_by_type.get(op_type, 0) + 1
                tokens_by_type[op_type] = tokens_by_type.get(op_type, 0) + log.get("tokens_used", 0)
            
            # Operations by endpoint
            operations_by_endpoint = {}
            for log in logs:
                endpoint = log.get("endpoint", "unknown")
                operations_by_endpoint[endpoint] = operations_by_endpoint.get(endpoint, 0) + 1
            
            # Daily breakdown
            daily_usage = {}
            for log in logs:
                date_key = log["timestamp"].strftime("%Y-%m-%d")
                if date_key not in daily_usage:
                    daily_usage[date_key] = {"operations": 0, "tokens": 0, "users": set()}
                daily_usage[date_key]["operations"] += 1
                daily_usage[date_key]["tokens"] += log.get("tokens_used", 0)
                daily_usage[date_key]["users"].add(log.get("user_id"))
            
            # Convert sets to counts
            for date_key in daily_usage:
                daily_usage[date_key]["unique_users"] = len(daily_usage[date_key]["users"])
                del daily_usage[date_key]["users"]
            
            # Top users by operations
            user_operations = {}
            for log in logs:
                uid = log.get("user_id")
                user_operations[uid] = user_operations.get(uid, 0) + 1
            top_users = sorted(user_operations.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "period_days": days,
                "total_operations": total_operations,
                "successful_operations": successful,
                "failed_operations": failed,
                "total_tokens": total_tokens,
                "unique_users": unique_users,
                "avg_response_time_ms": round(avg_response_time, 2),
                "success_rate": round((successful / total_operations * 100) if total_operations > 0 else 0, 2),
                "operations_by_type": operations_by_type,
                "tokens_by_type": tokens_by_type,
                "operations_by_endpoint": operations_by_endpoint,
                "daily_usage": daily_usage,
                "top_users": [{"user_id": uid, "operations": count} for uid, count in top_users],
                "timestamp": datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Failed to get global usage: {e}")
            return {"error": str(e)}
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """
        Get real-time statistics for the last hour.
        
        Returns:
            Dictionary with real-time stats
        """
        try:
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            
            recent_logs = list(self.usage_collection.find({
                "timestamp": {"$gte": one_hour_ago}
            }))
            
            total_operations = len(recent_logs)
            successful = sum(1 for log in recent_logs if log.get("success", True))
            total_tokens = sum(log.get("tokens_used", 0) for log in recent_logs)
            active_users = len(set(log.get("user_id") for log in recent_logs))
            
            # Operations in last 5 minutes
            five_min_ago = datetime.utcnow() - timedelta(minutes=5)
            recent_5min = sum(1 for log in recent_logs if log["timestamp"] >= five_min_ago)
            
            return {
                "last_hour": {
                    "total_operations": total_operations,
                    "successful_operations": successful,
                    "total_tokens": total_tokens,
                    "active_users": active_users,
                    "success_rate": round((successful / total_operations * 100) if total_operations > 0 else 100, 2)
                },
                "last_5_minutes": {
                    "operations": recent_5min
                },
                "timestamp": datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Failed to get realtime stats: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()

    def get_today_tokens(self, user_id: str) -> int:
        """Return total tokens used by user for today (UTC midnight to now)."""
        try:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            doc = self.daily_stats_collection.find_one({"user_id": user_id, "date": today})
            if not doc:
                return 0
            return int(doc.get("total_tokens", 0) or 0)
        except Exception as e:
            logger.error(f"Failed to get today's tokens for {user_id}: {e}")
            return 0

    def reset_user_daily_usage(self, user_id: str):
        """Reset a user's daily aggregated stats for today to zero.

        This affects the `daily_stats` collection only; raw `usage_logs` remain for audit.
        """
        try:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            self.daily_stats_collection.update_one(
                {"user_id": user_id, "date": today},
                {
                    "$set": {
                        "total_operations": 0,
                        "total_tokens": 0,
                        "successful_operations": 0,
                        "failed_operations": 0
                    }
                },
                upsert=True
            )
            logger.info(f"Reset daily usage for user {user_id} (date={today.isoformat()})")
        except Exception as e:
            logger.error(f"Failed to reset daily usage for {user_id}: {e}")


# Global instance
usage_tracker = UsageTracker()
