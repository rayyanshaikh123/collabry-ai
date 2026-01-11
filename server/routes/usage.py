"""
Usage tracking and monitoring endpoints.

Provides:
- User-level usage statistics
- Admin-level global analytics
- Real-time monitoring
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from server.schemas import (
    UsageStatsResponse,
    GlobalUsageResponse,
    RealtimeStatsResponse,
    ErrorResponse
)
from server.deps import get_current_user
from core.usage_tracker import usage_tracker
from datetime import datetime
import logging
from fastapi import Body

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/usage", tags=["usage"])


# Subscription tier limits (tokens per month)
SUBSCRIPTION_LIMITS = {
    "free": 10000,
    "basic": 50000,
    "pro": 200000,
    "enterprise": 1000000
}


@router.get(
    "/stats",
    response_model=GlobalUsageResponse,
    summary="Get public usage statistics",
    description="Get aggregated usage statistics for admin dashboard. No authentication required."
)
async def get_public_stats(
    days: int = Query(7, ge=1, le=365, description="Number of days to look back")
):
    """
    Get public usage statistics (no auth required).
    
    Returns aggregated statistics without exposing sensitive user data.
    Suitable for public admin dashboards.
    """
    try:
        usage_data = usage_tracker.get_global_usage(days=days)
        
        if "error" in usage_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve usage data: {usage_data['error']}"
            )
        
        # Keep real user IDs for admin dashboard (they can resolve names on frontend)
        # Top users already have user_id and operations from usage_tracker
        
        return GlobalUsageResponse(**usage_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting public stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve usage statistics: {str(e)}"
        )


@router.get(
    "/me",
    response_model=UsageStatsResponse,
    summary="Get my usage statistics",
    description="Get usage statistics for the authenticated user"
)
async def get_my_usage(
    request: Request,
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    user_id: str = Depends(get_current_user)
):
    """
    Get usage statistics for the current user.
    
    Returns:
        User-specific usage statistics including token usage, operations, and daily breakdown
    """
    try:
        usage_data = usage_tracker.get_user_usage(user_id, days=days)
        
        if "error" in usage_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve usage data: {usage_data['error']}"
            )
        
        # Get user subscription tier from JWT token
        subscription_tier = request.state.subscription_tier if hasattr(request.state, 'subscription_tier') else "free"
        limit = SUBSCRIPTION_LIMITS.get(subscription_tier, 10000)
        
        # Calculate usage percentage
        total_tokens = usage_data.get("total_tokens", 0)
        usage_percentage = (total_tokens / limit * 100) if limit > 0 else 0
        
        # Add subscription info
        usage_data["subscription_limit"] = limit
        usage_data["usage_percentage"] = round(usage_percentage, 2)
        
        return UsageStatsResponse(**usage_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting user usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve usage statistics: {str(e)}"
        )


@router.get(
    "/user/{user_id}",
    response_model=UsageStatsResponse,
    summary="Get user usage statistics (Admin only)",
    description="Get usage statistics for a specific user. Admin access required."
)
async def get_user_usage(
    user_id: str,
    request: Request,
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    admin_id: str = Depends(get_current_user)
):
    """
    Get usage statistics for a specific user (admin only).
    
    Args:
        user_id: Target user ID
        days: Number of days to look back
        
    Returns:
        User-specific usage statistics
        
    Note:
        This endpoint requires admin role.
    """
    try:
        # Admin role check
        from server.deps import is_admin
        auth_header = request.headers.get("authorization", "")
        token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
        
        if not is_admin(admin_id, token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        usage_data = usage_tracker.get_user_usage(user_id, days=days)
        
        if "error" in usage_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve usage data: {usage_data['error']}"
            )
        
        # Get subscription info (mock for now)
        subscription_tier = "free"
        limit = SUBSCRIPTION_LIMITS.get(subscription_tier, 10000)
        
        total_tokens = usage_data.get("total_tokens", 0)
        usage_percentage = (total_tokens / limit * 100) if limit > 0 else 0
        
        usage_data["subscription_limit"] = limit
        usage_data["usage_percentage"] = round(usage_percentage, 2)
        
        return UsageStatsResponse(**usage_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting user usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve usage statistics: {str(e)}"
        )


@router.get(
    "/global",
    response_model=GlobalUsageResponse,
    summary="Get global usage statistics (Admin only)",
    description="Get aggregated usage statistics across all users. Admin access required."
)
async def get_global_usage(
    request: Request,
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    admin_id: str = Depends(get_current_user)
):
    """
    Get global usage statistics across all users (admin only).
    
    Args:
        days: Number of days to look back
        
    Returns:
        Global usage statistics including all users, operations, and breakdowns
        
    Note:
        This endpoint requires admin role.
    """
    try:
        # Admin role check
        from server.deps import is_admin
        auth_header = request.headers.get("authorization", "")
        token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
        
        if not is_admin(admin_id, token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        usage_data = usage_tracker.get_global_usage(days=days)
        
        if "error" in usage_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve usage data: {usage_data['error']}"
            )
        
        return GlobalUsageResponse(**usage_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting global usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve global usage statistics: {str(e)}"
        )


@router.get(
    "/realtime",
    response_model=RealtimeStatsResponse,
    summary="Get real-time statistics (Admin only)",
    description="Get real-time usage statistics for the last hour. Admin access required."
)
async def get_realtime_stats(
    request: Request,
    admin_id: str = Depends(get_current_user)
):
    """
    Get real-time usage statistics (admin only).
    
    Returns:
        Real-time stats for the last hour including active users and operations
        
    Note:
        This endpoint requires admin role.
    """
    try:
        # Admin role check
        from server.deps import is_admin
        auth_header = request.headers.get("authorization", "")
        token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
        
        if not is_admin(admin_id, token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        stats = usage_tracker.get_realtime_stats()
        
        if "error" in stats:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve realtime stats: {stats['error']}"
            )
        
        return RealtimeStatsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting realtime stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve realtime statistics: {str(e)}"
        )


@router.post(
    "/reset-me",
    summary="Reset my daily usage",
    description="Reset the aggregated daily usage stats for the authenticated user (today)."
)
async def reset_my_usage(user_id: str = Depends(get_current_user)):
    try:
        usage_tracker.reset_user_daily_usage(user_id)
        return {"message": "Daily usage reset for current user."}
    except Exception as e:
        logger.exception(f"Error resetting usage for {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/reset/{target_user_id}",
    summary="Reset user daily usage (admin)",
    description="Reset the aggregated daily usage stats for a specific user. Admin only."
)
async def reset_user_usage(
    request: Request,
    target_user_id: str,
    admin_id: str = Depends(get_current_user)
):
    try:
        # Admin role check
        from server.deps import is_admin
        auth_header = request.headers.get("authorization", "")
        token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
        
        if not is_admin(admin_id, token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        usage_tracker.reset_user_daily_usage(target_user_id)
        return {"message": f"Daily usage reset for user {target_user_id}."}
    except Exception as e:
        logger.exception(f"Error resetting usage for {target_user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
