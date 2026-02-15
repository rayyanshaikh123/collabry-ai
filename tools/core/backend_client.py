"""
Backend API Client - Interface to Node.js Backend

Provides type-safe HTTP communication with the backend API
for strategy execution, exam mode, and scheduling metadata.
"""
import httpx
from typing import Optional, Dict, Any, List
from config import CONFIG
import logging

logger = logging.getLogger(__name__)


class BackendAPIClient:
    """Client for communicating with Node.js backend API"""
    
    def __init__(self, base_url: str = None, timeout: int = None):
        """
        Initialize backend API client
        
        Args:
            base_url: Backend URL (defaults to CONFIG.BACKEND_URL)
            timeout: Request timeout in seconds (defaults to CONFIG.BACKEND_TIMEOUT)
        """
        self.base_url = (base_url or CONFIG.BACKEND_URL).rstrip('/')
        self.api_prefix = CONFIG.BACKEND_API_PREFIX.rstrip('/')
        self.timeout = timeout or CONFIG.BACKEND_TIMEOUT
        self.client = httpx.AsyncClient(timeout=self.timeout)
        
        logger.info(f"BackendAPIClient initialized: {self.base_url}{self.api_prefix}")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
    
    def _build_url(self, endpoint: str) -> str:
        """Build full API URL"""
        endpoint = endpoint.lstrip('/')
        return f"{self.base_url}{self.api_prefix}/{endpoint}"
    
    def _build_headers(self, token: str = None) -> Dict[str, str]:
        """Build request headers with optional auth token"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers
    
    # ============================================================================
    # STRATEGY PATTERN APIs
    # ============================================================================
    
    async def get_recommended_mode(
        self, 
        plan_id: str, 
        token: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get recommended planner mode for a study plan
        
        Args:
            plan_id: Study plan ID
            token: User JWT token
            
        Returns:
            {
                "recommendedMode": "balanced|adaptive|emergency",
                "confidence": 85,
                "reasoning": ["..."],
                "metrics": {...},
                "warnings": [...]
            }
        """
        try:
            url = self._build_url(f"study-planner/plans/{plan_id}/recommended-mode")
            headers = self._build_headers(token)
            
            logger.info(f"Fetching recommended mode for plan {plan_id}")
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if data.get("success"):
                return data.get("data")
            
            logger.warning(f"Backend returned success=false: {data}")
            return None
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting recommended mode: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error getting recommended mode: {e}")
            return None
    
    async def get_exam_strategy(
        self, 
        plan_id: str, 
        token: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get exam strategy configuration for a plan
        
        Args:
            plan_id: Study plan ID
            token: User JWT token
            
        Returns:
            {
                "currentPhase": "practice_heavy",
                "daysToExam": 12,
                "intensityMultiplier": 1.3,
                "taskDensityPerDay": 4,
                "phaseConfig": {...}
            }
        """
        try:
            url = self._build_url(f"study-planner/plans/{plan_id}/exam-strategy")
            headers = self._build_headers(token)
            
            logger.info(f"Fetching exam strategy for plan {plan_id}")
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if data.get("success"):
                return data.get("data")
            
            logger.warning(f"Backend returned success=false: {data}")
            return None
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(f"Plan {plan_id} not found or has no exam strategy")
            else:
                logger.error(f"HTTP error getting exam strategy: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error getting exam strategy: {e}")
            return None
    
    async def get_available_strategies(self, token: str) -> List[Dict[str, Any]]:
        """
        Get available scheduling strategies
        
        Args:
            token: User JWT token
            
        Returns:
            [
                {
                    "name": "Balanced Mode",
                    "description": "...",
                    "className": "BalancedStrategy"
                },
                ...
            ]
        """
        try:
            url = self._build_url("study-planner/strategies")
            headers = self._build_headers(token)
            
            logger.info("Fetching available strategies")
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if data.get("success"):
                return data.get("data", [])
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting strategies: {e}")
            return []
    
    # ============================================================================
    # BEHAVIOR LEARNING APIs
    # ============================================================================
    
    async def get_behavior_profile(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Get user behavior profile for optimal scheduling
        
        Args:
            token: User JWT token
            
        Returns:
            {
                "productivityPeakHours": [9, 14, 20],
                "completionRateByTimeSlot": {...},
                "consistencyScore": 75,
                "optimalTimeOfDay": "morning"
            }
        """
        try:
            url = self._build_url("study-planner/analytics/behavior-profile")
            headers = self._build_headers(token)
            
            logger.info("Fetching user behavior profile")
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if data.get("success"):
                return data.get("data")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting behavior profile: {e}")
            return None
    
    # ============================================================================
    # PLAN MANAGEMENT APIs
    # ============================================================================
    
    async def get_plan_by_id(
        self, 
        plan_id: str, 
        token: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get study plan by ID
        
        Args:
            plan_id: Study plan ID
            token: User JWT token
            
        Returns:
            {
                "_id": "...",
                "title": "...",
                "examDate": "...",
                "examMode": true,
                "currentExamPhase": "...",
                ...
            }
        """
        try:
            url = self._build_url(f"study-planner/plans/{plan_id}")
            headers = self._build_headers(token)
            
            logger.info(f"Fetching plan {plan_id}")
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if data.get("success"):
                return data.get("data")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting plan: {e}")
            return None


# Singleton instance
_backend_client: Optional[BackendAPIClient] = None


def get_backend_client() -> BackendAPIClient:
    """Get or create singleton backend client"""
    global _backend_client
    if _backend_client is None:
        _backend_client = BackendAPIClient()
    return _backend_client


async def close_backend_client():
    """Close backend client connection"""
    global _backend_client
    if _backend_client is not None:
        await _backend_client.close()
        _backend_client = None
