"""
Planning Strategy API - AI returns ONLY strategy (no timestamps).

AI MUST NOT assign timestamps or place events. Scheduler does that.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import json

from server.deps import get_current_user, get_user_id
from server.schemas import ErrorResponse
from core.llm import get_langchain_llm

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai/v2", tags=["planning-strategy"])


# ---------------------------------------------------------------------------
# Strict structured output - NO timestamps, NO scheduling
# ---------------------------------------------------------------------------

class TopicStrategy(BaseModel):
    name: str
    difficultyScore: float = Field(ge=0, le=10, description="0-10")
    estimatedHours: float = Field(ge=0.1, le=100)
    priorityWeight: float = Field(ge=0, le=1)
    dependencies: List[str] = []
    revisionStrategy: str = "spaced"


class PlanningStrategyResponse(BaseModel):
    subjects: List[str] = []
    topics: List[TopicStrategy]
    totalEstimatedHours: float
    recommendedDailyLoadRange: dict  # { "minHours": float, "maxHours": float }
    emergencyPlan: Optional[dict] = None  # optional compression hints


class PlanningStrategyRequest(BaseModel):
    subject: str
    topics: List[str]
    difficulty: str = "medium"
    examDate: Optional[str] = None
    dailyStudyHours: float = Field(default=2, ge=0.5, le=12)
    planType: str = "custom"


def _heuristic_fallback(subject: str, topics: List[str], difficulty: str) -> dict:
    """Deterministic fallback when LLM fails. No timestamps."""
    hours_map = {"easy": 1.5, "medium": 2.5, "hard": 4.0}
    h = hours_map.get(difficulty.lower(), 2.5)
    total = len(topics) * h
    return {
        "subjects": [subject],
        "topics": [
            {
                "name": t,
                "difficultyScore": 5.0,
                "estimatedHours": h,
                "priorityWeight": 1.0 / max(len(topics), 1),
                "dependencies": [],
                "revisionStrategy": "spaced",
            }
            for t in topics
        ],
        "totalEstimatedHours": total,
        "recommendedDailyLoadRange": {"minHours": 1, "maxHours": 4},
        "emergencyPlan": None,
    }


def _parse_llm_strategy(text: str) -> Optional[dict]:
    """Parse LLM JSON; return None if invalid."""
    text = text.strip()
    for prefix in ("```json", "```"):
        if prefix in text:
            try:
                idx = text.index(prefix) + len(prefix)
                end = text.find("```", idx)
                if end == -1:
                    end = len(text)
                text = text[idx:end].strip()
            except Exception:
                pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def _validate_and_coerce(raw: dict, subject: str, topics: List[str]) -> dict:
    """Coerce raw LLM output into PlanningStrategyResponse shape. No timestamps allowed."""
    if not isinstance(raw.get("topics"), list) or len(raw["topics"]) == 0:
        raw["topics"] = [{"name": t, "difficultyScore": 5.0, "estimatedHours": 2.0, "priorityWeight": 1.0 / len(topics), "dependencies": [], "revisionStrategy": "spaced"} for t in topics]
    out_topics = []
    for i, t in enumerate(raw["topics"]):
        if isinstance(t, str):
            t = {"name": t, "difficultyScore": 5.0, "estimatedHours": 2.0, "priorityWeight": 1.0 / max(len(raw["topics"]), 1), "dependencies": [], "revisionStrategy": "spaced"}
        name = t.get("name") or (topics[i] if i < len(topics) else f"Topic {i+1}")
        out_topics.append({
            "name": str(name)[:200],
            "difficultyScore": max(0, min(10, float(t.get("difficultyScore", 5)))),
            "estimatedHours": max(0.1, min(100, float(t.get("estimatedHours", 2)))),
            "priorityWeight": max(0, min(1, float(t.get("priorityWeight", 1.0 / len(raw["topics"]))))),
            "dependencies": list(t.get("dependencies", [])) if isinstance(t.get("dependencies"), list) else [],
            "revisionStrategy": str(t.get("revisionStrategy", "spaced"))[:50],
        })
    total = raw.get("totalEstimatedHours")
    if total is None:
        total = sum(x["estimatedHours"] for x in out_topics)
    total = max(0.1, float(total))
    daily = raw.get("recommendedDailyLoadRange") or {}
    if not isinstance(daily, dict):
        daily = {}
    min_h = max(0.5, min(12, float(daily.get("minHours", 1))))
    max_h = max(min_h, min(12, float(daily.get("maxHours", 4))))
    return {
        "subjects": list(raw.get("subjects", [subject]))[:20],
        "topics": out_topics,
        "totalEstimatedHours": total,
        "recommendedDailyLoadRange": {"minHours": min_h, "maxHours": max_h},
        "emergencyPlan": raw.get("emergencyPlan") if isinstance(raw.get("emergencyPlan"), dict) else None,
    }


@router.post(
    "/planning-strategy",
    response_model=PlanningStrategyResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_planning_strategy(
    request: PlanningStrategyRequest,
    user_id: str = Depends(get_user_id),
) -> PlanningStrategyResponse:
    """
    Returns ONLY strategy: subjects, topics with difficulty/estimatedHours/priority/dependencies/revision.
    NO timestamps, NO scheduling. Scheduler uses this to place sessions.
    """
    try:
        prompt = f"""You are a study strategy expert. Output ONLY a JSON object. Do NOT assign any dates, times, or schedule.

Subject: {request.subject}
Topics: {', '.join(request.topics)}
Difficulty: {request.difficulty}
Plan type: {request.planType}
{f"Exam date: {request.examDate}" if request.examDate else ""}

Output this exact structure (no other fields, no timestamps):
{{
  "subjects": ["{request.subject}"],
  "topics": [
    {{
      "name": "topic name",
      "difficultyScore": 5,
      "estimatedHours": 2.0,
      "priorityWeight": 0.25,
      "dependencies": [],
      "revisionStrategy": "spaced"
    }}
  ],
  "totalEstimatedHours": 10,
  "recommendedDailyLoadRange": {{ "minHours": 1, "maxHours": 3 }},
  "emergencyPlan": null
}}

Rules: difficultyScore 0-10, estimatedHours per topic, priorityWeight 0-1 (sum to 1), revisionStrategy one word.
Return ONLY valid JSON, no markdown."""

        llm = get_langchain_llm()
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        raw = _parse_llm_strategy(text)

        if not raw:
            logger.warning("Planning strategy LLM parse failed, using heuristic fallback")
            raw = _heuristic_fallback(request.subject, request.topics, request.difficulty)
        else:
            raw = _validate_and_coerce(raw, request.subject, request.topics)

        return PlanningStrategyResponse(**raw)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Planning strategy error: %s", e)
        fallback = _heuristic_fallback(request.subject, request.topics, request.difficulty)
        return PlanningStrategyResponse(**fallback)
