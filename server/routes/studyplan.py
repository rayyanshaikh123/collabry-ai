"""
Study Plan Generation Route

Generates intelligent study plans with daily task breakdown using AI.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from server.deps import get_current_user
from server.schemas import ErrorResponse
from core.local_llm import create_llm
from config import CONFIG
import logging
import json

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["study-planner"])


class StudyPlanRequest(BaseModel):
    """Request for AI study plan generation"""
    subject: str = Field(..., description="Subject or topic to study")
    topics: List[str] = Field(..., description="List of topics to cover")
    startDate: str = Field(..., description="Start date (ISO format)")
    endDate: str = Field(..., description="End date (ISO format)")
    dailyStudyHours: float = Field(default=2, ge=0.5, le=12, description="Hours per day")
    preferredTimeSlots: List[str] = Field(default=["evening"], description="Preferred study times")
    difficulty: str = Field(default="intermediate", description="Difficulty level")
    planType: str = Field(default="custom", description="Plan type: exam, course, skill, custom")
    examDate: Optional[str] = Field(None, description="Exam/deadline date if applicable")
    currentKnowledge: Optional[str] = Field(None, description="Current knowledge level")
    goals: Optional[str] = Field(None, description="Learning goals")


class TaskGenerated(BaseModel):
    """Generated task structure"""
    title: str
    description: str
    topic: str
    scheduledDate: str
    duration: int  # minutes
    priority: str
    difficulty: str
    order: int
    resources: List[Dict[str, str]] = []


class StudyPlanResponse(BaseModel):
    """Response with generated study plan"""
    title: str
    description: str
    tasks: List[TaskGenerated]
    estimatedCompletionDays: int
    totalTasks: int
    recommendations: List[str]
    warnings: List[str] = []  # Complexity/timeline warnings


def calculate_study_days(start_date: datetime, end_date: datetime) -> int:
    """Calculate number of available study days"""
    delta = end_date - start_date
    return max(1, delta.days + 1)


def assess_plan_complexity(topics: List[str], num_days: int, daily_hours: float, difficulty: str) -> List[str]:
    """Assess if plan is realistic and return warnings if needed"""
    warnings = []
    
    # Calculate total available study hours
    total_hours = num_days * daily_hours
    
    # Estimate minimum hours needed per topic based on difficulty
    hours_per_topic = {
        'beginner': 8,     # 8 hours minimum per topic for beginners
        'intermediate': 15, # 15 hours for intermediate
        'advanced': 25      # 25+ hours for advanced topics
    }
    
    min_hours_needed = len(topics) * hours_per_topic.get(difficulty, 15)
    
    # Check if timeline is unrealistic
    if total_hours < min_hours_needed:
        shortage = min_hours_needed - total_hours
        warnings.append(
            f"⚠️ TIMELINE WARNING: You're trying to cover {len(topics)} {difficulty}-level topics in {num_days} days "
            f"({total_hours:.1f} total hours). This would require approximately {min_hours_needed:.0f} hours minimum. "
            f"You're {shortage:.0f} hours short. This plan provides an overview but may not allow for deep mastery."
        )
    
    # Check if too many topics for short duration
    if len(topics) > 10 and num_days < 14:
        warnings.append(
            f"⚠️ SCOPE WARNING: {len(topics)} topics in {num_days} days is very ambitious! "
            "Consider focusing on fewer topics for better retention, or extend your timeline."
        )
    
    # Check if daily hours are too high
    if daily_hours > 6:
        warnings.append(
            f"⚠️ SUSTAINABILITY WARNING: {daily_hours} hours daily is intense! "
            "Risk of burnout is high. Consider spreading learning over more days with 3-4 hours daily."
        )
    
    # Check for very short study periods
    if num_days < 3:
        warnings.append(
            "⚠️ DURATION WARNING: Learning in less than 3 days limits long-term retention. "
            "This will be a quick overview. For lasting understanding, extend to at least 7 days."
        )
    
    # Advanced topics in short time
    if difficulty == 'advanced' and total_hours < len(topics) * 20:
        warnings.append(
            "⚠️ DEPTH WARNING: Advanced topics require significant time for mastery. "
            "This plan covers fundamentals and key concepts. Expect to need additional practice time."
        )
    
    # Multiple complex topics
    complex_keywords = ['algorithm', 'system', 'architecture', 'design', 'theory', 'advanced', 'deep', 'machine learning', 'ai', 'calculus', 'physics']
    complex_topics = [t for t in topics if any(kw in t.lower() for kw in complex_keywords)]
    
    if len(complex_topics) >= 3 and num_days < 21:
        warnings.append(
            f"⚠️ COMPLEXITY WARNING: Detected {len(complex_topics)} complex topics ({', '.join(complex_topics[:3])}...). "
            "These require substantial practice and application. This plan covers theoretical foundation - "
            "practical mastery will need additional hands-on work."
        )
    
    return warnings


def normalize_difficulty(difficulty: str) -> str:
    """Normalize difficulty to match MongoDB schema: easy, medium, hard."""
    difficulty = difficulty.lower().strip()
    if difficulty in ['beginner', 'easy', 'simple']:
        return 'easy'
    elif difficulty in ['intermediate', 'medium', 'moderate', 'average']:
        return 'medium'
    elif difficulty in ['advanced', 'hard', 'difficult', 'expert']:
        return 'hard'
    return 'medium'  # default


def normalize_recommendations(recommendations: List) -> List[str]:
    """Normalize recommendations to list of strings."""
    normalized = []
    for rec in recommendations:
        if isinstance(rec, str):
            normalized.append(rec)
        elif isinstance(rec, dict):
            # Extract string from object (e.g., {"title": "..."})
            normalized.append(rec.get('title', rec.get('text', str(rec))))
        else:
            normalized.append(str(rec))
    return normalized


def _get_phase_description(phase: str, topic: str) -> str:
    """Generate description for learning phase"""
    descriptions = {
        "Introduction": f"Learn the basics of {topic}, understand key concepts and terminology",
        "Deep Dive": f"Explore {topic} in depth, study advanced concepts and techniques",
        "Practice": f"Apply {topic} knowledge through exercises, problems, and hands-on practice",
        "Review": f"Review and consolidate {topic}, test understanding and fill knowledge gaps"
    }
    return descriptions.get(phase, f"Study {topic}")


def distribute_topics_across_days(
    topics: List[str],
    num_days: int,
    daily_hours: float,
    difficulty: str
) -> List[Dict[str, Any]]:
    """Distribute topics across available days"""
    tasks = []
    
    # Estimate time per topic based on difficulty
    time_multiplier = {
        "beginner": 1.0,
        "intermediate": 1.5,
        "advanced": 2.0
    }
    base_time = 60  # minutes
    topic_duration = int(base_time * time_multiplier.get(difficulty, 1.0))
    
    # Calculate tasks per day
    daily_minutes = daily_hours * 60
    tasks_per_day = max(1, int(daily_minutes / topic_duration))
    
    # Distribute topics
    current_day = 0
    for i, topic in enumerate(topics):
        day_index = i // tasks_per_day
        if day_index >= num_days:
            day_index = num_days - 1
        
        # Determine priority (urgent for exam prep, high for early topics)
        if day_index < num_days * 0.2:
            priority = "high"
        elif day_index > num_days * 0.8:
            priority = "urgent"
        else:
            priority = "medium"
        
        tasks.append({
            "topic": topic,
            "day_offset": day_index,
            "duration": min(topic_duration, daily_minutes),
            "priority": priority,
            "order": i
        })
    
    return tasks


@router.post(
    "/generate-study-plan",
    response_model=StudyPlanResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Generate AI study plan",
    description="Generate a comprehensive study plan with daily tasks using AI"
)
async def generate_study_plan(
    request: StudyPlanRequest,
    user_id: str = Depends(get_current_user)
) -> StudyPlanResponse:
    """
    Generate an intelligent study plan with task breakdown.
    
    The AI analyzes:
    - Subject complexity
    - Available time
    - User preferences
    - Topic dependencies
    
    Returns structured plan with daily tasks.
    """
    try:
        logger.info(f"Generating study plan for user={user_id}, subject={request.subject}")
        
        # Normalize difficulty to match MongoDB schema
        request.difficulty = normalize_difficulty(request.difficulty)
        
        # Parse dates
        start_date = datetime.fromisoformat(request.startDate.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.endDate.replace('Z', '+00:00'))
        
        # Validate dates
        if end_date <= start_date:
            raise HTTPException(400, "End date must be after start date")
        
        num_days = calculate_study_days(start_date, end_date)
        logger.info(f"Plan duration: {num_days} days")
        
        # Assess plan complexity and generate warnings
        complexity_warnings = assess_plan_complexity(
            request.topics,
            num_days,
            request.dailyStudyHours,
            request.difficulty
        )
        logger.info(f"Generated {len(complexity_warnings)} warnings for plan complexity")
        
        # Calculate minimum tasks needed (at least 2 per topic)
        min_tasks = max(len(request.topics) * 2, num_days)
        
        # Build AI prompt optimized for llama3.1
        prompt = f"""You are an expert study planner. Create a detailed study schedule.

INPUT:
- Subject: {request.subject}
- Topics: {', '.join(request.topics)}
- Duration: {num_days} days
- Daily hours: {request.dailyStudyHours}
- Level: {request.difficulty}
{f"- Exam: {request.examDate}" if request.examDate else ""}

TASK:
Generate at least {min_tasks} study tasks covering all topics. Each topic should have multiple tasks.

OUTPUT FORMAT (JSON ONLY, NO MARKDOWN):
{{
  "title": "{request.subject} Study Plan",
  "description": "Complete study plan description",
  "tasks": [
    {{
      "title": "Introduction to [Topic]",
      "description": "Learn basic concepts and terminology",
      "topic": "{request.topics[0] if request.topics else 'Topic'}",
      "duration": 60,
      "priority": "high",
      "difficulty": "easy or medium or hard ONLY"
    }}
  ],
  "recommendations": [
    "Start with fundamentals",
    "Practice daily",
    "Review regularly"
  ]
}}

Generate the JSON now:"""

        # Initialize LLM with CONFIG
        llm = create_llm(CONFIG)
        
        # Generate plan with AI
        logger.info("Calling LLM for study plan generation...")
        response = llm.invoke(prompt)
        logger.info(f"LLM response length: {len(response)} chars")
        
        # Extract and parse JSON with multiple strategies
        response_text = response.strip()
        ai_plan = None
        
        # Strategy 1: Direct JSON parse
        try:
            ai_plan = json.loads(response_text)
            logger.info("✓ Parsed JSON directly")
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove markdown code blocks
        if not ai_plan:
            try:
                cleaned = response_text
                if '```json' in cleaned:
                    cleaned = cleaned.split('```json')[1].split('```')[0].strip()
                elif '```' in cleaned:
                    cleaned = cleaned.split('```')[1].split('```')[0].strip()
                ai_plan = json.loads(cleaned)
                logger.info("✓ Parsed JSON after removing markdown")
            except (json.JSONDecodeError, IndexError):
                pass
        
        # Strategy 3: Extract JSON from text
        if not ai_plan:
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx + 1]
                    ai_plan = json.loads(json_str)
                    logger.info("✓ Extracted JSON from response")
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Strategy 4: Fallback to programmatic generation
        if not ai_plan:
            logger.warning(f"All JSON parsing failed. Response preview: {response_text[:200]}")
            logger.info("Using intelligent fallback task distribution")
            
            # Fallback: Create intelligent programmatic plan
            task_distribution = distribute_topics_across_days(
                request.topics,
                num_days,
                request.dailyStudyHours,
                request.difficulty
            )
            
            # Expand topics into learning phases
            expanded_tasks = []
            phases = ["Introduction", "Deep Dive", "Practice", "Review"]
            
            for topic in request.topics:
                for phase in phases[:min(3, max(2, num_days // len(request.topics)))]:
                    expanded_tasks.append({
                        "topic": topic,
                        "phase": phase,
                        "title": f"{phase}: {topic}",
                        "description": _get_phase_description(phase, topic)
                    })
            
            ai_plan = {
                "title": f"{request.subject} Study Plan",
                "description": f"Master {len(request.topics)} topics over {num_days} days with structured learning phases",
                "tasks": [],
                "recommendations": [
                    f"Complete {request.dailyStudyHours} hours daily for best results",
                    "Start with fundamentals, progress to advanced concepts",
                    "Take 5-10 minute breaks every hour",
                    "Review previous topics before starting new ones",
                    "Practice actively with exercises and projects"
                ]
            }
            
            # Generate tasks from expanded distribution
            for i, task_info in enumerate(expanded_tasks):
                day_offset = (i * num_days) // len(expanded_tasks)
                ai_plan["tasks"].append({
                    "title": task_info["title"],
                    "description": task_info["description"],
                    "topic": task_info["topic"],
                    "duration": int((request.dailyStudyHours * 60) // (len(expanded_tasks) / num_days)),
                    "priority": "high" if i < len(expanded_tasks) * 0.3 else "medium",
                    "difficulty": request.difficulty,
                    "resources": []
                })
        
        # Ensure we have tasks
        if not ai_plan or not ai_plan.get("tasks"):
            logger.error("AI plan has no tasks, creating minimal plan")
            ai_plan = {
                "title": f"{request.subject} Study Plan",
                "description": f"Study {len(request.topics)} topics",
                "tasks": [{"title": f"Learn {topic}", "description": f"Study {topic} concepts", "topic": topic, "duration": 60, "priority": "medium", "difficulty": request.difficulty} for topic in request.topics],
                "recommendations": ["Study consistently", "Practice regularly", "Review often"]
            }
        
        logger.info(f"✓ AI plan has {len(ai_plan.get('tasks', []))} tasks")
        
        # Assign dates to tasks
        tasks_with_dates = []
        tasks_per_day = max(1, len(ai_plan.get("tasks", [])) // num_days)
        
        for i, task in enumerate(ai_plan.get("tasks", [])):
            day_offset = i // tasks_per_day
            if day_offset >= num_days:
                day_offset = num_days - 1
            
            task_date = start_date + timedelta(days=day_offset)
            
            tasks_with_dates.append(TaskGenerated(
                title=task.get("title", f"Study Session {i+1}"),
                description=task.get("description", "Study and practice"),
                topic=task.get("topic", request.topics[i % len(request.topics)]),
                scheduledDate=task_date.isoformat(),
                duration=task.get("duration", 60),
                priority=task.get("priority", "medium"),
                difficulty=task.get("difficulty", request.difficulty),
                order=i,
                resources=task.get("resources", [])
            ))
        
        # Normalize recommendations to strings
        raw_recommendations = ai_plan.get("recommendations", [
            "Stay consistent with your daily schedule",
            "Review regularly to reinforce learning",
            "Practice actively, don't just read passively"
        ])
        normalized_recommendations = normalize_recommendations(raw_recommendations)
        
        # Build response with warnings
        response = StudyPlanResponse(
            title=ai_plan.get("title", f"{request.subject} Study Plan"),
            description=ai_plan.get("description", f"Comprehensive study plan for {request.subject}"),
            tasks=tasks_with_dates,
            estimatedCompletionDays=num_days,
            totalTasks=len(tasks_with_dates),
            recommendations=normalized_recommendations,
            warnings=complexity_warnings
        )
        
        logger.info(f"Generated plan with {len(tasks_with_dates)} tasks over {num_days} days")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating study plan: {e}")
        raise HTTPException(500, f"Failed to generate study plan: {str(e)}")
