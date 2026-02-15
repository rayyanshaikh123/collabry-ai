"""
Study Plan Generation Route

Generates intelligent study plans with daily task breakdown using AI.
Integrates with backend strategy system for constraint-based scheduling.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from server.deps import get_current_user, get_user_id
from server.schemas import ErrorResponse
from core.llm import get_langchain_llm
from core.backend_client import get_backend_client
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
    planId: Optional[str] = Field(None, description="Existing plan ID for strategy context")
    authToken: Optional[str] = Field(None, description="User auth token for backend API calls")


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


def validate_task_duration(duration: int) -> int:
    """
    Validate and clamp task duration to MongoDB limits.
    Backend enforces: min=15 minutes, max=480 minutes (8 hours)
    """
    return max(15, min(int(duration) if duration is not None else 60, 480))


def validate_and_coerce_tasks(tasks: List[Dict[str, Any]], topics: List[str], difficulty: str) -> List[Dict[str, Any]]:
    """
    Schema validation for AI task list. Strips any timestamp fields from AI; scheduling is backend-only.
    Returns only task metadata: title, description, topic, duration, priority, difficulty, resources.
    """
    valid = []
    for i, t in enumerate(tasks) if isinstance(tasks, list) else []:
        if not isinstance(t, dict):
            continue
        title = (t.get("title") or f"Study {topics[i % len(topics)] if topics else 'Topic'}").strip()[:200]
        topic = (t.get("topic") or (topics[i % len(topics)] if topics else "General")).strip()[:200]
        duration = validate_task_duration(t.get("duration", 60))
        priority = str(t.get("priority") or "medium").lower()
        if priority not in ("low", "medium", "high", "urgent"):
            priority = "medium"
        diff = str(t.get("difficulty") or difficulty).lower()
        if diff not in ("easy", "medium", "hard"):
            diff = normalize_difficulty(diff)
        valid.append({
            "title": title,
            "description": (t.get("description") or "")[:1000],
            "topic": topic,
            "duration": duration,
            "priority": priority,
            "difficulty": diff,
            "resources": list(t.get("resources") or []) if isinstance(t.get("resources"), list) else [],
        })
    return valid


def apply_cognitive_load_limits(
    tasks: List[Dict[str, Any]],
    num_days: int,
    strategy_context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Apply cognitive load protection to task distribution.
    
    Constraints:
    - Max 4 tasks/day (balanced/adaptive mode)
    - Max 6-8 tasks/day (emergency mode)
    - Max 2 hard tasks/day
    - Balance difficulty across days
    
    Args:
        tasks: List of task dictionaries
        num_days: Number of study days
        strategy_context: Optional strategy metadata (intensity, mode)
        
    Returns:
        Filtered and redistributed tasks
    """
    # Determine max tasks per day from strategy
    max_tasks_per_day = 4  # Default (balanced mode)
    if strategy_context:
        mode = strategy_context.get('recommendedMode', 'balanced')
        if mode == 'emergency':
            max_tasks_per_day = 8
        elif mode == 'adaptive':
            task_density = strategy_context.get('phaseConfig', {}).get('taskDensityPerDay', 4)
            max_tasks_per_day = task_density
    
    logger.info(f"Applying cognitive load limits: max {max_tasks_per_day} tasks/day")
    
    # Group tasks by day
    tasks_by_day = {}
    for i, task in enumerate(tasks):
        day_index = i // max_tasks_per_day
        if day_index >= num_days:
            day_index = num_days - 1
        
        if day_index not in tasks_by_day:
            tasks_by_day[day_index] = []
        tasks_by_day[day_index].append(task)
    
    # Apply difficulty balancing: max 2 hard tasks per day
    balanced_tasks = []
    for day_index in sorted(tasks_by_day.keys()):
        day_tasks = tasks_by_day[day_index]
        
        # Count hard tasks
        hard_tasks = [t for t in day_tasks if t.get('difficulty') == 'hard']
        easy_medium_tasks = [t for t in day_tasks if t.get('difficulty') != 'hard']
        
        # If too many hard tasks, downgrade some to medium
        if len(hard_tasks) > 2:
            logger.warning(f"Day {day_index} has {len(hard_tasks)} hard tasks, limiting to 2")
            for task in hard_tasks[2:]:
                task['difficulty'] = 'medium'
                task['title'] = f"[Medium] {task['title']}"
        
        # Add tasks back (limit to max_tasks_per_day)
        balanced_tasks.extend(day_tasks[:max_tasks_per_day])
    
    logger.info(f"Cognitive load applied: {len(tasks)} → {len(balanced_tasks)} tasks")
    return balanced_tasks


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
    description="Generate a comprehensive study plan with daily tasks using AI with strategy-aware constraints"
)
async def generate_study_plan(
    request: StudyPlanRequest,
    user_id: str = Depends(get_user_id)
) -> StudyPlanResponse:
    """
    ⚠️ DEPRECATED - Use /ai/v2/generate-smart-schedule instead
    
    This endpoint generates DATE+DURATION tasks (legacy format).
    The new V2 endpoint generates TIME-SLOT SESSIONS with startTime/endTime.
    
    This will be removed in the next major version.
    
    The AI analyzes:
    - Subject complexity
    - Available time
    - User preferences
    - Topic dependencies
    - Backend strategy constraints (if planId provided)
    - Exam proximity and phase (if examDate present)
    - Cognitive load limits
    
    Returns structured plan with daily tasks.
    """
    # DEPRECATION WARNING
    logger.warning(
        f"⚠️ DEPRECATED ENDPOINT CALLED: /generate-study-plan by user {user_id}. "
        "Please migrate to /ai/v2/generate-smart-schedule which uses time-slot based scheduling."
    )
    
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
        
        # ========================================================================
        # PHASE 2: FETCH STRATEGY CONTEXT FROM BACKEND
        # ========================================================================
        strategy_context = None
        exam_strategy = None
        behavior_profile = None
        
        if request.planId and request.authToken:
            logger.info("🔗 Fetching strategy context from backend...")
            backend_client = get_backend_client()
            
            try:
                # Get recommended mode
                strategy_context = await backend_client.get_recommended_mode(
                    request.planId, 
                    request.authToken
                )
                if strategy_context:
                    logger.info(f"✓ Strategy context: {strategy_context.get('recommendedMode')} mode "
                              f"(confidence: {strategy_context.get('confidence')}%)")
                
                # Get exam strategy if exam date present
                if request.examDate:
                    exam_strategy = await backend_client.get_exam_strategy(
                        request.planId,
                        request.authToken
                    )
                    if exam_strategy:
                        logger.info(f"✓ Exam strategy: Phase={exam_strategy.get('currentPhase')}, "
                                  f"Intensity={exam_strategy.get('intensityMultiplier')}x, "
                                  f"Days to exam={exam_strategy.get('daysToExam')}")
                
                # Get user behavior profile for optimal scheduling
                behavior_profile = await backend_client.get_behavior_profile(
                    request.authToken
                )
                if behavior_profile:
                    logger.info(f"✓ Behavior profile: Peak hours={behavior_profile.get('productivityPeakHours')}, "
                              f"Consistency={behavior_profile.get('consistencyScore')}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch backend context: {e} - continuing with defaults")
        
        # Assess plan complexity and generate warnings
        complexity_warnings = assess_plan_complexity(
            request.topics,
            num_days,
            request.dailyStudyHours,
            request.difficulty
        )
        logger.info(f"Generated {len(complexity_warnings)} warnings for plan complexity")
        
        # ========================================================================
        # BUILD STRATEGY-AWARE LLM PROMPT
        # ========================================================================
        
        # Calculate task constraints based on strategy
        max_tasks_per_day = 4  # Default (balanced mode)
        intensity_multiplier = 1.0  # Default
        strategy_description = "standard balanced scheduling"
        
        if strategy_context:
            mode = strategy_context.get('recommendedMode', 'balanced')
            if mode == 'emergency':
                max_tasks_per_day = 8
                intensity_multiplier = 2.0
                strategy_description = "🚨 EMERGENCY MODE: Crisis compression with hyper time blocks (90-120 min sessions)"
            elif mode == 'adaptive':
                if exam_strategy:
                    max_tasks_per_day = exam_strategy.get('taskDensityPerDay', 4)
                    intensity_multiplier = exam_strategy.get('intensityMultiplier', 1.3)
                    phase = exam_strategy.get('currentPhase', 'unknown')
                    strategy_description = f"📚 ADAPTIVE MODE: Exam-driven ({phase} phase, {intensity_multiplier}x intensity)"
                else:
                    max_tasks_per_day = 4
                    intensity_multiplier = 1.3
                    strategy_description = "📚 ADAPTIVE MODE: Smart priority-based scheduling"
            else:
                strategy_description = "⚖️ BALANCED MODE: Standard scheduling with optimal distribution"
        
        # Calculate minimum tasks needed
        min_tasks = max(len(request.topics) * 2, num_days)
        recommended_tasks = min(min_tasks, num_days * max_tasks_per_day)
        
        # Build enhanced AI prompt with strategy context
        strategy_constraints = f"""
SCHEDULING STRATEGY: {strategy_description}
- Maximum tasks per day: {max_tasks_per_day}
- Intensity multiplier: {intensity_multiplier}x
- Cognitive load protection: Max 2 hard tasks per day
"""
        
        exam_context = ""
        if request.examDate and exam_strategy:
            exam_context = f"""
EXAM MODE ACTIVE:
- Exam date: {request.examDate}
- Days remaining: {exam_strategy.get('daysToExam')}
- Current phase: {exam_strategy.get('currentPhase')}
- Focus areas: {', '.join(exam_strategy.get('phaseConfig', {}).get('focusAreas', []))}
- Priority: HIGH-PRIORITY topics first, exam-critical concepts emphasized
"""
        
        behavior_context = ""
        if behavior_profile:
            peak_hours = behavior_profile.get('productivityPeakHours', [])
            optimal_slot = behavior_profile.get('optimalTimeOfDay', 'evening')
            behavior_context = f"""
USER BEHAVIOR INSIGHTS:
- Peak productivity hours: {', '.join(map(str, peak_hours))}
- Optimal time of day: {optimal_slot}
- Consistency score: {behavior_profile.get('consistencyScore', 0)}/100
"""
        
        prompt = f"""You are an expert study planner with constraint-based scheduling intelligence.

INPUT:
- Subject: {request.subject}
- Topics: {', '.join(request.topics)}
- Duration: {num_days} days
- Daily hours: {request.dailyStudyHours}
- Level: {request.difficulty}
{f"- Exam: {request.examDate}" if request.examDate else ""}

{strategy_constraints}
{exam_context if exam_context else ""}
{behavior_context if behavior_context else ""}

TASK:
Generate {recommended_tasks} study tasks covering all topics. Each topic should have multiple tasks.

CRITICAL CONSTRAINTS:
- Each task duration must be between 15 and 480 minutes (0.25 to 8 hours)
- Daily task count must not exceed {max_tasks_per_day} tasks
- Maximum 2 hard-difficulty tasks per day
- Total daily task durations should not exceed {int(request.dailyStudyHours * 60)} minutes
- Apply {intensity_multiplier}x intensity factor to pacing
{"- EMERGENCY MODE: Prioritize core concepts, skip low-priority details" if strategy_context and strategy_context.get('recommendedMode') == 'emergency' else ""}

OUTPUT FORMAT (JSON ONLY, NO MARKDOWN):
- Do NOT output scheduledDate, startTime, endTime, or any date/time. Scheduling is done by the backend.
- Output only: title, description, topic, duration (minutes), priority, difficulty for each task.

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

        # Initialize LLM
        llm = get_langchain_llm()
        
        # Generate plan with AI
        logger.info("Calling LLM for study plan generation...")
        response = llm.invoke(prompt)
        
        # Extract content from AIMessage (LangChain returns AIMessage, not string)
        if hasattr(response, 'content'):
            response_text = response.content.strip()
        else:
            response_text = str(response).strip()
        
        logger.info(f"LLM response length: {len(response_text)} chars")
        
        # Extract and parse JSON with multiple strategies
        ai_plan = None
        parse_errors = []
        
        # Strategy 1: Direct JSON parse
        try:
            ai_plan = json.loads(response_text)
            logger.info("✓ Parsed JSON directly")
        except json.JSONDecodeError as e:
            parse_errors.append(f"Direct parse: {e}")
        
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
            except (json.JSONDecodeError, IndexError) as e:
                parse_errors.append(f"Markdown removal: {e}")
        
        # Strategy 3: Extract JSON from text
        if not ai_plan:
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx + 1]
                    ai_plan = json.loads(json_str)
                    logger.info("✓ Extracted JSON from response")
            except (json.JSONDecodeError, ValueError) as e:
                parse_errors.append(f"JSON extraction: {e}")
        
        # Strategy 4: Fallback to programmatic generation
        if not ai_plan:
            logger.warning(f"All JSON parsing failed. Errors: {parse_errors}")
            logger.warning(f"Response preview: {response_text[:300]}")
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
                calculated_duration = int((request.dailyStudyHours * 60) // (len(expanded_tasks) / num_days))
                ai_plan["tasks"].append({
                    "title": task_info["title"],
                    "description": task_info["description"],
                    "topic": task_info["topic"],
                    "duration": validate_task_duration(calculated_duration),
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
                "tasks": [{"title": f"Learn {topic}", "description": f"Study {topic} concepts", "topic": topic, "duration": validate_task_duration(60), "priority": "medium", "difficulty": request.difficulty} for topic in request.topics],
                "recommendations": ["Study consistently", "Practice regularly", "Review often"]
            }
        
        logger.info(f"✓ AI plan has {len(ai_plan.get('tasks', []))} tasks")
        
        # ========================================================================
        # SCHEMA VALIDATION: strip any AI timestamps; only task metadata is used
        # ========================================================================
        raw_tasks = validate_and_coerce_tasks(
            ai_plan.get("tasks", []),
            request.topics,
            request.difficulty,
        )
        if not raw_tasks:
            raw_tasks = [
                {"title": f"Learn {t}", "description": f"Study {t}", "topic": t, "duration": 60, "priority": "medium", "difficulty": request.difficulty, "resources": []}
                for t in request.topics
            ]
        ai_plan["tasks"] = raw_tasks
        
        # ========================================================================
        # PHASE 2: APPLY COGNITIVE LOAD PROTECTION
        # ========================================================================
        
        # Apply cognitive load limits based on strategy context
        if strategy_context or exam_strategy:
            logger.info("🧠 Applying cognitive load protection...")
            context_for_limits = {
                'recommendedMode': strategy_context.get('recommendedMode') if strategy_context else 'balanced',
                'phaseConfig': exam_strategy if exam_strategy else {}
            }
            filtered_tasks = apply_cognitive_load_limits(raw_tasks, num_days, context_for_limits)
        else:
            # Apply default cognitive load limits
            filtered_tasks = apply_cognitive_load_limits(raw_tasks, num_days, None)
        
        logger.info(f"✓ Cognitive load applied: {len(raw_tasks)} → {len(filtered_tasks)} tasks")
        
        # Assign dates to tasks
        tasks_with_dates = []
        tasks_per_day = max(1, len(filtered_tasks) // num_days)
        
        for i, task in enumerate(filtered_tasks):
            day_offset = i // tasks_per_day
            if day_offset >= num_days:
                day_offset = num_days - 1
            
            task_date = start_date + timedelta(days=day_offset)
            
            # Validate duration to ensure it meets MongoDB constraints (15-480 minutes)
            raw_duration = task.get("duration", 60)
            validated_duration = validate_task_duration(raw_duration)
            
            tasks_with_dates.append(TaskGenerated(
                title=task.get("title", f"Study Session {i+1}"),
                description=task.get("description", "Study and practice"),
                topic=task.get("topic", request.topics[i % len(request.topics)]),
                scheduledDate=task_date.isoformat(),
                duration=validated_duration,
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
        
        # Add strategy-specific recommendations
        if strategy_context:
            mode = strategy_context.get('recommendedMode')
            if mode == 'emergency':
                raw_recommendations.insert(0, "⚠️ Emergency mode active: Focus on high-priority topics only")
                raw_recommendations.insert(1, "🔥 Use hyper time blocks (90-120 min) for intensive study")
            elif mode == 'adaptive':
                if exam_strategy:
                    phase = exam_strategy.get('currentPhase', '')
                    raw_recommendations.insert(0, f"📚 {phase.replace('_', ' ').title()} phase: Adjust study intensity accordingly")
        
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
