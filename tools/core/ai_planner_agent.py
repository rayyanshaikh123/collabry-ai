"""
AI Planner Agent - Real Calendar-Based Session Scheduler

This agent generates TRUE TIME-SLOT BASED study sessions, not date-based tasks.

KEY DIFFERENCE FROM OLD generate_study_plan:
- OLD: Returns {scheduledDate, duration} → ambiguous timing
- NEW: Returns {startTime, endTime} → precise calendar blocks

ARCHITECTURE FLOW:
1. Frontend sends plan request
2. Backend injects context (user profile, existing calendar, constraints)
3. Backend queries slot engine for available time blocks
4. AI Agent maps topics → time slots with cognitive constraints
5. Constraint validator ensures schedule viability
6. Sessions persisted as StudyEvent (not StudyTask)
7. Frontend renders real calendar with drag-to-reschedule

This eliminates the "stacking durations on dates" anti-pattern.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import asyncio
from core.llm import get_langchain_llm
from core.backend_client import get_backend_client

logger = logging.getLogger(__name__)


class AIPlannerAgent:
    """
    Intelligent planner that generates TIME-SLOT SESSIONS not DATE TASKS.
    
    Uses interval scheduling algorithms, not naive sequential stacking.
    """
    
    def __init__(self):
        self.llm = get_langchain_llm()
        self.backend_client = get_backend_client()
        
    async def generate_smart_schedule(
        self,
        user_id: str,
        plan_request: Dict[str, Any],
        auth_token: str
    ) -> Dict[str, Any]:
        """
        Generate intelligent calendar-based schedule.
        
        Pipeline:
        1. Collect context (profile, calendar, constraints)
        2. Find available slots (slot engine)
        3. AI maps topics → slots
        4. Validate constraints
        5. Return sessions with startTime/endTime
        
        Args:
            user_id: User identifier
            plan_request: {
                planId: str
                subject: str
                topics: List[str]
                startDate: ISO string
                endDate: ISO string
                dailyStudyHours: float
                examDate: Optional[ISO string]
                preferences: {
                    preferredTimes: List[str]  # morning, afternoon, evening, night
                    maxSessionDuration: int  # minutes
                    breakFrequency: int  # minutes between breaks
                    focusType: str  # deep_work, mixed, light
                }
            }
            auth_token: JWT for backend API calls
            
        Returns:
            {
                "sessions": [
                    {
                        "title": str,
                        "description": str,
                        "topic": str,
                        "startTime": "2026-02-13T09:00:00Z",
                        "endTime": "2026-02-13T10:30:00Z",
                        "type": "deep_work" | "light" | "review",
                        "difficulty": "easy" | "medium" | "hard",
                        "priority": "low" | "medium" | "high" | "urgent",
                        "deepWork": bool,
                        "estimatedEffort": int (1-10 scale)
                    }
                ],
                "statistics": {
                    "totalSessions": int,
                    "totalHours": float,
                    "avgSessionDuration": int,
                    "deepWorkSessions": int,
                    "coverageByTopic": Dict[str, int]
                },
                "warnings": List[str],
                "recommendations": List[str]
            }
        """
        try:
            logger.info(f"🎯 Starting smart schedule generation for user {user_id}")
            
            # ================================================================
            # PHASE 1: CONTEXT COLLECTION
            # ================================================================
            logger.info("📊 Phase 1: Collecting student context...")
            
            context_response = await self.backend_client.post(
                "/api/planner/scheduler/context",
                {
                    "planInput": {
                        "subject": plan_request["subject"],
                        "topics": plan_request["topics"],
                        "examDate": plan_request.get("examDate"),
                        "difficulty": plan_request.get("difficulty", "medium")
                    }
                },
                auth_token
            )
            
            if not context_response or "error" in context_response:
                logger.warning("Context collection failed, using minimal context")
                context = self._generate_fallback_context(plan_request)
            else:
                context = context_response
                logger.info(f"✓ Context collected: {len(context.get('topics', []))} topics analyzed")
            
            # ================================================================
            # PHASE 2: SLOT DISCOVERY
            # ================================================================
            logger.info("🔍 Phase 2: Finding available time slots...")
            
            # Parse dates
            start_date = datetime.fromisoformat(plan_request["startDate"].replace("Z", "+00:00"))
            end_date = datetime.fromisoformat(plan_request["endDate"].replace("Z", "+00:00"))
            
            # Map preferred times to time ranges
            time_preferences = plan_request.get("preferences", {}).get("preferredTimes", ["morning", "afternoon"])
            preferred_slots = self._map_preferences_to_slots(time_preferences)
            
            # Query slot engine
            slots_response = await self.backend_client.post(
                "/api/planner/scheduler/slots",
                {
                    "startDate": start_date.isoformat(),
                    "endDate": end_date.isoformat(),
                    "dailyStudyHours": plan_request.get("dailyStudyHours", 2),
                    "preferredTimeSlots": preferred_slots,
                    "sleepSchedule": plan_request.get("preferences", {}).get("sleepSchedule", {
                        "start": "23:00",
                        "end": "07:00"
                    })
                },
                auth_token
            )
            
            if not slots_response or "error" in slots_response:
                logger.warning("Slot engine failed, generating programmatic slots")
                available_slots = self._generate_fallback_slots(start_date, end_date, plan_request["dailyStudyHours"])
            else:
                available_slots = slots_response.get("slots", [])
                logger.info(f"✓ Found {len(available_slots)} available slots")
            
            # ================================================================
            # PHASE 3: AI SESSION GENERATION
            # ================================================================
            logger.info("🤖 Phase 3: AI generating time-slot sessions...")
            
            sessions = await self._generate_sessions_with_ai(
                plan_request,
                context,
                available_slots,
                auth_token
            )
            
            logger.info(f"✓ Generated {len(sessions)} sessions")
            
            # ================================================================
            # PHASE 4: CONSTRAINT VALIDATION
            # ================================================================
            logger.info("✅ Phase 4: Validating schedule constraints...")
            
            validation_response = await self.backend_client.post(
                "/api/planner/scheduler/validate",
                {
                    "schedule": sessions,
                    "studentContext": context,
                    "dailyHoursLimit": plan_request.get("dailyStudyHours", 2) * 1.2,  # 20% buffer
                    "examDate": plan_request.get("examDate")
                },
                auth_token
            )
            
            if validation_response and validation_response.get("valid"):
                logger.info(f"✓ Schedule validated (score: {validation_response.get('score', 0)})")
                warnings = validation_response.get("warnings", [])
            else:
                logger.warning("Validation failed or returned warnings")
                warnings = validation_response.get("violations", []) if validation_response else ["Schedule may have constraint violations"]
            
            # ================================================================
            # PHASE 5: COMPUTE STATISTICS & RECOMMENDATIONS
            # ================================================================
            statistics = self._compute_statistics(sessions)
            recommendations = self._generate_recommendations(sessions, context, validation_response)
            
            result = {
                "sessions": sessions,
                "statistics": statistics,
                "warnings": warnings,
                "recommendations": recommendations,
                "metadata": {
                    "generatedAt": datetime.utcnow().isoformat(),
                    "userId": user_id,
                    "planId": plan_request.get("planId"),
                    "subject": plan_request["subject"]
                }
            }
            
            logger.info(f"✅ Smart schedule generation complete: {len(sessions)} sessions, "
                       f"{statistics['totalHours']:.1f} hours, {len(warnings)} warnings")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Smart schedule generation failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "sessions": [],
                "statistics": {},
                "warnings": [f"Generation failed: {str(e)}"],
                "recommendations": ["Please try again or contact support"]
            }
    
    async def _generate_sessions_with_ai(
        self,
        plan_request: Dict[str, Any],
        context: Dict[str, Any],
        available_slots: List[Dict[str, Any]],
        auth_token: str
    ) -> List[Dict[str, Any]]:
        """
        Use AI to intelligently map topics to time slots.
        
        CRITICAL: AI MUST output startTime/endTime, NOT date+duration
        """
        
        # Calculate dynamic temperature based on exam urgency
        exam_date = plan_request.get("examDate")
        temperature = self._calculate_temperature(exam_date, plan_request["startDate"])
        
        # Build AI prompt with EXPLICIT TIME-SLOT requirements
        prompt = self._build_session_prompt(plan_request, context, available_slots)
        
        # Call LLM with retry and fallback strategies
        sessions = None
        parse_strategies = [
            self._parse_direct_json,
            self._parse_markdown_json,
            self._parse_regex_json,
            lambda x: self._parse_with_retry(x, prompt, temperature)
        ]
        
        logger.info(f"Calling LLM (temp={temperature:.2f})...")
        response = self.llm.invoke(prompt, config={"temperature": temperature})
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"LLM response: {len(response_text)} chars")
        
        # Try each parsing strategy
        for i, strategy in enumerate(parse_strategies):
            try:
                sessions = strategy(response_text)
                if sessions and len(sessions) > 0:
                    logger.info(f"✓ Parsed sessions using strategy {i+1}")
                    break
            except Exception as e:
                logger.warning(f"Parse strategy {i+1} failed: {e}")
                continue
        
        # Fallback: Programmatic generation if all strategies fail
        if not sessions or len(sessions) == 0:
            logger.warning("All AI parsing strategies failed, using intelligent fallback")
            sessions = self._generate_programmatic_sessions(plan_request, available_slots)
        
        # Post-process: Ensure all sessions have startTime/endTime
        validated_sessions = []
        for session in sessions:
            if "startTime" in session and "endTime" in session:
                validated_sessions.append(session)
            else:
                logger.warning(f"Session missing time fields: {session.get('title', 'Unknown')}")
        
        return validated_sessions
    
    def _build_session_prompt(
        self,
        plan_request: Dict[str, Any],
        context: Dict[str, Any],
        available_slots: List[Dict[str, Any]]
    ) -> str:
        """
        Build AI prompt that FORCES time-slot output.
        
        The prompt MUST explicitly reject date+duration format.
        """
        
        # Extract key info
        subject = plan_request["subject"]
        topics = plan_request["topics"]
        daily_hours = plan_request.get("dailyStudyHours", 2)
        difficulty = plan_request.get("difficulty", "medium")
        exam_date = plan_request.get("examDate")
        
        # Format available slots for AI
        slots_summary = self._format_slots_for_prompt(available_slots[:20])  # Show first 20 slots
        
        # Context summary
        context_summary = f"""
STUDENT PROFILE:
- Learning pace: {context.get('learningPace', 'medium')}
- Past performance: {context.get('averageScore', 'N/A')}/100
- Completion rate: {context.get('completionRate', 'N/A')}%
- Available daily: {daily_hours} hours
"""
        
        exam_context = ""
        if exam_date:
            exam_context = f"""
⚠️ EXAM MODE:
- Exam date: {exam_date}
- PRIORITIZE: Core concepts, practice problems, review sessions
- URGENCY: High-priority topics FIRST
"""
        
        prompt = f"""You are an elite AI scheduling assistant. You do NOT generate task lists with durations. You generate PRECISE CALENDAR SESSIONS with exact start and end times.

❌ FORBIDDEN OUTPUT FORMAT:
{{
  "scheduledDate": "2026-02-13",
  "duration": 90
}}

✅ REQUIRED OUTPUT FORMAT:
{{
  "title": "Master Arrays & Pointers",
  "description": "Deep dive into array manipulation and pointer arithmetic",
  "topic": "Data Structures",
  "startTime": "2026-02-13T09:00:00Z",
  "endTime": "2026-02-13T10:30:00Z",
  "type": "deep_work",
  "difficulty": "medium",
  "priority": "high",
  "deepWork": true,
  "estimatedEffort": 8
}}

ASSIGNMENT:
Subject: {subject}
Topics to cover: {', '.join(topics)}
Difficulty: {difficulty}

{context_summary}
{exam_context}

AVAILABLE TIME SLOTS (you MUST use these):
{slots_summary}

CONSTRAINTS:
✓ Use ONLY slots from available list
✓ Never exceed slot boundaries
✓ Deep work sessions: 90-120 minutes max
✓ Light sessions: 30-60 minutes
✓ Max 2 hard-difficulty sessions per day
✓ Distribute topics evenly
✓ Balance cognitive load

SESSION TYPES:
- deep_work: 90-120 min, focused study, no distractions
- practice: 60-90 min, hands-on exercises
- review: 30-60 min, consolidation
- light: 30-45 min, easy content

OUTPUT (JSON ONLY, NO MARKDOWN):
{{
  "sessions": [
    {{
      "title": "Session Name",
      "description": "What will be studied",
      "topic": "Topic name from list",
      "startTime": "YYYY-MM-DDTHH:MM:SSZ",
      "endTime": "YYYY-MM-DDTHH:MM:SSZ",
      "type": "deep_work|practice|review|light",
      "difficulty": "easy|medium|hard",
      "priority": "low|medium|high|urgent",
      "deepWork": boolean,
      "estimatedEffort": 1-10
    }}
  ]
}}

Generate {min(len(available_slots), len(topics) * 3)} sessions now:"""
        
        return prompt
    
    def _format_slots_for_prompt(self, slots: List[Dict[str, Any]]) -> str:
        """Format available slots in human-readable format for AI"""
        if not slots:
            return "No predefined slots available - generate from scratch"
        
        formatted = []
        for i, slot in enumerate(slots[:15]):  # Limit to prevent token overflow
            start = slot.get("startTime", "")
            end = slot.get("endTime", "")
            duration = slot.get("durationMinutes", "")
            quality = slot.get("quality", 0)
            
            formatted.append(f"{i+1}. {start} → {end} ({duration} min, quality: {quality})")
        
        return "\n".join(formatted)
    
    def _calculate_temperature(self, exam_date: Optional[str], start_date: str) -> float:
        """
        Calculate LLM temperature based on exam urgency.
        
        - No exam: 0.70 (balanced creativity)
        - Normal timeline: 0.65 (slightly more focused)
        - Exam approaching: 0.60-0.65 (more deterministic)
        - Crisis mode: 0.55 (very focused)
        """
        if not exam_date:
            return 0.70
        
        try:
            exam_dt = datetime.fromisoformat(exam_date.replace("Z", "+00:00"))
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            days_until_exam = (exam_dt - start_dt).days
            
            if days_until_exam <= 7:
                return 0.55  # Crisis mode
            elif days_until_exam <= 14:
                return 0.60  # Urgent
            elif days_until_exam <= 30:
                return 0.65  # Moderate urgency
            else:
                return 0.70  # Normal
        except:
            return 0.70
    
    # ========================================================================
    # PARSING STRATEGIES (4-tier fallback)
    # ========================================================================
    
    def _parse_direct_json(self, text: str) -> List[Dict[str, Any]]:
        """Strategy 1: Direct JSON parse"""
        data = json.loads(text)
        return data.get("sessions", [])
    
    def _parse_markdown_json(self, text: str) -> List[Dict[str, Any]]:
        """Strategy 2: Extract from markdown code blocks"""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        data = json.loads(text)
        return data.get("sessions", [])
    
    def _parse_regex_json(self, text: str) -> List[Dict[str, Any]]:
        """Strategy 3: Regex extraction"""
        import re
        json_match = re.search(r'\{.*"sessions".*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return data.get("sessions", [])
        return []
    
    async def _parse_with_retry(self, original_response: str, original_prompt: str, temperature: float) -> List[Dict[str, Any]]:
        """Strategy 4: Retry with explicit instructions"""
        retry_prompt = f"""The previous response was not valid JSON. Here it is:

{original_response[:500]}

Please provide ONLY valid JSON with this structure:
{{
  "sessions": [
    {{
      "title": "...",
      "startTime": "2026-02-13T09:00:00Z",
      "endTime": "2026-02-13T10:30:00Z",
      ...
    }}
  ]
}}

NO MARKDOWN, NO EXTRA TEXT. ONLY JSON:"""
        
        response = self.llm.invoke(retry_prompt, config={"temperature": 0.3})  # Lower temp for retry
        text = response.content if hasattr(response, 'content') else str(response)
        
        return self._parse_direct_json(text)
    
    # ========================================================================
    # FALLBACK GENERATION
    # ========================================================================
    
    def _generate_programmatic_sessions(
        self,
        plan_request: Dict[str, Any],
        available_slots: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Intelligent programmatic session generation when AI fails.
        
        Uses interval scheduling algorithm to optimally assign topics to slots.
        """
        logger.info("Using programmatic session generation")
        
        topics = plan_request["topics"]
        difficulty = plan_request.get("difficulty", "medium")
        
        sessions = []
        slots_used = 0
        
        # Cycle through topics and assign to slots
        for i, slot in enumerate(available_slots):
            if slots_used >= len(topics) * 3:  # Limit sessions per topic
                break
            
            topic_index = i % len(topics)
            topic = topics[topic_index]
            
            # Determine session type based on slot quality
            quality = slot.get("quality", 50)
            if quality >= 80:
                session_type = "deep_work"
                difficulty_level = difficulty
            elif quality >= 60:
                session_type = "practice"
                difficulty_level = "medium"
            else:
                session_type = "review"
                difficulty_level = "easy"
            
            session = {
                "title": f"Study {topic}" if i % 3 == 0 else f"Practice {topic}" if i % 3 == 1 else f"Review {topic}",
                "description": f"{'Deep dive' if session_type == 'deep_work' else 'Practice' if session_type == 'practice' else 'Review'} {topic} concepts",
                "topic": topic,
                "startTime": slot["startTime"],
                "endTime": slot["endTime"],
                "type": session_type,
                "difficulty": difficulty_level,
                "priority": "high" if i < len(topics) else "medium",
                "deepWork": session_type == "deep_work",
                "estimatedEffort": 8 if session_type == "deep_work" else 6 if session_type == "practice" else 4
            }
            
            sessions.append(session)
            slots_used += 1
        
        logger.info(f"Generated {len(sessions)} programmatic sessions")
        return sessions
    
    def _generate_fallback_slots(
        self,
        start_date: datetime,
        end_date: datetime,
        daily_hours: float
    ) -> List[Dict[str, Any]]:
        """Generate default time slots when slot engine fails"""
        slots = []
        current_date = start_date
        
        while current_date <= end_date:
            # Morning slot: 9am-11am
            morning_start = current_date.replace(hour=9, minute=0, second=0)
            morning_end = morning_start + timedelta(hours=2)
            slots.append({
                "startTime": morning_start.isoformat(),
                "endTime": morning_end.isoformat(),
                "durationMinutes": 120,
                "quality": 85,
                "deepWork": True
            })
            
            # Afternoon slot: 2pm-4pm
            afternoon_start = current_date.replace(hour=14, minute=0, second=0)
            afternoon_end = afternoon_start + timedelta(hours=2)
            slots.append({
                "startTime": afternoon_start.isoformat(),
                "endTime": afternoon_end.isoformat(),
                "durationMinutes": 120,
                "quality": 75,
                "deepWork": True
            })
            
            # Evening slot: 7pm-9pm (if daily_hours > 4)
            if daily_hours > 4:
                evening_start = current_date.replace(hour=19, minute=0, second=0)
                evening_end = evening_start + timedelta(hours=2)
                slots.append({
                    "startTime": evening_start.isoformat(),
                    "endTime": evening_end.isoformat(),
                    "durationMinutes": 120,
                    "quality": 65,
                    "deepWork": False
                })
            
            current_date += timedelta(days=1)
        
        return slots
    
    def _generate_fallback_context(self, plan_request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate minimal context when backend fails"""
        return {
            "learningPace": "medium",
            "averageScore": 75,
            "completionRate": 80,
            "topics": [{"name": t, "estimatedHours": 3} for t in plan_request["topics"]]
        }
    
    def _map_preferences_to_slots(self, preferred_times: List[str]) -> List[Dict[str, Any]]:
        """Map user time preferences to slot configuration"""
        slot_mapping = {
            "morning": {"start": "06:00", "end": "12:00"},
            "afternoon": {"start": "12:00", "end": "18:00"},
            "evening": {"start": "18:00", "end": "23:00"},
            "night": {"start": "23:00", "end": "02:00"}
        }
        
        return [slot_mapping.get(t, slot_mapping["morning"]) for t in preferred_times]
    
    def _compute_statistics(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute session statistics"""
        if not sessions:
            return {}
        
        total_minutes = 0
        deep_work_count = 0
        topic_coverage = {}
        
        for session in sessions:
            # Calculate duration
            try:
                start = datetime.fromisoformat(session["startTime"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(session["endTime"].replace("Z", "+00:00"))
                duration = (end - start).total_seconds() / 60
                total_minutes += duration
            except:
                pass
            
            # Count deep work
            if session.get("deepWork", False):
                deep_work_count += 1
            
            # Track topic coverage
            topic = session.get("topic", "Unknown")
            topic_coverage[topic] = topic_coverage.get(topic, 0) + 1
        
        return {
            "totalSessions": len(sessions),
            "totalHours": round(total_minutes / 60, 1),
            "avgSessionDuration": int(total_minutes / len(sessions)) if sessions else 0,
            "deepWorkSessions": deep_work_count,
            "coverageByTopic": topic_coverage
        }
    
    def _generate_recommendations(
        self,
        sessions: List[Dict[str, Any]],
        context: Dict[str, Any],
        validation: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Based on session distribution
        deep_work_count = sum(1 for s in sessions if s.get("deepWork", False))
        if deep_work_count < len(sessions) * 0.3:
            recommendations.append("⚠️ Consider adding more deep work sessions (90-120 min) for complex topics")
        
        # Based on validation
        if validation and not validation.get("valid", True):
            recommendations.append("⚠️ Schedule has constraint violations - consider spreading sessions over more days")
        
        # Based on context
        completion_rate = context.get("completionRate", 100)
        if completion_rate < 70:
            recommendations.append("💡 Your completion rate is low - try shorter, more achievable sessions")
        
        # General recommendations
        recommendations.extend([
            "🎯 Start with fundamentals before advanced topics",
            "⏰ Use Pomodoro: 25 min focus + 5 min break",
            "📝 Review previous session before starting new topics",
            "🎮 Gamify: Track streaks and celebrate milestones"
        ])
        
        return recommendations


# Global instance
_agent_instance = None

def get_planner_agent() -> AIPlannerAgent:
    """Get singleton planner agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AIPlannerAgent()
    return _agent_instance
