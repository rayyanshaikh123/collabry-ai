"""
Generate Study Plan Tool - Create structured study schedules.

This tool generates personalized study plans with daily task breakdown
based on the user's goals and timeline.
"""

import json
from typing import Optional, List, Any
from datetime import datetime, timedelta
from langchain_core.tools import tool
from core.llm import get_async_openai_client, get_llm_config
from rag.retriever import get_retriever


@tool
async def generate_study_plan(
    subject: Optional[str] = "Personalized Study Session",
    topics: Optional[str] = "main concepts and key terms",
    user_id: str = "default",
    duration_days: Any = 7,
    daily_hours: Any = 2.0,
    difficulty: str = "intermediate",
    notebook_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
) -> str:
    """
    Generate a structured study plan with daily tasks and milestones.
    
    Use this tool when the user wants to:
    - Create a study schedule
    - Plan their learning journey
    - Break down a large topic into manageable tasks
    
    Examples of when to use:
    - "Create a study plan for learning Python in 2 weeks"
    - "Help me plan my exam preparation for calculus"
    - "Make a schedule to learn Spanish over the next month"
    
    Args:
        subject: The subject or course name
        topics: Comma-separated list of topics to cover (or single topic description)
        user_id: User identifier (injected by agent)
        duration_days: Number of days for the study plan (default 7)
        daily_hours: Hours available per day (default 2.0)
        difficulty: Learning difficulty level: beginner, intermediate, advanced
        notebook_id: Optional notebook context
    
    Returns:
        JSON study plan with daily tasks:
        {
          "title": "Study Plan Title",
          "subject": "Subject Name",
          "totalDays": 7,
          "dailyHours": 2.0,
          "tasks": [
            {
              "day": 1,
              "date": "2024-01-01",
              "title": "Task Title",
              "description": "What to do",
              "duration": 120,
              "topics": ["topic1"],
              "goals": ["goal1", "goal2"]
            }
          ],
          "milestones": [...],
          "recommendations": [...]
        }
    """
    try:
        # Parse topics
        topic_list = [t.strip() for t in topics.split(',') if t.strip()]
        if not topic_list:
            topic_list = [topics]
        
        # Validate and robustly parse inputs
        try:
            import re
            
            # Parse duration_days
            if isinstance(duration_days, str):
                match = re.search(r'\d+', duration_days)
                duration_days = int(match.group()) if match else 7
            else:
                duration_days = int(duration_days)
                
            # Parse daily_hours
            if isinstance(daily_hours, str):
                match = re.search(r'\d+(\.\d+)?', daily_hours)
                daily_hours = float(match.group()) if match else 2.0
            else:
                daily_hours = float(daily_hours)
        except Exception:
            duration_days = 7
            daily_hours = 2.0

        duration_days = max(1, min(duration_days, 90))  # Limit 1-90 days
        daily_hours = max(0.5, min(daily_hours, 12))  # Limit 0.5-12 hours
        
        # Calculate start and end dates
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days - 1)
        
        # Get context from sources if available
        context = ""
        if notebook_id:
            try:
                retriever = get_retriever(user_id=user_id, notebook_id=notebook_id, source_ids=source_ids)
                query = f"{subject} {' '.join(topic_list[:3])}"
                docs = retriever.invoke(query)
                if docs:
                    context = "\n\n".join([doc.page_content for doc in docs[:2]])
            except Exception:
                context = ""
        
        # Get OpenAI client
        client = get_async_openai_client()
        
        # Build context prompt section (fix for f-string backslash issue)
        context_prompt = ""
        if context:
            context_prompt = f"Reference material from student's notes:\n{context}\n\n"
        
        # Create prompt for study plan
        prompt = f"""Generate a detailed study plan for:

Subject: {subject}
Topics: {', '.join(topic_list)}
Duration: {duration_days} days
Daily Study Time: {daily_hours} hours
Difficulty Level: {difficulty}

{context_prompt}Create a JSON study plan with this structure:
{{
  "title": "Descriptive plan title",
  "subject": "{subject}",
  "totalDays": {duration_days},
  "dailyHours": {daily_hours},
  "difficulty": "{difficulty}",
  "tasks": [
    {{
      "day": 1,
      "date": "{start_date.strftime('%Y-%m-%d')}",
      "title": "Day 1 task title",
      "description": "Detailed description of what to study and do",
      "duration": 120,
      "topics": ["topic covered this day"],
      "goals": ["specific learning goal 1", "specific learning goal 2"],
      "resources": ["suggested resource or activity"]
    }}
  ],
  "milestones": [
    {{
      "day": 3,
      "title": "Milestone title",
      "description": "What should be achieved"
    }}
  ],
  "recommendations": [
    "Study tip 1",
    "Study tip 2"
  ],
  "assessments": [
    {{
      "day": 7,
      "type": "quiz",
      "description": "Test your knowledge on topics 1-3"
    }}
  ]
}}

Guidelines:
- Distribute topics evenly across days
- Start with fundamentals, build to advanced concepts
- Include review days for longer plans
- Add milestones every 3-5 days
- Ensure total daily duration matches daily_hours * 60 minutes
- Be specific about what to study each day
- Include practical exercises and assessments

Return ONLY the JSON, no additional text."""

        # Generate study plan
        llm_config = get_llm_config()
        response = await client.chat.completions.create(
            model=llm_config.model,
            messages=[
                {"role": "system", "content": "You are an expert educational planner. Create realistic, well-structured study plans. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2500
        )
        
        # Extract and validate JSON
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        # Validate JSON
        plan_data = json.loads(content)
        
        # Ensure required structure
        if "title" not in plan_data:
            plan_data["title"] = f"{subject} Study Plan"
        if "tasks" not in plan_data or not isinstance(plan_data["tasks"], list):
            return json.dumps({
                "error": "Invalid study plan structure",
                "message": "Generated plan missing 'tasks' array"
            })
        
        return json.dumps(plan_data, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": "Failed to parse study plan JSON",
            "message": str(e),
            "subject": subject
        })
    except Exception as e:
        return json.dumps({
            "error": "Failed to generate study plan",
            "message": str(e),
            "subject": subject
        })
