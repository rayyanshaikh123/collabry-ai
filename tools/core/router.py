from typing import Optional, Dict, Any, List, Tuple
import json
import re
from core.llm import chat_completion
from core.session_state import SessionTaskState
# SECURITY FIX - Phase 2: Add Pydantic for schema validation
from pydantic import BaseModel, validator, ValidationError
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# SECURITY FIX - Phase 2: Strict schema validation for router output

class ActionType(str, Enum):
    START_TASK = "START_TASK"
    MODIFY_PARAM = "MODIFY_PARAM"  
    ASK_ARTIFACT = "ASK_ARTIFACT"
    ANSWER_GENERAL = "ANSWER_GENERAL"
    CLARIFY = "CLARIFY"

class TaskType(str, Enum):
    QUIZ = "QUIZ"
    FLASHCARDS = "FLASHCARDS"
    MINDMAP = "MINDMAP"
    SUMMARY = "SUMMARY"
    STUDY_PLAN = "STUDY_PLAN"
    COURSE_SEARCH = "COURSE_SEARCH"
    NONE = "NONE"

class RetrievalPolicy(str, Enum):
    STRICT_SELECTED = "STRICT_SELECTED"
    PREFER_SELECTED = "PREFER_SELECTED" 
    AUTO_EXPAND = "AUTO_EXPAND"
    GLOBAL = "GLOBAL"

class RetrievalMode(str, Enum):
    CHUNK_SEARCH = "CHUNK_SEARCH"
    FULL_DOCUMENT = "FULL_DOCUMENT"
    MULTI_DOC_SYNTHESIS = "MULTI_DOC_SYNTHESIS"
    NONE = "NONE"

class RouterResponse(BaseModel):
    """SECURITY FIX: Validated router response schema."""
    action: ActionType
    task: TaskType = TaskType.NONE
    retrieval_policy: RetrievalPolicy = RetrievalPolicy.GLOBAL
    retrieval_mode: RetrievalMode = RetrievalMode.NONE
    param_updates: Optional[Dict[str, Any]] = {}
    thought: Optional[str] = ""
    
    @validator('param_updates')
    def validate_param_updates(cls, v):
        """SECURITY FIX: Strict parameter validation."""
        if not v:
            return {}
            
        safe_params = {}
        safe_keys = ['topic', 'difficulty', 'count', 'subject', 'topics', 'num_questions', 'num_cards']
        
        for key, value in v.items():
            if key not in safe_keys:
                continue
                
            # Only allow safe value types
            if isinstance(value, (str, int, float, bool)) or value is None:
                if isinstance(value, str):
                    # Sanitize strings - remove dangerous characters
                    value = re.sub(r'[<>{}();\'"\\$]', '', value)
                    value = value[:200]  # Limit length
                    
                    # Reject MongoDB operators and SQL injection attempts
                    dangerous_patterns = ['$ne', '$gt', '$lt', '$in', '$or', '$and', 'drop table', 'select from']
                    if any(pattern in value.lower() for pattern in dangerous_patterns):
                        continue
                        
                safe_params[key] = value
                
        return safe_params

def _detect_ambiguous_references(message: str, session_state: SessionTaskState) -> bool:
    """
    SECURITY FIX - Phase 3: Detect ambiguous references that require clarification.
    """
    message_lower = message.lower().strip()
    
    # Check for vague pronouns without clear antecedents
    vague_patterns = [
        r'\bit\b(?!\s+(is|was|will|should|would))',  # "it" not followed by verbs
        r'\bthat\s+(thing|one|stuff|issue|problem)',  # "that thing"
        r'\bthose\s+(things|ones)',  # "those things"
        r'\bthis\s+(thing|one|stuff|issue)',  # "this thing"
        r'\bthe\s+(previous|last|earlier|above)\s+(one|thing)',  # "the previous one"
        r'\bwhat\s+we\s+(did|discussed|talked about)\s+(before|earlier)',  # "what we did before"
        r'\bhelp\s+with\s+(it|that|this)$',  # "help with it" 
        r'\bmore\s+(of|about)\s+(it|that|this)',  # "more of it"
        r'\bchange\s+(it|that|this)',  # "change it"
        r'\bfix\s+(it|that|this)',  # "fix it" 
    ]
    
    for pattern in vague_patterns:
        if re.search(pattern, message_lower):
            logger.debug(f"🤔 Detected ambiguous reference: {pattern}")
            return True
    
    # Check for incomplete sentences that need clarification
    incomplete_patterns = [
        r'^(make|create|generate|do|start|begin)\s+(a|an|the)?\s*$',  # "make a"
        r'^(about|on|for)\s+$',  # "about"
        r'^(yes|no|maybe),?\s*$',  # Single word responses without context
        r'^(more|less|bigger|smaller|harder|easier)\s*$',  # Relative terms without reference
    ]
    
    for pattern in incomplete_patterns:
        if re.search(pattern, message_lower):
            logger.debug(f"🤔 Detected incomplete request: {pattern}")
            return True
    
    # Check for follow-up questions without sufficient context
    if not session_state.active_task:
        # No active task context
        followup_patterns = [
            r'^(and|also|plus|additionally)',  # Starting with conjunctions
            r'\bquestion\s+\d+',  # "question 3" without task context
            r'\bcard\s+\d+',  # "card 2" without task context  
            r'\bmake\s+it\s+(easier|harder|longer|shorter)',  # Modifications without context
        ]
        
        for pattern in followup_patterns:
            if re.search(pattern, message_lower):
                logger.debug(f"🤔 Detected follow-up without context: {pattern}")
                return True
    
    return False

ROUTER_PROMPT = """You are the CONTROL PLANNER of a Study Assistant system.

Your job is to decide what the system should do and how information must be retrieved.
You DO NOT answer the user. You ONLY decide the plan.

---
TASK ACTIONS

START_TASK: Begin a new tool-based task.
MODIFY_PARAM: Tweak parameters of an existing task (difficulty, count).
ASK_ARTIFACT: User is asking about a specific item in a quiz/mindmap/report.
ANSWER_GENERAL: Small-talk or general questions not needing sources.
CLARIFY: Ask the user for more info if needed.

---
TASK TYPES
QUIZ, FLASHCARDS, MINDMAP, SUMMARY, STUDY_PLAN, COURSE_SEARCH, NONE

---
RETRIEVAL POLICY (AUTHORITY)

STRICT_SELECTED: Answer must come ONLY from selected sources.
PREFER_SELECTED: Use selected sources first, fallback to notebook if needed.
AUTO_EXPAND: Search across all user knowledge.
GLOBAL: General info; ignore notebook sources.

---
RETRIEVAL MODE (HOW TO READ DATA)

CHUNK_SEARCH: Semantic search over chunks. (Used for specific detail questions)
FULL_DOCUMENT: Process the entire raw source. (Used for summary, recap, overview)
MULTI_DOC_SYNTHESIS: Combine info across >1 source. (Used for comparisons)
NONE: No retrieval needed.

---
SESSION STATE:
Active Task: %s
Current Topic: %s
Selected Sources (%d):
%s

RESPONSE FORMAT (JSON ONLY):
{
  "action": "START_TASK | MODIFY_PARAM | ASK_ARTIFACT | ANSWER_GENERAL | SWITCH_TASK | CLARIFY",
  "task": "QUIZ | FLASHCARDS | MINDMAP | SUMMARY | STUDY_PLAN | COURSE_SEARCH | NONE",
  "retrieval_policy": "STRICT_SELECTED | PREFER_SELECTED | AUTO_EXPAND | GLOBAL",
  "retrieval_mode": "CHUNK_SEARCH | FULL_DOCUMENT | MULTI_DOC_SYNTHESIS | NONE",
  "param_updates": {
    "topic": string | null,
    "difficulty": string | null,
    "count": number | null
  },
  "thought": "brief reasoning"
}
"""

async def route_message(
    message: str,
    history: List[Dict[str, str]],
    session_state: SessionTaskState,
    selected_sources: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Route a message using the Control Planner architecture.
    SECURITY FIX - Phase 2/3: Enhanced with schema validation and ambiguity detection.
    """
    
    # SECURITY FIX - Phase 3: Check for ambiguous references first
    if _detect_ambiguous_references(message, session_state):
        logger.info(f"🤔 Forcing clarification for ambiguous message: {message[:50]}...")
        return {
            "action": "CLARIFY",
            "task": "NONE",
            "retrieval_policy": "GLOBAL",
            "retrieval_mode": "NONE",
            "param_updates": {},
            "thought": "Message contains ambiguous references that need clarification."
        }
    
    active_task = session_state.active_task or "none"
    current_topic = session_state.task_params.get("topic") or "none"
    
    sources_text = ""
    if selected_sources:
        for s in selected_sources:
            sources_text += f"- {s.get('name')} ({s.get('type')})\n"
    else:
        sources_text = "None"

    # SECURITY FIX - Phase 2: Sanitize history to prevent prompt injection
    safe_history = []
    for h in history[-5:]:
        content = h['content'][:300]
        # Remove potential prompt injection attempts
        content = re.sub(r'(ignore|forget|disregard)\s+(previous|all|above|system|instructions)', 
                        '[filtered]', content, flags=re.IGNORECASE)
        safe_history.append(f"{h['role']}: {content}")
    
    history_text = "\n".join(safe_history)
    
    prompt = ROUTER_PROMPT % (active_task, current_topic, len(selected_sources or []), sources_text)
    
    try:
        response = await chat_completion(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"History:\n{history_text}\n\nUser Message: {message}\n\nRespond only in JSON format."}
            ],
            temperature=0,
            max_tokens=300,
            stream=False,
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content.strip()
        
        # SECURITY FIX: Safe markdown stripping
        if raw_content.startswith("```"):
            raw_content = re.sub(r"```(?:json)?", "", raw_content).strip("`").strip()
            
        # SECURITY FIX: Strict schema validation instead of raw JSON parsing
        try:
            router_response = RouterResponse.parse_raw(raw_content)
            logger.info(f"✅ Router output validated successfully")
            return router_response.dict()
        except ValidationError as e:
            logger.warning(f"🚨 Router output failed validation: {e}")
            # Safe fallback - force clarification
            return {
                "action": "CLARIFY",
                "task": "NONE",
                "retrieval_policy": "GLOBAL", 
                "retrieval_mode": "NONE",
                "param_updates": {},
                "thought": "I need clarification to help you better."
            }
        
    except Exception as e:
        logger.error(f"⚠️ Planner error: {e}")
        return {
            "action": "ANSWER_GENERAL",
            "task": "NONE",
            "retrieval_policy": "GLOBAL",
            "retrieval_mode": "NONE", 
            "param_updates": {},
            "thought": "Planner failed, falling back to general chat."
        }

def _detect_mutation(message: str, state: SessionTaskState) -> Optional[Dict[str, Any]]:
    """Simple heuristic for common follow-up mutations."""
    if not state.active_task:
        return None
        
    msg = message.lower()
    updates = {}
    
    # Check for number changes
    num_match = re.search(r'\b(\d+)\b', msg)
    if num_match:
        val = int(num_match.group(1))
        if state.active_task == "quiz" and ("question" in msg or "make it" in msg):
            updates["num_questions"] = val
        elif state.active_task == "flashcards" and ("card" in msg or "make it" in msg):
            updates["num_cards"] = val
            
    # Check for difficulty
    if "easy" in msg or "beginner" in msg:
        updates["difficulty"] = "easy"
    elif "hard" in msg or "advanced" in msg:
        updates["difficulty"] = "hard"
    elif "medium" in msg:
        updates["difficulty"] = "medium"
        
    return updates if updates else None
