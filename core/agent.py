"""
Study Assistant Agent - Simplified Linear Architecture.

This version replaces the ReAct agent with a direct flow:
User -> Router LLM -> Tool Execution -> Formatter LLM.
SessionTaskState is used to manage conversational context and mutations.
"""

from typing import AsyncGenerator, Optional, Dict, Any, List, Tuple
from core.llm import get_langchain_llm, get_llm_config, chat_completion
from core.conversation import get_conversation_manager
from core.session_state import (
    get_session_state,
    SessionTaskState,
    save_session_state,
    load_session_state,
)
from core.router import route_message
from core.validator import validate_retrieval_plan
from core.retrieval_service import get_hybrid_context, fetch_source_metadata
from core.tool_registry import registry
from core.response_manager import ResponseManager
from core.language import detect_session_language, build_language_instructions, normalize_query_with_cache
from tools import ALL_TOOLS
import json
import re
import logging

logger = logging.getLogger(__name__)

def _clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    SECURITY FIX - Phase 2: Enhanced parameter sanitization.
    Remove placeholder strings, validate types, and prevent injection attacks.
    """
    cleaned = {}
    for k, v in params.items():
        # Reject None values
        if v is None:
            continue
            
        # SECURITY FIX: Strict type validation - only allow safe types
        if not isinstance(v, (str, int, float, bool)):
            logger.warning(f"🚨 Rejected parameter {k} with unsafe type {type(v)}: {v}")
            continue
            
        if isinstance(v, str):
            # Remove placeholder strings
            if v.lower() in ("none", "null", "n/a", "undefined", ""):
                continue
                
            # SECURITY FIX: Reject MongoDB operators and injection patterns
            dangerous_patterns = [
                '$ne', '$gt', '$lt', '$in', '$or', '$and',  # MongoDB operators
                'drop table', 'delete from', 'insert into', 'update set',  # SQL injection
                '<script>', 'javascript:', 'eval(', 'function(',  # XSS attempts
                '../', '.\\',  # Path traversal
                '; rm -rf', '; del ',  # Command injection
            ]
            
            if any(dangerous in str(v).lower() for dangerous in dangerous_patterns):
                logger.warning(f"🚨 Rejected parameter {k} with dangerous content: {v}")
                continue
                
            # Sanitize and limit string length  
            v = re.sub(r'[<>{}();\'"\\$]', '', v)  # Remove dangerous chars
            v = v[:200]  # Reasonable length limit
            
        # Validate numeric ranges
        elif isinstance(v, int):
            if k in ['count', 'num_questions', 'num_cards']:
                if v < 1 or v > 100:  # Reasonable bounds
                    logger.warning(f"🚨 Parameter {k} out of range: {v}")
                    continue
            
        cleaned[k] = v
    
    logger.info(f"✅ Cleaned parameters: {list(cleaned.keys())}")
    return cleaned

def _is_artifact_related_question(message: str, artifact_context: Dict[str, Any]) -> bool:
    """
    SECURITY FIX - Phase 2: Detect if user question refers to previously generated artifact.
    """
    if not artifact_context:
        return False
        
    artifact_type = artifact_context.get('type', '').lower()
    message_lower = message.lower()
    
    # Check for explicit artifact type references
    artifact_indicators = {
        'quiz': ['question', 'quiz', 'answer', 'correct', 'wrong'],
        'flashcards': ['card', 'flashcard', 'front', 'back'],
        'mindmap': ['mindmap', 'mind map', 'node', 'branch'], 
        'summary': ['summary', 'summarized', 'point'],
        'report': ['report', 'section'],
        'study_plan': ['plan', 'schedule', 'timeline']
    }
    
    if artifact_type in artifact_indicators:
        indicators = artifact_indicators[artifact_type]
        if any(indicator in message_lower for indicator in indicators):
            return True
    
    # Check for positional references (question 3, card 2, etc.)
    if re.search(r'\b(question|card|item|point|step)\s+\d+', message_lower):
        return True
        
    # Check for relative references that likely refer to recent content
    relative_refs = ['this', 'that', 'it', 'them', 'these', 'those']
    if any(ref in message_lower for ref in relative_refs):
        return True
        
    return False

async def _handle_artifact_question(
    message: str,
    artifact_context: Dict[str, Any],
    history: List,
    session_language: Optional[str] = None,
    is_mixed_language: bool = False,
) -> str:
    """
    SECURITY FIX - Phase 2: Handle questions about previously generated artifacts.
    """
    artifact_content = artifact_context.get('content', '')
    artifact_type = artifact_context.get('type', '')
    
    # Truncate artifact content for prompt (prevent context overflow)
    if len(str(artifact_content)) > 2000:
        content_preview = str(artifact_content)[:2000] + "...[truncated]"
    else:
        content_preview = str(artifact_content)
    
    system_prompt = f"""
    You are helping the user with their {artifact_type} that you previously generated.
    
    The {artifact_type} content is:
    {content_preview}
    
    Answer the user's question about this {artifact_type} specifically.
    Be precise and refer to the exact content when possible.
    If asked about a specific item (like "question 3"), identify it clearly.
    """

    # Multilingual behavior for artifact follow-up as well.
    if session_language:
        system_prompt += "\n\n" + build_language_instructions(session_language, is_mixed_language)
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add recent conversation context
    for h in history[-3:]:  # Limit context to prevent overflow
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})
    
    response = await chat_completion(messages, stream=False)
    return response.choices[0].message.content

def _sanitize_conversation_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    SECURITY FIX - Phase 3: Filter conversation history for prompt injection patterns.
    """
    sanitized_history = []
    
    for message in history:
        content = message.get("content", "")
        role = message.get("role", "user")
        
        # Known prompt injection patterns to filter
        injection_patterns = [
            # Direct instruction attempts
            r'(ignore|forget|disregard)\s+(previous|all|above|system|instructions)',
            r'you\s+are\s+now\s+(a|an)\s+',
            r'your\s+(new|actual|real)\s+(role|instructions|purpose)',
            
            # System prompt exposure attempts  
            r'(show|tell|reveal|display)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)',
            r'what\s+(are|were)\s+your\s+(original|initial)\s+instructions',
            r'(repeat|echo)\s+(back\s+)?your\s+instructions',
            
            # Role manipulation attempts
            r'act\s+as\s+(if\s+)?you\s+(are|were)',
            r'pretend\s+(you\s+are|to\s+be)',
            r'roleplay\s+as',
            
            # Context injection attempts
            r'\\n\\n(system|user|assistant):',
            r'\{"role":\s*"(system|assistant)"',
            
            # Jailbreak attempts
            r'(developer|admin|root)\s+mode',
            r'break\s+out\s+of\s+(character|role)',
            r'stop\s+(acting|being)',
        ]
        
        # Check for injection patterns
        has_injection = False
        for pattern in injection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning(f"🚨 Prompt injection detected in {role} message: {pattern}")
                has_injection = True
                break
        
        if has_injection:
            # Replace with sanitized message but preserve conversational flow
            sanitized_content = "[Content filtered for security]"
            if role == "user":
                sanitized_content = "User message contained problematic content."
            elif role == "assistant":
                sanitized_content = "I provided a response to the user."
        else:
            # Standard sanitization - remove potential injection markers
            sanitized_content = content
            # Remove obvious injection markers but preserve normal content
            sanitized_content = re.sub(r'\\n\\n(system|user|assistant):', ' [filtered] ', sanitized_content)
            sanitized_content = re.sub(r'\{"role":\s*"[^"]*"[^}]*\}', '[filtered]', sanitized_content)
            
        # Limit message length to prevent context overflow attacks
        sanitized_content = sanitized_content[:1000]
        
        sanitized_history.append({
            "role": role,
            "content": sanitized_content
        })
    
    logger.debug(f"✅ Sanitized {len(history)} history messages")
    return sanitized_history


async def run_agent(
    user_id: str,
    session_id: str,
    message: str,
    notebook_id: Optional[str] = None,
    use_rag: bool = False,
    source_ids: Optional[List[str]] = None,
    stream: bool = True,
    sender_name: Optional[str] = None,
    is_collaborative: bool = False,
    token: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Main entry point using the simplified linear flow.
    SECURITY FIX - Phase 2/3: Enhanced with artifact precedence and security fixes.
    """
    # 1. Initialize State with proper user isolation
    session_state = get_session_state(session_id, user_id)

    # PHASE 2 — SESSION STATE STORAGE (Redis-backed with in-memory fallback)
    # Use a composite identifier to preserve user isolation semantics.
    composite_id = f"{user_id}:{session_id}"
    try:
        cached_state = await load_session_state(composite_id)
        if cached_state:
            session_state.apply_dict(cached_state)
    except Exception as e:
        logger.warning(f"Failed to hydrate session state from Redis for {composite_id}: {e}")
    conv_manager = get_conversation_manager()
    history = conv_manager.get_history(user_id, session_id, limit=10)

    # Detect and store the session_language based on the latest user message.
    # This is done once per turn and reused across planner/tool/formatter steps.
    prev_lang = session_state.session_language or "en"
    session_lang, lang_conf, is_mixed_lang = detect_session_language(message, previous_language=prev_lang)
    session_state.session_language = session_lang
    logger.info(f"🌐 Session {session_id}: language={session_lang} (conf={lang_conf:.2f}, mixed={is_mixed_lang})")

    # PHASE 3 — QUERY NORMALIZATION CACHE
    # Use a cached, normalized form of the query for retrieval-only logic.
    # The planner and conversation continue to see the raw user message.
    normalized_query = await normalize_query_with_cache(message)
    
    # SECURITY FIX - Phase 2: Check for artifact context BEFORE retrieval planning
    artifact_context = session_state.get_artifact_context()
    
    if artifact_context and _is_artifact_related_question(message, artifact_context):
        yield {"type": "thinking", "content": f"💭 Referring to previous {artifact_context['type']}..."}
        
        # Handle artifact-specific questions without retrieval
        response = await _handle_artifact_question(
            message,
            artifact_context,
            history,
            session_language=session_lang,
            is_mixed_language=is_mixed_lang,
        )
        
        # Stream response tokens
        for token_chunk in re.split(r"(\s+)", response):
            if token_chunk.strip(): 
                yield {"type": "token", "content": token_chunk}
        
        yield {"type": "complete", "message": response}
        conv_manager.save_turn(user_id, session_id, message, response, notebook_id,
                               sender_id=user_id, sender_name=sender_name)

        # Refresh Redis TTL for this session state
        try:
            await save_session_state(composite_id, session_state.to_dict(), ttl_seconds=None)
        except Exception as e:
            logger.warning(f"Failed to persist session state to Redis for {composite_id}: {e}")
        return
    
    # Pre-fetch source metadata for the Planner
    selected_sources = []
    if source_ids:
        try:
            # SECURITY FIX: Use validated source IDs
            from core.retrieval_service import validate_source_ids
            validated_source_ids = validate_source_ids(source_ids)
            selected_sources = await fetch_source_metadata(notebook_id, validated_source_ids, token)
        except ValueError as e:
            yield {"type": "error", "message": f"Invalid source selection: {str(e)}"}
            return
    
    # 2. Control Planner - Decide Task & Initial Retrieval Strategy
    planner_output = await route_message(message, history, session_state, selected_sources)
    thought = planner_output.get("thought")
    action = planner_output.get("action")
    task = planner_output.get("task")
    
    # 3. Retrieval Validator - Determine Final Policy & Mode (Deterministic)
    validation = validate_retrieval_plan(
        policy=planner_output.get("retrieval_policy", "GLOBAL"),
        mode=planner_output.get("retrieval_mode", "NONE"),
        selected_sources_count=len(source_ids or [])
    )
    
    final_policy = validation["final_policy"]
    final_mode = validation["final_mode"]
    
    if validation["changed"]:
        logger.info(f"🛡️ Validator corrected plan: {validation['reason']}")

    # Emit thinking step for UI
    if thought:
        msg = f"💭 {thought}"
        if validation["changed"]:
            msg += f" (Note: {validation['reason']})"
        yield {"type": "thinking", "content": msg}

    # 4. Handle Task Actions
    if action == "START_TASK" or action == "MODIFY_PARAM":
        # Resolve tool name from task
        tool_name = f"generate_{task.lower()}" if task != "COURSE_SEARCH" else "search_web"
        if task == "SUMMARY": tool_name = "summarize_notes"
        
        params = planner_output.get("param_updates", {})
        params = _clean_params(params)
        
        # Map 'topic' to 'query' or 'subject' where needed
        if tool_name == "search_web" and "topic" in params:
            params["query"] = params.pop("topic")
        elif tool_name == "generate_study_plan" and "topic" in params:
            params["subject"] = params.get("topic")
            params["topics"] = params.pop("topic")
        
        yield {"type": "tool_start", "tool": tool_name, "inputs": params}
        
        # Track task in session state
        session_state.set_task(task, tool_name, params)
        
        # Locate tool object
        tool_obj = next((t for t in ALL_TOOLS if t.name == tool_name), None)
        if not tool_obj:
            yield {"type": "error", "message": f"Tool '{tool_name}' not found."}
            return
            
        # Context Injection
        final_params = params.copy()
        if "user_id" not in final_params: final_params["user_id"] = user_id
        if "notebook_id" not in final_params and notebook_id: final_params["notebook_id"] = notebook_id
        if "source_ids" not in final_params and source_ids: final_params["source_ids"] = source_ids
        
        # Inject Validated Policy & Mode into tool params
        final_params["retrieval_policy"] = final_policy
        final_params["retrieval_mode"] = final_mode
        final_params["token"] = token
        
        try:
            # Execute tool
            print(f"🛠️ Executing {tool_name} with {final_params}")
            if hasattr(tool_obj, "ainvoke"):
                output = await tool_obj.ainvoke(final_params)
            else:
                output = tool_obj.invoke(final_params)
                
            yield {"type": "tool_end", "tool": tool_name, "output": output}
            
            # Dynamic Formatting
            metadata = registry.get_metadata(tool_name)
            if metadata:
                content = await ResponseManager.format_response(tool_name, output, metadata, final_params)
            else:
                content = str(output)
            
            # Store generated artifact for follow-up questions
            session_state.store_artifact(tool_name.replace('generate_', ''), content, {"tool": tool_name, "params": final_params})
            
            # Stream tokens
            if isinstance(content, str):
                for token_chunk in re.split(r"(\s+)", content):
                    if token_chunk: yield {"type": "token", "content": token_chunk}
                yield {"type": "complete", "message": content}
                conv_manager.save_turn(user_id, session_id, message, content, notebook_id,
                                       sender_id=user_id, sender_name=sender_name)
            else:
                msg = f"I've successfully performed '{tool_name}' for you."
                yield {"type": "token", "content": msg}
                yield {"type": "complete", "message": msg}
                conv_manager.save_turn(user_id, session_id, message, msg, notebook_id,
                                       sender_id=user_id, sender_name=sender_name)
                
        except Exception as e:
            print(f"❌ Execution error: {e}")
            yield {"type": "error", "message": f"I hit a snag running the {tool_name} tool."}
            
        return

    # 5. Hybrid Retrieval Stage (For general chat grounded in sources)
    context = await get_hybrid_context(
        user_id=user_id,
        notebook_id=notebook_id,
        policy=final_policy,
        mode=final_mode,
        source_ids=source_ids or [],
        query=normalized_query,
        token=token
    )

    # 6. Standard Conversation with Grounded Context
    system_prompt = "You are a helpful study assistant. Be supportive and keep answers clear."
    
    # Source Awareness in System Prompt
    if selected_sources:
        source_names = ", ".join([s['name'] for s in selected_sources])
        system_prompt += f"\n\nYou are currently grounded in the following sources: {source_names}."

    if context:
        system_prompt += f"\n\nUse the following context to help answer the user. Be concise and prioritize information from the context if it matches the query.\n\nCONTEXT:\n{context}"
        if final_policy == "STRICT_SELECTED":
            system_prompt += "\n\nCRITICAL: Use ONLY the provided context. If the answer is not there, say you don't know based on the selected sources."
        else:
            system_prompt += "\n\nIf the provided context doesn't contain the answer but is related, say so. If completely unrelated, you may use your general knowledge but mention it's not in the sources."
    elif final_mode == "NONE" and notebook_id:
        system_prompt += "\n\nYou have access to the notebook but no specific context was retrieved for this turn. If the user is asking about the sources, suggest that they ask for a 'summary' or 'more detail'."

    # Append language-specific instructions so the formatter LLM stays in the user's language.
    if session_lang:
        system_prompt += "\n\n" + build_language_instructions(session_lang, is_mixed_lang)

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    # For collaborative sessions, include sender context
    if is_collaborative and sender_name:
        messages[0]["content"] += f"\n\nNote: This is a collaborative study session. The current speaker is '{sender_name}'."
    
    # SECURITY FIX - Phase 3: Sanitize conversation history before including in prompt
    for h in _sanitize_conversation_history(history):
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})
    
    try:
        resp_stream = await chat_completion(messages, stream=True)
        full_text = ""
        async for chunk in resp_stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                yield {"type": "token", "content": delta}
        
        yield {"type": "complete", "message": full_text}
        conv_manager.save_turn(user_id, session_id, message, full_text, notebook_id,
                               sender_id=user_id, sender_name=sender_name)

        # Refresh Redis TTL for this session state at the end of the turn
        try:
            await save_session_state(composite_id, session_state.to_dict(), ttl_seconds=None)
        except Exception as e:
            logger.warning(f"Failed to persist session state to Redis for {composite_id}: {e}")
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        yield {"type": "error", "message": "I'm having trouble connecting right now."}


def _coerce_quiz_to_frontend_schema(tool_output: str) -> str:
    """Convert tool JSON into frontend quiz array schema."""
    raw = (tool_output or "").strip()
    try:
        # If it's already a list, it might be the frontend schema already
        data = json.loads(raw)
        if isinstance(data, list): return raw
    except Exception:
        return raw

    # Tool returns { quiz_title, difficulty, questions: [{options:{A..D}, correct_answer:"A"}] }
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        out = []
        for q in data.get("questions", []):
            if not isinstance(q, dict): continue
            options_dict = q.get("options") if isinstance(q.get("options"), dict) else {}
            options = [
                options_dict.get("A", ""),
                options_dict.get("B", ""),
                options_dict.get("C", ""),
                options_dict.get("D", ""),
            ]
            letter = str(q.get("correct_answer") or "A").strip().upper()[:1]
            correct_idx = ord(letter) - 65 if 'A' <= letter <= 'D' else 0
            out.append({
                "question": str(q.get("question", "")),
                "options": options,
                "correctAnswer": correct_idx,
                "explanation": str(q.get("explanation", "")),
            })
        if out: return json.dumps(out, ensure_ascii=False)
    return raw


def _coerce_flashcards_to_frontend_schema(tool_output: str, fallback_title: str) -> str:
    """Convert tool JSON into frontend flashcards schema."""
    raw = (tool_output or "").strip()
    try:
        data = json.loads(raw)
    except Exception:
        return raw

    if isinstance(data, dict) and isinstance(data.get("cards"), list):
        title = data.get("title") if isinstance(data.get("title"), str) else fallback_title
        cards_out = []
        for i, card in enumerate(data.get("cards", [])):
            if not isinstance(card, dict): continue
            front, back = card.get("front"), card.get("back")
            if not front or not back: continue
            
            category = None
            tags = card.get("tags")
            if isinstance(tags, list) and tags:
                category = str(tags[0]).strip()[:40]

            out_card = {"id": f"card-{i+1}", "front": str(front).strip(), "back": str(back).strip()}
            if category: out_card["category"] = category
            cards_out.append(out_card)
            
        if cards_out: return json.dumps({"title": title, "cards": cards_out}, ensure_ascii=False)
    return raw


# Non-streaming wrapper for sync-like interactions


async def chat(
    user_id: str,
    session_id: str,
    message: str,
    notebook_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
) -> str:
    """Non-streaming wrapper."""
    response = ""
    async for event in run_agent(user_id, session_id, message, notebook_id=notebook_id, source_ids=source_ids):
        if event["type"] == "token":
            response += event["content"]
        elif event["type"] == "complete":
            response = event["message"]
    return response
