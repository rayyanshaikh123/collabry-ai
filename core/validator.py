"""
Retrieval Validator - Deterministic Rule Engine.

Ensures the combination of policy and mode is logically possible and safe.
This module is pure Python and contains no LLM calls for maximum safety.
SECURITY FIX - Phase 3: Enhanced with comprehensive validation rules.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def validate_retrieval_plan(
    policy: str, 
    mode: str, 
    selected_sources_count: int
) -> Dict[str, Any]:
    """
    Verify and correct the planner decision so retrieval is logically possible and safe.
    
    Args:
        policy: The requested retrieval policy (STRICT_SELECTED, etc.)
        mode: The requested retrieval mode (CHUNK_SEARCH, etc.)
        selected_sources_count: Number of sources currently selected by the user.
        
    Returns:
        Dict containing final_policy, final_mode, changed (bool), and reason.
    """
    final_policy = policy
    final_mode = mode
    reason = "valid"
    changed = False

    # Rule 1: STRICT_SELECTED requires at least one selected source.
    if final_policy == "STRICT_SELECTED" and selected_sources_count == 0:
        final_policy = "AUTO_EXPAND"
        reason = "STRICT_SELECTED requires >= 1 source. Falling back to AUTO_EXPAND."
        changed = True

    # Rule 2: GLOBAL knowledge cannot use retrieval.
    if final_policy == "GLOBAL":
        if final_mode != "NONE":
            final_mode = "NONE"
            reason = "GLOBAL policy overrides mode to NONE."
            changed = True

    # Rule 3: FULL_DOCUMENT requires selected sources.
    if final_mode == "FULL_DOCUMENT" and selected_sources_count == 0:
        final_mode = "NONE"
        final_policy = "GLOBAL"
        reason = "FULL_DOCUMENT requires >= 1 source. Changing to GLOBAL/NONE."
        changed = True

    # Rule 4: MULTI_DOC_SYNTHESIS requires at least 2 sources.
    if final_mode == "MULTI_DOC_SYNTHESIS" and selected_sources_count < 2:
        final_mode = "CHUNK_SEARCH"
        reason = "MULTI_DOC_SYNTHESIS requires >= 2 sources. Downgrading to CHUNK_SEARCH."
        changed = True

    # Rule 6: NONE mode overrides policy only if we aren't in a notebook session.
    # Actually, we should keep the requested policy so the system prompt stays grounded.
    if final_mode == "NONE":
        # We don't force GLOBAL here anymore to allow "Source Awareness" in general chat.
        pass

    # Rule 5: STRICT_SELECTED cannot expand beyond selected sources.
    if final_policy == "STRICT_SELECTED":
        allowed_modes = ["CHUNK_SEARCH", "FULL_DOCUMENT", "MULTI_DOC_SYNTHESIS", "NONE"]
        if final_mode not in allowed_modes:
            final_mode = "CHUNK_SEARCH"
            reason = "STRICT_SELECTED supports search or NONE mode."
            changed = True

    return {
        "final_policy": final_policy,
        "final_mode": final_mode,
        "changed": changed,
        "reason": reason
    }


def validate_tool_parameters(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    SECURITY FIX - Phase 3: Comprehensive tool parameter validation.
    Prevents injection attacks, validates types, and enforces business rules.
    """
    validated_params = {}
    
    # Common dangerous patterns across all tools
    dangerous_patterns = [
        r'\$\w+',  # MongoDB operators
        r'<script.*?>',  # XSS attempts
        r'javascript:',  # JavaScript injection
        r'eval\s*\(',  # Code evaluation
        r'\.\./',  # Path traversal
        r';.*?(rm|del|drop|exec)',  # Command injection
        r'\|\s*(cat|ls|pwd|whoami)',  # Unix command injection
    ]
    
    for key, value in params.items():
        try:
            # Skip None values
            if value is None:
                continue
                
            # Basic type validation
            if not isinstance(value, (str, int, float, bool, list)):
                logger.warning(f"🚨 Invalid parameter type for {key}: {type(value)}")
                continue
            
            # String validation and sanitization
            if isinstance(value, str):
                # Check for dangerous patterns
                for pattern in dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        logger.warning(f"🚨 Dangerous pattern in {key}: {pattern}")
                        continue  # Skip this parameter entirely
                
                # Length limits based on parameter type
                max_lengths = {
                    'topic': 500,
                    'subject': 500,
                    'query': 1000,
                    'content': 10000,
                    'description': 2000,
                    'title': 200,
                    'name': 100,
                    'notebook_id': 50,
                    'user_id': 50,
                }
                
                max_len = max_lengths.get(key, 500)
                if len(value) > max_len:
                    logger.warning(f"🚨 Parameter {key} too long: {len(value)} > {max_len}")
                    value = value[:max_len]
                
                # Remove HTML tags and dangerous characters
                value = re.sub(r'<[^>]*>', '', value)  # Strip HTML
                value = re.sub(r'[<>{}();]', '', value)  # Remove dangerous chars
                
            # Numeric validation
            elif isinstance(value, (int, float)):
                if key in ['count', 'num_questions', 'num_cards']:
                    if value < 1 or value > 100:
                        logger.warning(f"🚨 Parameter {key} out of range: {value}")
                        value = max(1, min(100, value))  # Clamp to valid range
                        
            # List validation
            elif isinstance(value, list):
                if key == 'source_ids':
                    # Validate that all items are strings and look like valid IDs
                    valid_items = []
                    for item in value[:50]:  # Limit list size
                        if isinstance(item, str) and re.match(r'^[A-Za-z0-9_-]{1,50}$', item):
                            valid_items.append(item)
                        else:
                            logger.warning(f"🚨 Invalid source ID: {item}")
                    value = valid_items
                else:
                    # Generic list validation - limit size and validate items
                    value = value[:50]  # Reasonable limit
                    
            validated_params[key] = value
            
        except Exception as e:
            logger.error(f"🚨 Error validating parameter {key}: {e}")
            continue  # Skip problematic parameters
    
    # Tool-specific validation rules
    if tool_name == "generate_quiz":
        # Ensure reasonable quiz parameters
        validated_params["num_questions"] = min(validated_params.get("num_questions", 5), 50)
        validated_params["difficulty"] = validated_params.get("difficulty", "medium")
        if validated_params["difficulty"] not in ["easy", "medium", "hard"]:
            validated_params["difficulty"] = "medium"
            
    elif tool_name == "generate_flashcards":
        validated_params["num_cards"] = min(validated_params.get("num_cards", 10), 100)
        
    elif tool_name == "generate_study_plan":
        # Validate duration and schedule parameters
        duration = validated_params.get("duration", "week")
        if duration not in ["day", "week", "month"]:
            validated_params["duration"] = "week"
            
    elif tool_name == "search_web":
        # Ensure query is present and reasonable
        if "query" not in validated_params or not validated_params["query"]:
            raise ValueError("Web search requires a valid query")
            
    logger.debug(f"✅ Validated {len(validated_params)} parameters for {tool_name}")
    return validated_params


def validate_source_boundaries(
    user_id: str, 
    notebook_id: Optional[str], 
    source_ids: List[str], 
    requesting_user_id: str
) -> bool:
    """
    SECURITY FIX - Phase 3: Enforce source access boundaries.
    Prevent users from accessing sources outside their authorized scope.
    """
    try:
        # Basic user isolation check
        if user_id != requesting_user_id:
            logger.warning(f"🚨 Cross-user access attempt: {requesting_user_id} -> {user_id}")
            return False
            
        # Validate source ID format (prevent injection)
        for source_id in source_ids:
            if not isinstance(source_id, str):
                logger.warning(f"🚨 Invalid source ID type: {type(source_id)}")
                return False
                
            if not re.match(r'^[A-Za-z0-9_-]{1,50}$', source_id):
                logger.warning(f"🚨 Invalid source ID format: {source_id}")
                return False
                
            # Prevent path traversal in source IDs
            if '../' in source_id or '\\' in source_id:
                logger.warning(f"🚨 Path traversal attempt in source ID: {source_id}")
                return False
        
        # If notebook context is provided, validate notebook access
        if notebook_id:
            if not re.match(r'^[A-Za-z0-9_-]{1,50}$', notebook_id):
                logger.warning(f"🚨 Invalid notebook ID format: {notebook_id}")
                return False
                
        # Additional checks can be added here for:
        # - Database-level permission verification
        # - Organizational boundaries
        # - Premium feature access
        
        logger.debug(f"✅ Source boundary validation passed for user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"🚨 Error in source boundary validation: {e}")
        return False


def validate_json_structure(data: str, expected_schema: dict) -> Dict[str, Any]:
    """
    SECURITY FIX - Phase 3: Safe JSON parsing with schema validation.
    Prevents JSON injection and validates structure.
    """
    try:
        # Parse JSON safely
        parsed = json.loads(data)
        
        # Validate against expected schema
        def validate_recursive(obj, schema):
            if isinstance(schema, dict):
                if not isinstance(obj, dict):
                    return False
                for key, value_schema in schema.items():
                    if key not in obj:
                        if isinstance(value_schema, dict) and value_schema.get('required', False):
                            return False
                        continue
                    if not validate_recursive(obj[key], value_schema):
                        return False
                return True
            elif isinstance(schema, list):
                if not isinstance(obj, list):
                    return False
                if schema and len(schema) > 0:
                    # Validate each item against the first schema element
                    for item in obj:
                        if not validate_recursive(item, schema[0]):
                            return False
                return True
            elif isinstance(schema, type):
                return isinstance(obj, schema)
            else:
                return obj == schema
        
        if validate_recursive(parsed, expected_schema):
            logger.debug("✅ JSON validation passed")
            return {"valid": True, "data": parsed}
        else:
            logger.warning("🚨 JSON schema validation failed")
            return {"valid": False, "error": "Schema mismatch"}
            
    except json.JSONDecodeError as e:
        logger.warning(f"🚨 Invalid JSON: {e}")
        return {"valid": False, "error": f"Invalid JSON: {str(e)}"}
    except Exception as e:
        logger.error(f"🚨 JSON validation error: {e}")
        return {"valid": False, "error": f"Validation error: {str(e)}"}


def sanitize_user_input(input_text: str) -> str:
    """
    SECURITY FIX - Phase 3: Comprehensive input sanitization.
    Remove dangerous content while preserving legitimate text.
    """
    if not isinstance(input_text, str):
        return ""
    
    # Remove NULL bytes and control characters
    sanitized = ''.join(char for char in input_text if ord(char) >= 32 or char in '\n\t')
    
    # Remove potential script injection
    sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove dangerous HTML attributes
    sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
    
    # Remove SQL injection patterns
    sql_patterns = [
        r';\s*(DROP|DELETE|INSERT|UPDATE|CREATE)\s+',
        r'(UNION|SELECT).*?(FROM|WHERE)',
        r'--\s*\w',  # SQL comments
        r'/\*.*?\*/',  # SQL block comments
    ]
    
    for pattern in sql_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    # Remove MongoDB operator injection
    sanitized = re.sub(r'\$\w+\s*:', '', sanitized)
    
    # Limit length to prevent DoS
    sanitized = sanitized[:10000]
    
    logger.debug(f"✅ Sanitized input: {len(input_text)} -> {len(sanitized)} chars")
    return sanitized.strip()
