"""
Training Dataset Validator for Collabry Fine-Tuning

Validates JSONL training data for:
- Format correctness
- Tool name validation
- Argument schema validation
- Message structure verification
- Balance across tool types
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter

# Valid tool names
VALID_TOOLS = {
    "search_sources",
    "generate_quiz",
    "generate_flashcards",
    "generate_mindmap",
    "generate_study_plan",
    "summarize_notes"
}

# Required parameters for each tool
TOOL_PARAMS = {
    "search_sources": {"query", "user_id"},
    "generate_quiz": {"notebook_id", "user_id", "num_questions", "topic"},
    "generate_flashcards": {"notebook_id", "user_id", "topic"},
    "generate_mindmap": {"topic", "user_id"},
    "generate_study_plan": {"subject", "topics", "user_id", "duration_days"},
    "summarize_notes": {"notebook_id", "user_id", "topic"}
}


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_messages_format(messages: List[Dict]) -> None:
    """Validate message format structure."""
    if not messages:
        raise ValidationError("Empty messages array")
    
    # First message should be system
    if messages[0]["role"] != "system":
        raise ValidationError("First message must be system role")
    
    # Check for valid roles
    valid_roles = {"system", "user", "assistant", "tool"}
    for i, msg in enumerate(messages):
        if "role" not in msg:
            raise ValidationError(f"Message {i} missing 'role' field")
        
        if msg["role"] not in valid_roles:
            raise ValidationError(f"Invalid role '{msg['role']}' in message {i}")
        
        # Tool messages need tool_call_id
        if msg["role"] == "tool" and "tool_call_id" not in msg:
            raise ValidationError(f"Tool message {i} missing 'tool_call_id'")


def validate_tool_calls(messages: List[Dict]) -> List[str]:
    """Validate tool calls in messages and return list of tools used."""
    tools_used = []
    
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tool_call in msg["tool_calls"]:
                # Check structure
                if "id" not in tool_call:
                    raise ValidationError(f"Tool call in message {i} missing 'id'")
                
                if "function" not in tool_call:
                    raise ValidationError(f"Tool call in message {i} missing 'function'")
                
                func = tool_call["function"]
                
                # Validate tool name
                tool_name = func.get("name")
                if not tool_name:
                    raise ValidationError(f"Tool call in message {i} missing function name")
                
                if tool_name not in VALID_TOOLS:
                    raise ValidationError(
                        f"Invalid tool name '{tool_name}' in message {i}. "
                        f"Valid tools: {VALID_TOOLS}"
                    )
                
                tools_used.append(tool_name)
                
                # Validate arguments
                if "arguments" not in func:
                    raise ValidationError(f"Tool call {tool_name} in message {i} missing arguments")
                
                try:
                    args = json.loads(func["arguments"])
                except json.JSONDecodeError:
                    raise ValidationError(f"Tool call {tool_name} in message {i} has invalid JSON arguments")
                
                # Check required parameters
                required_params = TOOL_PARAMS.get(tool_name, set())
                provided_params = set(args.keys())
                missing_params = required_params - provided_params
                
                if missing_params:
                    raise ValidationError(
                        f"Tool call {tool_name} in message {i} missing required parameters: {missing_params}"
                    )
    
    return tools_used


def validate_conversation_flow(messages: List[Dict]) -> None:
    """Validate logical conversation flow."""
    roles = [msg["role"] for msg in messages]
    
    # After assistant tool_call, expect tool message
    for i in range(len(messages) - 1):
        if messages[i]["role"] == "assistant" and messages[i].get("tool_calls"):
            if messages[i + 1]["role"] != "tool":
                raise ValidationError(
                    f"Expected tool message after assistant tool call at position {i}, "
                    f"got {messages[i + 1]['role']}"
                )


def validate_example(example: Dict, line_num: int) -> List[str]:
    """
    Validate a single training example.
    
    Returns:
        List of tool names used in this example
    """
    try:
        # Check top-level structure
        if "messages" not in example:
            raise ValidationError("Missing 'messages' field")
        
        messages = example["messages"]
        
        # Validate message format
        validate_messages_format(messages)
        
        # Validate tool calls
        tools_used = validate_tool_calls(messages)
        
        # Validate conversation flow
        validate_conversation_flow(messages)
        
        return tools_used
    
    except ValidationError as e:
        raise ValidationError(f"Line {line_num}: {str(e)}")


def validate_dataset(file_path: str) -> Dict[str, any]:
    """
    Validate entire dataset file.
    
    Returns:
        Dictionary with validation statistics
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"🔍 Validating {path.name}...")
    print("=" * 60)
    
    examples_count = 0
    errors = []
    tool_usage = Counter()
    no_tool_examples = 0
    multi_tool_examples = 0
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                example = json.loads(line)
                tools_used = validate_example(example, line_num)
                
                examples_count += 1
                
                if not tools_used:
                    no_tool_examples += 1
                else:
                    tool_usage.update(tools_used)
                    if len(tools_used) > 1:
                        multi_tool_examples += 1
            
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")
            except ValidationError as e:
                errors.append(str(e))
    
    print(f"\n📊 Validation Results:")
    print(f"  Total examples: {examples_count}")
    print(f"  No-tool examples: {no_tool_examples}")
    print(f"  Multi-tool examples: {multi_tool_examples}")
    print(f"  Errors found: {len(errors)}")
    
    if tool_usage:
        print(f"\n🔧 Tool Usage Distribution:")
        for tool, count in tool_usage.most_common():
            percentage = (count / sum(tool_usage.values())) * 100
            print(f"  {tool:25} {count:4} ({percentage:5.1f}%)")
    
    if errors:
        print(f"\n❌ Errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  • {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return {"valid": False, "error_count": len(errors)}
    else:
        print(f"\n✅ Dataset is valid!")
        
        # Check balance
        if tool_usage:
            max_count = max(tool_usage.values())
            min_count = min(tool_usage.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 3:
                print(f"\n⚠️  Warning: Tool usage is imbalanced (ratio: {imbalance_ratio:.1f}:1)")
                print(f"   Consider adding more examples for underrepresented tools.")
            else:
                print(f"\n✓ Tool distribution is reasonably balanced (ratio: {imbalance_ratio:.1f}:1)")
        
        return {
            "valid": True,
            "total_examples": examples_count,
            "no_tool_examples": no_tool_examples,
            "multi_tool_examples": multi_tool_examples,
            "tool_usage": dict(tool_usage)
        }


def main():
    parser = argparse.ArgumentParser(description="Validate Collabry training dataset")
    parser.add_argument("file", help="Path to JSONL training file")
    parser.add_argument("--strict", action="store_true", help="Enable strict validation")
    
    args = parser.parse_args()
    
    try:
        result = validate_dataset(args.file)
        
        if result["valid"]:
            print("\n" + "=" * 60)
            print("🎉 Validation passed! Dataset is ready for fine-tuning.")
            exit(0)
        else:
            print("\n" + "=" * 60)
            print("❌ Validation failed. Please fix the errors above.")
            exit(1)
    
    except Exception as e:
        print(f"\n❌ Validation error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
