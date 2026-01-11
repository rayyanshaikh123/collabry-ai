# tools/tool_manager.py
from typing import Dict, Any, List
import importlib
import pkgutil
import logging

logger = logging.getLogger(__name__)

def list_tools() -> Dict[str, Any]:
    # import tools package and read get_tools (runtime)
    try:
        import tools
        return {"tools": [t["name"] for t in tools.get_tools()]}
    except Exception as e:
        return {"error": str(e)}

def run_tool_by_name(name: str, args: dict):
    try:
        import tools
        for t in tools.get_tools():
            if t["name"] == name:
                func = t["func"]
                if isinstance(args, dict):
                    return func(**args)
                return func(args)
        return {"error": "tool not found"}
    except Exception as e:
        return {"error": str(e)}

TOOL = {"name": "tool_manager", "func": list_tools, "description": "List available tools or run a named tool via run_tool_by_name (use run_tool helper)."}
