# tools/run_tool.py
import importlib

def run_tool(name: str, args: dict = None):
    # attempts to find tool in tools registry dynamically
    try:
        from tools import get_tools
        for t in get_tools():
            if t.get("name") == name:
                func = t.get("func")
                if callable(func):
                    if args and isinstance(args, dict):
                        return func(**args)
                    else:
                        return func(args)
        return {"error": "tool not found"}
    except Exception as e:
        return {"error": str(e)}

TOOL = {"name": "run_tool", "func": run_tool, "description": "Run another tool by name (internal helper)."}
