# tools/__init__.py

"""
Dynamic Tool Loader for Collabry AI Engine
Automatically discovers all study-platform-relevant tools with a TOOL={} export.

Legacy tools (browser_control, system_automation, task_scheduler) are excluded.
Active tools: web_search, web_scraper, read_file, write_file, doc_generator, 
              ppt_generator
"""

import pkgutil
import importlib
import inspect
import logging

logger = logging.getLogger(__name__)

# Tools excluded from loading (moved to legacy_tools/)
EXCLUDED_TOOLS = {
    "browser_control",
    "system_automation", 
    "task_scheduler",
    "tool_loader",  # Internal module, not a tool
    "tool_manager",  # Internal module
}


def load_tools():
    """
    Returns dict[str, dict] where each dict has {name, func, description}.
    Used by agent.py's create_agent().
    """
    tools = {}

    # Scan this package (tools/)
    for module_info in pkgutil.iter_modules(__path__):
        module_name = module_info.name
        
        # Skip excluded modules
        if module_name in EXCLUDED_TOOLS or module_name.startswith("_"):
            continue

        try:
            module = importlib.import_module(f"tools.{module_name}")

            # Only register modules that expose a TOOL dict
            if hasattr(module, "TOOL"):
                tool_def = module.TOOL
                name = tool_def.get("name")
                func = tool_def.get("func")

                if name and callable(func):
                    tools[name] = tool_def
                    logger.info(f"Loaded tool: {name}")
        except Exception as e:
            logger.error(f"Failed to load tool '{module_name}': {e}")

    return tools


def get_tools():
    """
    Returns list[dict] for backward compatibility with tool_manager.
    Each dict has {name, func, description}.
    """
    tools_dict = load_tools()
    return list(tools_dict.values())
