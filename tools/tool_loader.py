import importlib
import pkgutil
import logging
from pathlib import Path

logger = logging.getLogger("tools")


def load_tools():
    """Auto-discover tools inside the tools/ folder."""
    tools = {}

    # Ensure we iterate the tools package directory even when __path__ isn't
    # available in some import contexts.
    package_dir = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        name = module_info.name

        # don’t load internal modules
        if name.startswith("_") or name == "tool_loader":
            continue

        try:
            module = importlib.import_module(f"tools.{name}")
            # Support legacy dict-based tool definition
            if hasattr(module, "TOOL"):
                tool = module.TOOL
                tools[tool["name"]] = tool
                logger.info(f"Loaded tool: {tool['name']}")
            # Support LangChain @tool decorated callables (have .name attr)
            for attr in dir(module):
                obj = getattr(module, attr)
                if callable(obj) and hasattr(obj, "name") and not obj.__name__.startswith("_"):
                    tname = getattr(obj, "name")
                    if tname not in tools:  # don't overwrite dict variant
                        tools[tname] = {"name": tname, "func": obj, "description": (obj.__doc__ or "").strip()}
                        logger.info(f"Loaded tool (decorated): {tname}")
        except Exception as e:
            logger.error(f"Failed to load tool '{name}': {e}")
    return tools
