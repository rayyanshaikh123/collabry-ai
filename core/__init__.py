"""Core package for COLLABRY.

Note: avoid importing heavy modules (like `agent`) at package import time so
helper scripts that only need `MemoryManager` can run even if `agent.py` has
syntax errors during iterative development.
"""

from .memory import MemoryManager

__all__ = ["MemoryManager"]
