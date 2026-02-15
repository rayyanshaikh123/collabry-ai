"""
Tool Registry - LangChain tools for the study assistant.

All tools follow the LangChain @tool decorator pattern.
Tools are automatically discovered and registered with the agent.
"""

from tools.search_sources import search_sources
from tools.search_web import search_web
from tools.summarize import summarize_notes
from tools.generate_quiz import generate_quiz
from tools.generate_flashcards import generate_flashcards
from tools.generate_mindmap import generate_mindmap
from tools.generate_study_plan import generate_study_plan
from tools.generate_report import generate_report
from tools.generate_infographic import generate_infographic


# All tools available to the agent
ALL_TOOLS = [
    search_web,           # Web search for courses, tutorials, current info
    search_sources,       # Search user's uploaded documents
    summarize_notes,
    generate_quiz,
    generate_flashcards,
    generate_mindmap,
    generate_study_plan,
    generate_report,
    generate_infographic,
]


def get_tools(user_id: str, notebook_id: str = None):
    """
    Get tools with context injection.
    
    Args:
        user_id: User identifier to inject into tool context
        notebook_id: Optional notebook identifier
    
    Returns:
        List of tools with context bound
    """
    # For now, return tools as-is
    # Context injection will be handled in agent.py
    return ALL_TOOLS


__all__ = [
    'ALL_TOOLS',
    'get_tools',
    'search_web',
    'search_sources',
    'summarize_notes',
    'generate_quiz',
    'generate_flashcards',
    'generate_mindmap',
    'generate_study_plan',
    'generate_report',
    'generate_infographic',
]
