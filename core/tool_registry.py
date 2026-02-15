from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from tools import ALL_TOOLS

@dataclass
class ToolMetadata:
    name: str
    type: str  # "chat", "artifact", "search"
    output_format: str  # "markdown", "json_quiz", "json_flashcards", "json_mindmap", "raw"
    description: str

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Any] = {t.name: t for t in ALL_TOOLS}
        self._metadata: Dict[str, ToolMetadata] = self._initialize_metadata()

    def _initialize_metadata(self) -> Dict[str, ToolMetadata]:
        """Define metadata for all current tools."""
        return {
            "search_web": ToolMetadata(
                name="search_web",
                type="search",
                output_format="markdown",
                description="Global internet search for learning resources"
            ),
            "search_sources": ToolMetadata(
                name="search_sources",
                type="search",
                output_format="markdown",
                description="Search internal study materials"
            ),
            "generate_quiz": ToolMetadata(
                name="generate_quiz",
                type="artifact",
                output_format="json_quiz",
                description="Create practice quizzes"
            ),
            "generate_flashcards": ToolMetadata(
                name="generate_flashcards",
                type="artifact",
                output_format="json_flashcards",
                description="Create study flashcards"
            ),
            "generate_mindmap": ToolMetadata(
                name="generate_mindmap",
                type="artifact",
                output_format="json_mindmap",
                description="Create visual concept maps"
            ),
            "summarize_notes": ToolMetadata(
                name="summarize_notes",
                type="chat",
                output_format="markdown",
                description="Condense materials into summaries"
            ),
            "generate_report": ToolMetadata(
                name="generate_report",
                type="artifact",
                output_format="markdown",
                description="Generate comprehensive study reports"
            ),
            "generate_infographic": ToolMetadata(
                name="generate_infographic",
                type="artifact",
                output_format="raw", # JSON for infographics
                description="Create visual summaries"
            ),
            "generate_study_plan": ToolMetadata(
                name="generate_study_plan",
                type="chat",
                output_format="markdown",
                description="Create learning schedules"
            )
        }

    def get_tool(self, name: str) -> Optional[Any]:
        return self._tools.get(name)

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        return self._metadata.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

# Singleton instance
registry = ToolRegistry()
