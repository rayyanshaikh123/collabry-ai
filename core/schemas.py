"""Pydantic schemas for tool output validation."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional

class QuizQuestion(BaseModel):
    """Single quiz question with validation."""
    question: str = Field(..., min_length=10, description="The quiz question text")
    options: Dict[str, str] = Field(..., description="Answer options (A, B, C, D)")
    correct_answer: str = Field(..., pattern="^[A-D]$", description="Correct answer letter")
    explanation: str = Field(..., min_length=5, description="Explanation of the answer")
    
    @field_validator('options')
    @classmethod
    def validate_options(cls, v):
        """Ensure options has exactly A, B, C, D keys."""
        required_keys = {'A', 'B', 'C', 'D'}
        if set(v.keys()) != required_keys:
            raise ValueError(f"Options must have exactly keys: {required_keys}, got {set(v.keys())}")
        # Ensure all values are non-empty
        for key, val in v.items():
            if not val or not val.strip():
                raise ValueError(f"Option {key} cannot be empty")
        return v

class Quiz(BaseModel):
    """Complete quiz output with validation."""
    quiz_title: str = Field(..., min_length=1, description="Title of the quiz")
    difficulty: str = Field(..., pattern="^(easy|medium|hard)$", description="Difficulty level")
    questions: List[QuizQuestion] = Field(..., min_length=1, max_length=20, description="Quiz questions")

class FlashcardCard(BaseModel):
    """Single flashcard with validation."""
    front: str = Field(..., min_length=3, description="Front of the flashcard")
    back: str = Field(..., min_length=3, description="Back of the flashcard")
    tags: Optional[List[str]] = Field(default_factory=list, description="Optional tags")

class Flashcards(BaseModel):
    """Complete flashcard deck with validation."""
    title: str = Field(..., min_length=1, description="Deck title")
    cards: List[FlashcardCard] = Field(..., min_length=1, max_length=50, description="Flashcards")

class MindmapNode(BaseModel):
    """Single node in a mindmap."""
    id: str = Field(..., description="Node ID")
    label: str = Field(..., min_length=1, description="Node label")
    children: Optional[List['MindmapNode']] = Field(default_factory=list, description="Child nodes")

class Mindmap(BaseModel):
    """Complete mindmap structure."""
    title: str = Field(..., min_length=1, description="Mindmap title")
    root: MindmapNode = Field(..., description="Root node")

# Enable forward references for recursive models
MindmapNode.model_rebuild()
