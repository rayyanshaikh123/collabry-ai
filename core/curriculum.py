"""
Curriculum Manager - Structured Sequential Teaching Content
Treats knowledge as predetermined topic progression (NOT semantic search per message)
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class Subtopic:
    """Atomic teachable concept (5-10 minutes)"""
    id: str
    name: str
    content: str  # Core explanation text (200-300 words)
    key_concepts: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # "easy", "medium", "hard"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "key_concepts": self.key_concepts,
            "examples": self.examples,
            "difficulty": self.difficulty
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Subtopic":
        return cls(**data)


@dataclass
class Topic:
    """Main topic with multiple subtopics"""
    id: str
    name: str
    subtopics: List[Subtopic]
    prerequisites: List[str] = field(default_factory=list)  # Topic IDs required before this
    estimated_minutes: int = 30
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "subtopics": [s.to_dict() for s in self.subtopics],
            "prerequisites": self.prerequisites,
            "estimated_minutes": self.estimated_minutes
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Topic":
        data["subtopics"] = [Subtopic.from_dict(s) for s in data.get("subtopics", [])]
        return cls(**data)


@dataclass
class LessonPlan:
    """Complete structured curriculum for a subject"""
    id: str
    notebook_id: str
    title: str
    description: str
    topics: List[Topic]
    total_estimated_minutes: int = 0
    
    def __post_init__(self):
        # Calculate total time
        if self.total_estimated_minutes == 0:
            self.total_estimated_minutes = sum(t.estimated_minutes for t in self.topics)
    
    def get_topic_by_id(self, topic_id: str) -> Optional[Topic]:
        """Find topic by ID"""
        for topic in self.topics:
            if topic.id == topic_id:
                return topic
        return None
    
    def get_subtopic_by_id(self, subtopic_id: str) -> Optional[Subtopic]:
        """Find subtopic by ID across all topics"""
        for topic in self.topics:
            for subtopic in topic.subtopics:
                if subtopic.id == subtopic_id:
                    return subtopic
        return None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "notebook_id": self.notebook_id,
            "title": self.title,
            "description": self.description,
            "topics": [t.to_dict() for t in self.topics],
            "total_estimated_minutes": self.total_estimated_minutes
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "LessonPlan":
        data["topics"] = [Topic.from_dict(t) for t in data.get("topics", [])]
        return cls(**data)
    
    def save(self, filepath: str):
        """Save lesson plan to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved lesson plan to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "LessonPlan":
        """Load lesson plan from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class Question:
    """Single pedagogical question"""
    id: str
    text: str
    expected_answer: str
    difficulty: str  # "easy", "medium", "hard"
    question_type: str  # "factual", "conceptual", "application"
    hints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "expected_answer": self.expected_answer,
            "difficulty": self.difficulty,
            "type": self.question_type,
            "hints": self.hints
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Question":
        # Handle both "type" and "question_type" keys
        if "type" in data and "question_type" not in data:
            data["question_type"] = data.pop("type")
        return cls(**data)


class CurriculumManager:
    """
    Manages structured curriculum - sequential topic progression
    NO semantic search per message - deterministic teaching flow
    """
    
    def __init__(self, notebook_id: str, curricula_dir: Optional[str] = None):
        self.notebook_id = notebook_id
        self.curricula_dir = curricula_dir or os.path.join(
            os.path.dirname(__file__), "..", "data", "curricula"
        )
        self.questions_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "questions"
        )
        
        # Ensure directories exist
        Path(self.curricula_dir).mkdir(parents=True, exist_ok=True)
        Path(self.questions_dir).mkdir(parents=True, exist_ok=True)
        
        self.lesson_plan: Optional[LessonPlan] = None
        self.asked_questions: Set[str] = set()  # Track asked question IDs
        
        logger.info(f"CurriculumManager initialized for notebook: {notebook_id}")
    
    def load_curriculum(self) -> LessonPlan:
        """
        Load or create curriculum for this notebook
        Returns structured lesson plan
        """
        curriculum_file = os.path.join(self.curricula_dir, f"{self.notebook_id}.json")
        
        if os.path.exists(curriculum_file):
            logger.info(f"Loading existing curriculum from {curriculum_file}")
            self.lesson_plan = LessonPlan.load(curriculum_file)
        else:
            logger.info(f"No curriculum found, loading default template")
            # Load default template (photosynthesis example)
            self.lesson_plan = self._load_default_curriculum()
            # Save for future use
            self.lesson_plan.save(curriculum_file)
        
        return self.lesson_plan
    
    def _load_default_curriculum(self) -> LessonPlan:
        """
        Load default curriculum template
        For MVP: Hard-coded photosynthesis curriculum
        Future: Generate from RAG documents
        """
        return LessonPlan(
            id=f"curriculum_{self.notebook_id}",
            notebook_id=self.notebook_id,
            title="Introduction to Photosynthesis",
            description="Comprehensive guide to understanding photosynthesis in plants",
            topics=[
                Topic(
                    id="topic_1",
                    name="What is Photosynthesis?",
                    subtopics=[
                        Subtopic(
                            id="subtopic_1_1",
                            name="Definition and Overview",
                            content="""Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. This process is fundamental to life on Earth as it produces oxygen and organic compounds that other organisms depend on. The word photosynthesis comes from Greek: 'photo' meaning light and 'synthesis' meaning putting together. Plants use sunlight, water, and carbon dioxide to create glucose and oxygen.""",
                            key_concepts=["light energy", "chemical energy", "glucose", "oxygen production"],
                            examples=["Plants in your garden", "Trees in forests", "Algae in oceans"],
                            difficulty="easy"
                        ),
                        Subtopic(
                            id="subtopic_1_2",
                            name="Where Photosynthesis Occurs",
                            content="""Photosynthesis takes place primarily in the leaves of plants, specifically in specialized organelles called chloroplasts. Chloroplasts contain chlorophyll, a green pigment that absorbs light energy. The structure of a leaf is optimized for photosynthesis with a large surface area to capture sunlight and pores called stomata that allow gas exchange. Each chloroplast has internal membrane structures called thylakoids where the light-dependent reactions occur.""",
                            key_concepts=["chloroplasts", "chlorophyll", "leaves", "stomata", "thylakoids"],
                            examples=["Leaf cross-section", "Chloroplast structure"],
                            difficulty="medium"
                        )
                    ],
                    prerequisites=[],
                    estimated_minutes=15
                ),
                Topic(
                    id="topic_2",
                    name="The Photosynthesis Equation",
                    subtopics=[
                        Subtopic(
                            id="subtopic_2_1",
                            name="Chemical Equation",
                            content="""The overall equation for photosynthesis is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. This means six molecules of carbon dioxide plus six molecules of water, using light energy, produce one molecule of glucose and six molecules of oxygen. This is a simplified representation of a complex series of chemical reactions. Understanding this equation helps us see the inputs and outputs of the process.""",
                            key_concepts=["carbon dioxide", "water", "glucose", "oxygen", "chemical equation"],
                            examples=["Balanced chemical equation", "Molecular models"],
                            difficulty="medium"
                        )
                    ],
                    prerequisites=["topic_1"],
                    estimated_minutes=10
                ),
                Topic(
                    id="topic_3",
                    name="Light-Dependent Reactions",
                    subtopics=[
                        Subtopic(
                            id="subtopic_3_1",
                            name="Light Absorption and Electron Transport",
                            content="""The light-dependent reactions occur in the thylakoid membranes of chloroplasts. When light hits chlorophyll, it excites electrons to a higher energy state. These energized electrons move through an electron transport chain, releasing energy used to pump hydrogen ions across the membrane. This creates a concentration gradient that drives ATP synthesis. Water molecules are split to replace the electrons, releasing oxygen as a byproduct.""",
                            key_concepts=["electron transport chain", "ATP synthesis", "water splitting", "oxygen release"],
                            examples=["Thylakoid membrane diagram", "Electron flow"],
                            difficulty="hard"
                        )
                    ],
                    prerequisites=["topic_1", "topic_2"],
                    estimated_minutes=15
                )
            ]
        )
    
    def get_topic_content(self, subtopic_id: str) -> str:
        """
        Get explanation content for a subtopic
        NO semantic search - direct lookup
        """
        if not self.lesson_plan:
            self.load_curriculum()
        
        subtopic = self.lesson_plan.get_subtopic_by_id(subtopic_id)
        if subtopic:
            return subtopic.content
        
        logger.warning(f"Subtopic {subtopic_id} not found")
        return ""
    
    def get_question(self, subtopic_id: str, difficulty: str = "medium") -> Optional[Dict[str, Any]]:
        """
        Get a question for a subtopic at specified difficulty
        Returns structured question data
        """
        question_bank = self._load_question_bank(subtopic_id)
        
        if not question_bank:
            logger.warning(f"No questions found for subtopic {subtopic_id}")
            return None
        
        # Filter by difficulty
        matching_questions = [q for q in question_bank if q.difficulty == difficulty]
        
        # Fallback to any difficulty if none match
        if not matching_questions:
            matching_questions = question_bank
        
        # Return first unasked question
        for question in matching_questions:
            if question.id not in self.asked_questions:
                self.asked_questions.add(question.id)
                return {
                    "id": question.id,
                    "question": question.text,
                    "answer": question.expected_answer,
                    "type": question.question_type,
                    "hints": question.hints
                }
        
        # All questions asked, reset and return first
        if question_bank:
            self.asked_questions.clear()
            q = question_bank[0]
            self.asked_questions.add(q.id)
            return {
                "id": q.id,
                "question": q.text,
                "answer": q.expected_answer,
                "type": q.question_type,
                "hints": q.hints
            }
        
        return None
    
    def _load_question_bank(self, subtopic_id: str) -> List[Question]:
        """Load questions for a subtopic from JSON file"""
        question_file = os.path.join(self.questions_dir, f"{subtopic_id}.json")
        
        if os.path.exists(question_file):
            with open(question_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [Question.from_dict(q) for q in data]
        else:
            # Create default questions
            logger.info(f"Generating default questions for {subtopic_id}")
            return self._generate_default_questions(subtopic_id)
    
    def _generate_default_questions(self, subtopic_id: str) -> List[Question]:
        """
        Generate default questions for a subtopic
        For MVP: Hard-coded questions
        Future: LLM-generated question bank
        """
        # Default questions for photosynthesis topics
        default_banks = {
            "subtopic_1_1": [
                Question(
                    id="q_1_1_1",
                    text="What is photosynthesis?",
                    expected_answer="Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose.",
                    difficulty="easy",
                    question_type="factual",
                    hints=["Think about what plants do with sunlight", "It involves converting energy"]
                ),
                Question(
                    id="q_1_1_2",
                    text="What are the main products of photosynthesis?",
                    expected_answer="The main products are glucose (for energy storage) and oxygen.",
                    difficulty="easy",
                    question_type="factual",
                    hints=["Think about what plants produce", "One is a gas we breathe"]
                )
            ],
            "subtopic_1_2": [
                Question(
                    id="q_1_2_1",
                    text="Where in the plant cell does photosynthesis occur?",
                    expected_answer="Photosynthesis occurs in the chloroplasts.",
                    difficulty="medium",
                    question_type="factual",
                    hints=["It's an organelle", "It contains chlorophyll"]
                )
            ],
            "subtopic_2_1": [
                Question(
                    id="q_2_1_1",
                    text="What are the three main inputs needed for photosynthesis?",
                    expected_answer="The three inputs are carbon dioxide, water, and light energy.",
                    difficulty="medium",
                    question_type="conceptual",
                    hints=["Check the equation", "Think about what plants take in"]
                )
            ]
        }
        
        questions = default_banks.get(subtopic_id, [
            Question(
                id=f"q_{subtopic_id}_default",
                text=f"Can you explain what you learned about {subtopic_id}?",
                expected_answer="Various acceptable answers",
                difficulty="medium",
                question_type="conceptual",
                hints=["Think about the key concepts", "What was most important?"]
            )
        ])
        
        # Save for future use
        question_file = os.path.join(self.questions_dir, f"{subtopic_id}.json")
        with open(question_file, 'w', encoding='utf-8') as f:
            json.dump([q.to_dict() for q in questions], f, indent=2)
        
        return questions
    
    def reset_asked_questions(self):
        """Reset the set of asked questions (for new session)"""
        self.asked_questions.clear()
