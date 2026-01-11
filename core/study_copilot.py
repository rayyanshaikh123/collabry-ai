"""
Study Copilot - Pedagogical enhancements for the Collabry agent.

This module adds study-focused capabilities:
- Concept extraction from content
- Follow-up question generation
- Step-by-step explanations
- Clarifying question detection
"""
import json
import logging
from typing import List, Dict, Any, Optional
from core.local_llm import LocalLLM
from core.prompt_templates import (
    CONCEPT_EXTRACTION_PROMPT,
    FOLLOW_UP_QUESTIONS_PROMPT
)

logger = logging.getLogger(__name__)


class StudyCopilot:
    """
    Study-focused enhancements for the agent.
    
    Provides pedagogical capabilities like concept extraction,
    follow-up question generation, and learning guidance.
    """
    
    def __init__(self, llm: LocalLLM):
        """
        Initialize Study Copilot.
        
        Args:
            llm: Language model for generating study content
        """
        self.llm = llm
    
    def extract_concepts(self, content: str, max_concepts: int = 5) -> List[Dict[str, str]]:
        """
        Extract key concepts from content.
        
        Args:
            content: Text content to analyze
            max_concepts: Maximum number of concepts to extract
            
        Returns:
            List of concept dictionaries with name, definition, example, related
        """
        if not content or len(content.strip()) < 20:
            return []
        
        try:
            prompt = CONCEPT_EXTRACTION_PROMPT.format(content=content[:2000])  # Limit length
            prompt += f"\n\nExtract up to {max_concepts} most important concepts."
            
            response = self.llm.invoke(prompt)
            logger.debug(f"Concept extraction response: {response[:200]}")
            
            # Try to extract JSON array
            concepts = []
            if "[" in response:
                start = response.find("[")
                end = response.rfind("]") + 1
                if start != -1 and end > start:
                    try:
                        json_str = response[start:end]
                        # Clean up common LLM formatting issues
                        json_str = json_str.replace("\n", " ").replace("\r", "")
                        # Handle single quotes
                        json_str = json_str.replace("'", "\"")
                        concepts = json.loads(json_str)
                        logger.debug(f"Successfully parsed {len(concepts)} concepts")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse concepts JSON: {e}")
                        # Try to parse individual objects if array parse fails
                        import re
                        objects = re.findall(r'\{[^}]+\}', response)
                        logger.debug(f"Found {len(objects)} object patterns")
                        for obj_str in objects[:max_concepts]:
                            try:
                                # Fix common JSON issues
                                obj_str = obj_str.replace("'", "\"")
                                concepts.append(json.loads(obj_str))
                            except Exception as parse_err:
                                logger.debug(f"Failed to parse object: {parse_err}")
                                pass
            
            # If still no concepts, try to manually extract from text
            if not concepts:
                logger.warning("No JSON found, attempting manual extraction")
                # Simple fallback: extract from natural language if LLM didn't use JSON
                concepts = self._extract_concepts_from_text(response, max_concepts)
            
            # Validate structure
            valid_concepts = []
            for concept in concepts[:max_concepts]:
                if isinstance(concept, dict) and "name" in concept:
                    valid_concepts.append({
                        "name": concept.get("name", ""),
                        "definition": concept.get("definition", ""),
                        "example": concept.get("example", ""),
                        "related": concept.get("related", [])
                    })
            
            return valid_concepts
            
        except Exception as e:
            logger.exception(f"Concept extraction failed: {e}")
            return []
    
    def _extract_concepts_from_text(self, text: str, max_concepts: int) -> List[Dict[str, Any]]:
        """
        Fallback: Extract concepts from natural language text.
        
        Args:
            text: Text response from LLM
            max_concepts: Maximum concepts to extract
            
        Returns:
            List of concept dictionaries
        """
        concepts = []
        
        # Look for patterns like "**Concept Name**" or "Concept:"
        import re
        
        # Pattern 1: Look for bold or capitalized concept names
        name_patterns = [
            r'\*\*([A-Z][^\*]+)\*\*',  # **Concept Name**
            r'(?:^|\n)([A-Z][a-zA-Z\s]+):\s*([^\n]+)',  # Concept: definition
            r'(?:^|\n)\d+\.\s*([A-Z][a-zA-Z\s]+):\s*([^\n]+)',  # 1. Concept: definition
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            for match in matches[:max_concepts]:
                if isinstance(match, tuple) and len(match) >= 2:
                    name, definition = match[0], match[1]
                else:
                    name = match if isinstance(match, str) else str(match)
                    definition = ""
                
                concepts.append({
                    "name": name.strip(),
                    "definition": definition.strip() if definition else f"A concept related to the topic",
                    "example": "",
                    "related": []
                })
                
                if len(concepts) >= max_concepts:
                    break
            
            if len(concepts) >= max_concepts:
                break
        
        return concepts
    
    def generate_follow_up_questions(
        self,
        topic: str,
        explanation: str,
        count: int = 3
    ) -> List[str]:
        """
        Generate follow-up questions to deepen understanding.
        
        Args:
            topic: Main topic or question being discussed
            explanation: The explanation/answer provided
            count: Number of questions to generate
            
        Returns:
            List of follow-up questions
        """
        if not explanation or len(explanation.strip()) < 20:
            return self._default_follow_up_questions(topic)
        
        try:
            prompt = FOLLOW_UP_QUESTIONS_PROMPT.format(explanation=explanation[:1500])
            prompt += f"\n\nGenerate exactly {count} questions as a JSON array of strings."
            
            response = self.llm.invoke(prompt)
            
            # Extract JSON array
            questions = []
            if "[" in response:
                start = response.find("[")
                end = response.rfind("]") + 1
                if start != -1 and end > start:
                    try:
                        questions = json.loads(response[start:end])
                        # Filter to strings only
                        questions = [q for q in questions if isinstance(q, str)][:count]
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse follow-up questions JSON")
            
            # Fallback if parsing failed
            if not questions:
                questions = self._default_follow_up_questions(topic)
            
            return questions[:count]
            
        except Exception as e:
            logger.exception(f"Follow-up question generation failed: {e}")
            return self._default_follow_up_questions(topic)
    
    def _default_follow_up_questions(self, topic: str) -> List[str]:
        """Generate default follow-up questions based on topic."""
        return [
            f"Can you explain {topic} in simpler terms?",
            f"What are some real-world applications of {topic}?",
            f"How does {topic} relate to other concepts we've discussed?"
        ]
    
    def needs_clarification(self, user_input: str) -> Optional[str]:
        """
        Detect if user input is too vague and generate a clarifying question.
        
        Args:
            user_input: User's question or statement
            
        Returns:
            Clarifying question if needed, None if input is clear enough
        """
        user_lower = user_input.lower().strip()
        words = user_lower.split()
        
        # Check for vague pronouns (it, this, that)
        vague_pronouns = ["it", "this", "that", "these", "those"]
        
        # Pattern 1: "what is it", "tell me about it", etc.
        if any(pronoun in words for pronoun in vague_pronouns):
            # Check common vague patterns
            if "what is it" in user_lower or "what's it" in user_lower:
                return "What specific concept or topic are you asking about?"
            if "tell me about it" in user_lower or "explain it" in user_lower:
                return "What aspect would you like to learn about?"
            if "about it" in user_lower or user_lower.startswith("it "):
                return "I want to make sure I understand correctly. What are you referring to?"
        
        # Pattern 2: "explain this" without context
        if ("explain this" in user_lower or "what is this" in user_lower) and len(words) <= 3:
            return "What specifically would you like me to explain?"
        
        # Pattern 3: Very short questions with question words but no topic
        if len(words) < 3:
            if user_lower in ["why", "how", "what", "when", "where"]:
                return "Could you provide more context about your question?"
        
        return None
    
    def format_step_by_step(self, content: str, steps: List[str]) -> str:
        """
        Format content as step-by-step explanation.
        
        Args:
            content: Introduction/context
            steps: List of step descriptions
            
        Returns:
            Formatted step-by-step explanation
        """
        formatted = content.strip()
        if steps:
            formatted += "\n\n**Step-by-step:**\n"
            for i, step in enumerate(steps, 1):
                formatted += f"\n{i}. {step}"
        return formatted
    
    def add_pedagogical_context(self, answer: str, topic: str) -> Dict[str, Any]:
        """
        Enhance an answer with pedagogical elements.
        
        Args:
            answer: Base answer text
            topic: Topic being discussed
            
        Returns:
            Enhanced response with follow-up questions and metadata
        """
        # Generate follow-up questions
        follow_ups = self.generate_follow_up_questions(topic, answer)
        
        # Structure response
        enhanced = {
            "answer": answer,
            "follow_up_questions": follow_ups,
            "learning_tip": self._get_learning_tip()
        }
        
        return enhanced
    
    def _get_learning_tip(self) -> str:
        """Get a random learning tip."""
        tips = [
            "💡 Try explaining this concept in your own words to reinforce understanding.",
            "💡 Create a quick sketch or diagram to visualize the relationships.",
            "💡 Think of a personal example where this concept applies.",
            "💡 Quiz yourself: Can you explain this without looking at your notes?",
            "💡 Connect this to something you already know well.",
            "💡 Practice active recall: Close your notes and write what you remember.",
        ]
        import random
        return random.choice(tips)
    
    def is_study_intent(self, intent: Optional[str]) -> bool:
        """
        Check if the detected intent is study-related.
        
        Args:
            intent: Intent classification result
            
        Returns:
            True if study-related intent
        """
        if not intent:
            return False
        
        study_intents = [
            "explanation",
            "definition",
            "question",
            "learning",
            "study",
            "homework",
            "concept",
            "summary",
            "example"
        ]
        
        intent_lower = intent.lower()
        return any(study_word in intent_lower for study_word in study_intents)
    
    def suggest_study_tools(self, context: str) -> List[Dict[str, str]]:
        """
        Suggest relevant study tools based on context.
        
        Args:
            context: Current conversation context
            
        Returns:
            List of tool suggestions with descriptions
        """
        suggestions = []
        
        context_lower = context.lower()
        
        if "notes" in context_lower or "write down" in context_lower:
            suggestions.append({
                "tool": "write_file",
                "reason": "Save your notes for later review"
            })
        
        if "document" in context_lower or "study guide" in context_lower:
            suggestions.append({
                "tool": "doc_generator",
                "reason": "Create a formatted study document"
            })
        
        if "presentation" in context_lower or "slides" in context_lower:
            suggestions.append({
                "tool": "ppt_generator",
                "reason": "Build a presentation for your topic"
            })
        
        if "search" in context_lower or "find" in context_lower or "look up" in context_lower:
            suggestions.append({
                "tool": "web_search",
                "reason": "Search for additional resources"
            })
        
        if "website" in context_lower or "article" in context_lower or "url" in context_lower:
            suggestions.append({
                "tool": "web_scrape",
                "reason": "Extract content from a specific webpage"
            })
        
        return suggestions[:3]  # Limit to top 3 suggestions
