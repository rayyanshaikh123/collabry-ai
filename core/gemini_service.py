"""
Unified Gemini Service for Collabry AI Engine

ARCHITECTURE:
------------
This service replaces multiple specialized models with a single Gemini-powered pipeline:

OLD ARCHITECTURE:
- Ollama (Llama 3.1) → Chat/Generation
- spaCy en_core_web_sm → NER/Entity Extraction
- Custom Intent Classifier → Intent Routing
- FAISS + all-MiniLM-L6-v2 → Vector Search (PRESERVED)

NEW ARCHITECTURE:
- Google Gemini (1.5 Flash/Pro) → Unified reasoning engine for:
  * Chat and generation
  * Intent classification (replaces custom classifier)
  * Entity extraction (replaces spaCy)
  * Semantic parsing and routing
- FAISS + all-MiniLM-L6-v2 → Vector Search (PRESERVED)

REQUEST FLOW:
User Query → FAISS Retrieval → Gemini (single call) → Structured Response

BENEFITS:
- Single API call per request (reduced latency)
- No local model dependencies (Ollama, spaCy)
- Cost-free hosting on Google AI Studio
- Better reasoning and context understanding
- Structured outputs with JSON mode
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# Gemini SDK
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    logger.warning("google-genai not installed. Install with: pip install google-genai")
    GEMINI_AVAILABLE = False


@dataclass
class IntentResult:
    """Structured intent classification result"""
    intent: str  # chat, qa, summarize, explain, analyze, plan, generate
    confidence: float  # 0.0 - 1.0
    entities: Dict[str, Any]  # extracted entities
    tool_calls: List[str]  # suggested tools to invoke
    reasoning: str  # explanation of classification


@dataclass
class GeminiResponse:
    """Unified response from Gemini"""
    text: str
    intent: Optional[IntentResult] = None
    metadata: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None


class GeminiService:
    """
    Unified Gemini client for all AI operations.
    
    Replaces:
    - LocalLLM (Ollama)
    - IntentClassifier
    - spaCy NLP pipeline
    
    Single point of integration with Google Gemini API.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash-lite",
        temperature: float = 0.2,
        max_tokens: int = 8192,
        timeout: int = 120
    ):
        """
        Initialize Gemini service.
        
        Args:
            api_key: Google AI Studio API key
            model_name: gemini-2.0-flash-lite (fast), gemini-1.5-pro (accurate)
            temperature: 0.0-1.0 (lower = more deterministic)
            max_tokens: Max output tokens
            timeout: Request timeout in seconds
        """
        if not GEMINI_AVAILABLE:
            raise RuntimeError("google-genai package not installed")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        # Ensure model name has "models/" prefix for v1beta API
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)
        
        logger.info(f"✓ GeminiService initialized: {model_name}")
    
    # ================================================================
    # CORE GENERATION
    # ================================================================
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> str:
        """
        Generate text using Gemini.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions (for structured outputs)
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: Enable JSON structured output
            
        Returns:
            Generated text
        """
        try:
            config = types.GenerateContentConfig(
                temperature=temperature or self.temperature,
                max_output_tokens=max_tokens or self.max_tokens,
                response_mime_type="application/json" if json_mode else "text/plain"
            )
            
            # Combine system and user prompts
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=config
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"[Gemini] Generation failed: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """
        Stream text generation.
        
        Yields text chunks as they arrive.
        """
        try:
            config = types.GenerateContentConfig(
                temperature=temperature or self.temperature,
                max_output_tokens=self.max_tokens
            )
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=full_prompt,
                config=config
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"[Gemini] Streaming failed: {e}")
            raise
    
    # ================================================================
    # INTENT CLASSIFICATION (Replaces custom classifier)
    # ================================================================
    
    def classify_intent(
        self,
        query: str,
        context: Optional[str] = None
    ) -> IntentResult:
        """
        Classify user intent using Gemini.
        
        Replaces: IntentClassifier (HuggingFace model)
        
        Args:
            query: User query
            context: Optional conversation context
            
        Returns:
            IntentResult with classification and entities
        """
        system_prompt = """You are an intent classification system for a study AI assistant.

Classify the user's query into ONE of these intents:
- chat: casual conversation, greetings, general questions
- qa: specific questions requiring factual answers
- summarize: requests to summarize documents/topics
- explain: requests for explanations or clarifications
- analyze: requests for analysis or insights
- plan: study planning, scheduling, goal setting
- generate: content generation (quiz, mindmap, flashcards, notes)
- search: web search or external information lookup

Extract entities:
- topics: main subjects/topics mentioned
- document_refs: references to uploaded documents
- time_refs: time-related mentions (dates, deadlines)
- actions: specific actions requested

Suggest tools to invoke (if any):
- web_search: for current events, external information
- doc_generator: for document generation
- mindmap_generator: for concept mapping
- ppt_generator: for presentations
- read_file: to read uploaded documents
- write_file: to save generated content

Return ONLY valid JSON:
{
  "intent": "intent_name",
  "confidence": 0.0-1.0,
  "entities": {
    "topics": ["topic1", "topic2"],
    "document_refs": ["doc1.pdf"],
    "time_refs": ["tomorrow", "next week"],
    "actions": ["action1"]
  },
  "tool_calls": ["tool_name"],
  "reasoning": "brief explanation"
}"""
        
        full_query = query
        if context:
            full_query = f"Context: {context}\n\nQuery: {query}"
        
        try:
            result = self.generate(
                prompt=full_query,
                system_prompt=system_prompt,
                temperature=0.1,  # Low temperature for deterministic classification
                json_mode=True
            )
            
            data = json.loads(result)
            return IntentResult(
                intent=data.get("intent", "chat"),
                confidence=data.get("confidence", 0.8),
                entities=data.get("entities", {}),
                tool_calls=data.get("tool_calls", []),
                reasoning=data.get("reasoning", "")
            )
            
        except Exception as e:
            logger.error(f"[Gemini] Intent classification failed: {e}")
            # Fallback to chat intent
            return IntentResult(
                intent="chat",
                confidence=0.5,
                entities={},
                tool_calls=[],
                reasoning="Fallback due to classification error"
            )
    
    # ================================================================
    # ENTITY EXTRACTION (Replaces spaCy)
    # ================================================================
    
    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Extract named entities using Gemini.
        
        Replaces: spaCy en_core_web_sm NER
        
        Args:
            text: Text to analyze
            entity_types: Specific entity types to extract (optional)
            
        Returns:
            Dictionary of entity type -> list of entities
        """
        types_str = ", ".join(entity_types) if entity_types else "all relevant entities"
        
        system_prompt = f"""Extract {types_str} from the text.

Common entity types:
- PERSON: people names
- ORG: organizations, companies
- GPE: countries, cities, locations
- DATE: dates, times
- TOPIC: academic subjects, topics
- CONCEPT: key concepts, theories
- EVENT: events, milestones

Return ONLY valid JSON:
{{
  "PERSON": ["name1", "name2"],
  "ORG": ["org1"],
  "GPE": ["location1"],
  "DATE": ["date1"],
  "TOPIC": ["topic1", "topic2"],
  "CONCEPT": ["concept1"],
  "EVENT": ["event1"]
}}

Only include entity types that are found in the text."""
        
        try:
            result = self.generate(
                prompt=text,
                system_prompt=system_prompt,
                temperature=0.1,
                json_mode=True
            )
            
            return json.loads(result)
            
        except Exception as e:
            logger.error(f"[Gemini] Entity extraction failed: {e}")
            return {}
    
    # ================================================================
    # RAG-ENHANCED GENERATION
    # ================================================================
    
    def generate_with_context(
        self,
        query: str,
        retrieved_docs: List[str],
        intent: Optional[str] = None,
        system_instructions: Optional[str] = None
    ) -> GeminiResponse:
        """
        Generate response with RAG context.
        
        This is the UNIFIED pipeline that combines:
        - Intent understanding
        - Context retrieval (FAISS)
        - Response generation
        
        Args:
            query: User query
            retrieved_docs: Documents retrieved from FAISS
            intent: Pre-classified intent (optional)
            system_instructions: Additional instructions
            
        Returns:
            GeminiResponse with text and metadata
        """
        # Build context from retrieved documents
        context_block = ""
        if retrieved_docs:
            context_block = "RETRIEVED CONTEXT:\n" + "\n\n".join([
                f"[Document {i+1}]\n{doc}" 
                for i, doc in enumerate(retrieved_docs[:5])  # Limit to top 5
            ])
        
        # Build system prompt based on intent
        base_prompt = """You are an intelligent study assistant (Collabry).

Your capabilities:
- Answer questions accurately using provided context
- Summarize documents and topics concisely
- Explain complex concepts clearly
- Analyze information and provide insights
- Generate study materials (notes, quizzes, flashcards)
- Plan study schedules and track progress

Guidelines:
- Use the retrieved context when available
- If context is insufficient, acknowledge limitations
- Be concise and educational
- Format responses for readability (use markdown)
- Cite sources when using retrieved documents
- Don't hallucinate - only use provided information"""
        
        if system_instructions:
            base_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{system_instructions}"
        
        # Build full prompt
        full_prompt = ""
        if context_block:
            full_prompt += f"{context_block}\n\n"
        
        full_prompt += f"USER QUERY:\n{query}"
        
        if intent:
            full_prompt = f"[Intent: {intent}]\n\n{full_prompt}"
        
        try:
            response_text = self.generate(
                prompt=full_prompt,
                system_prompt=base_prompt,
                temperature=self.temperature
            )
            
            return GeminiResponse(
                text=response_text,
                metadata={
                    "intent": intent,
                    "docs_used": len(retrieved_docs),
                    "model": self.model_name
                }
            )
            
        except Exception as e:
            logger.error(f"[Gemini] RAG generation failed: {e}")
            raise
    
    # ================================================================
    # STRUCTURED OUTPUTS
    # ================================================================
    
    def generate_structured(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output.
        
        Useful for:
        - Quiz generation
        - Study plan creation
        - Flashcard generation
        
        Args:
            prompt: Generation prompt
            output_schema: Expected JSON schema (for documentation)
            system_prompt: System instructions
            
        Returns:
            Parsed JSON response
        """
        schema_str = json.dumps(output_schema, indent=2)
        
        structured_system = f"""Generate a response in VALID JSON format matching this schema:

{schema_str}

Ensure all required fields are present and properly formatted."""
        
        if system_prompt:
            structured_system = f"{system_prompt}\n\n{structured_system}"
        
        try:
            result = self.generate(
                prompt=prompt,
                system_prompt=structured_system,
                temperature=0.3,
                json_mode=True
            )
            
            return json.loads(result)
            
        except json.JSONDecodeError as e:
            logger.error(f"[Gemini] Invalid JSON output: {e}")
            raise ValueError(f"Failed to parse structured output: {e}")
        except Exception as e:
            logger.error(f"[Gemini] Structured generation failed: {e}")
            raise


# ================================================================
# FACTORY FUNCTION (LangChain compatibility)
# ================================================================

def create_gemini_service(config: Optional[Dict[str, Any]] = None) -> GeminiService:
    """
    Factory function to create GeminiService from config.
    
    Replaces: create_llm() from local_llm.py
    
    Args:
        config: Configuration dictionary (uses environment if None)
        
    Returns:
        Initialized GeminiService
    """
    if config is None:
        from config import CONFIG
        config = CONFIG
    
    api_key = config.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in config or environment")
    
    return GeminiService(
        api_key=api_key,
        model_name=config.get("gemini_model", "gemini-2.0-flash-lite"),
        temperature=config.get("temperature", 0.2),
        max_tokens=config.get("gemini_max_tokens", 8192),
        timeout=config.get("gemini_timeout", 120)
    )


# ================================================================
# BACKWARD COMPATIBILITY (LangChain LLM interface)
# ================================================================

try:
    from langchain_core.language_models.llms import LLM
    
    class GeminiLLM(LLM):
        """
        LangChain-compatible wrapper for GeminiService.
        
        Provides backward compatibility with existing LangChain code.
        """
        
        gemini_service: GeminiService
        
        @property
        def _llm_type(self) -> str:
            return "gemini"
        
        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            **kwargs
        ) -> str:
            return self.gemini_service.generate(prompt)
        
        def _stream(self, prompt: str, **kwargs):
            return self.gemini_service.generate_stream(prompt)
    
    def create_llm(config: Optional[Dict[str, Any]] = None) -> GeminiLLM:
        """
        Create LangChain-compatible Gemini LLM.
        
        Drop-in replacement for local_llm.create_llm()
        """
        service = create_gemini_service(config)
        return GeminiLLM(gemini_service=service)
    
except ImportError:
    logger.warning("LangChain not available - skipping LLM compatibility wrapper")
