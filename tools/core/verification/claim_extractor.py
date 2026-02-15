"""
Hybrid Claim Extractor - Combines deterministic extraction with LLM for best results.

Deterministic extraction (regex, SymPy) for:
- Numeric claims (15 × 17 = 255)
- Equations (E = mc²)
- Dates (1947, 15th August)
- Definitions

LLM extraction only for:
- Conceptual claims
- Nuanced statements
"""

import re
import logging
from typing import List, Dict, Any, Optional
from core.llm import chat_completion
from core.verification.validators import Claim, ClaimType

logger = logging.getLogger(__name__)


class HybridClaimExtractor:
    """Hybrid claim extraction using deterministic + LLM approaches."""
    
    def __init__(self):
        self.max_claims = 10  # Limit to prevent explosion
    
    async def extract_claims(
        self,
        answer: str,
        syllabus_context: Optional[Dict[str, Any]] = None
    ) -> List[Claim]:
        """
        Extract claims from an answer using hybrid approach.
        
        Args:
            answer: The answer text to extract claims from
            syllabus_context: Optional syllabus context
        
        Returns:
            List of extracted claims (max 10)
        """
        all_claims = []
        
        # PHASE 1: Deterministic extraction (0 LLM calls)
        numeric_claims = self._extract_numeric_claims(answer)
        equation_claims = self._extract_equations(answer)
        date_claims = self._extract_dates(answer)
        definition_claims = self._extract_definitions(answer)
        
        all_claims.extend(numeric_claims)
        all_claims.extend(equation_claims)
        all_claims.extend(date_claims)
        all_claims.extend(definition_claims)
        
        # PHASE 2: LLM extraction for conceptual claims (1 LLM call)
        # Only if we haven't hit the limit
        if len(all_claims) < self.max_claims:
            remaining_slots = self.max_claims - len(all_claims)
            conceptual_claims = await self._llm_extract_conceptual(
                answer,
                max_claims=remaining_slots
            )
            all_claims.extend(conceptual_claims)
        
        # PHASE 3: Deduplicate and prioritize
        deduplicated = self._deduplicate_claims(all_claims)
        
        # Limit to max_claims
        return deduplicated[:self.max_claims]
    
    def _extract_numeric_claims(self, text: str) -> List[Claim]:
        """
        Extract arithmetic/numeric statements using regex.
        
        Examples:
            "15 × 17 = 255" -> Claim
            "2 + 3 = 5" -> Claim
        """
        claims = []
        
        # Patterns for arithmetic operations
        patterns = [
            r'(\d+)\s*[×*]\s*(\d+)\s*=\s*(\d+)',  # Multiplication
            r'(\d+)\s*[+]\s*(\d+)\s*=\s*(\d+)',   # Addition
            r'(\d+)\s*[-−]\s*(\d+)\s*=\s*(\d+)',  # Subtraction
            r'(\d+)\s*[/÷]\s*(\d+)\s*=\s*(\d+)',  # Division
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                claim_text = match.group(0)
                claims.append(Claim(
                    text=claim_text,
                    type=ClaimType.NUMERIC,
                    metadata={"extraction_method": "regex", "pattern": pattern}
                ))
        
        return claims
    
    def _extract_equations(self, text: str) -> List[Claim]:
        """
        Extract mathematical equations using pattern matching.
        
        Examples:
            "E = mc²" -> Claim
            "F = ma" -> Claim
        """
        claims = []
        
        # Pattern for simple equations (variable = expression)
        equation_pattern = r'\b([A-Z][a-z]?)\s*=\s*([A-Za-z0-9²³⁴\s+\-*/()]+)\b'
        
        matches = re.finditer(equation_pattern, text)
        for match in matches:
            claim_text = match.group(0)
            # Filter out numeric equations (already caught above)
            if not re.match(r'^\d+\s*=', claim_text):
                claims.append(Claim(
                    text=claim_text,
                    type=ClaimType.EQUATION,
                    metadata={"extraction_method": "regex"}
                ))
        
        return claims
    
    def _extract_dates(self, text: str) -> List[Claim]:
        """
        Extract date claims using regex.
        
        Examples:
            "India gained independence in 1947" -> Claim
            "15th August 1947" -> Claim
        """
        claims = []
        
        # Patterns for dates
        date_patterns = [
            r'\b(in\s+)?(\d{4})\b',  # Year
            r'\b(\d{1,2})(st|nd|rd|th)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract surrounding context (sentence)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                claims.append(Claim(
                    text=context,
                    type=ClaimType.DATE,
                    metadata={"extraction_method": "regex", "date": match.group(0)}
                ))
        
        return claims
    
    def _extract_definitions(self, text: str) -> List[Claim]:
        """
        Extract definition claims using pattern matching.
        
        Examples:
            "Photosynthesis is defined as..." -> Claim
            "X is the process of..." -> Claim
        """
        claims = []
        
        # Patterns for definitions
        definition_patterns = [
            r'([A-Z][a-z]+)\s+is\s+defined\s+as\s+([^.]+)\.',
            r'([A-Z][a-z]+)\s+is\s+the\s+([^.]+)\.',
            r'([A-Z][a-z]+)\s+refers\s+to\s+([^.]+)\.',
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                claim_text = match.group(0)
                claims.append(Claim(
                    text=claim_text,
                    type=ClaimType.DEFINITION,
                    metadata={"extraction_method": "regex", "term": match.group(1)}
                ))
        
        return claims
    
    async def _llm_extract_conceptual(
        self,
        answer: str,
        max_claims: int = 10
    ) -> List[Claim]:
        """
        Use LLM to extract conceptual claims.
        
        This is the ONLY LLM call in claim extraction.
        """
        try:
            prompt = f"""Extract the key factual claims from the following answer. 
Focus on conceptual statements, not arithmetic or equations (those are handled separately).

Answer: {answer}

Extract up to {max_claims} conceptual claims. Return them as a JSON array of strings.
Example: ["Claim 1", "Claim 2", "Claim 3"]

Only return the JSON array, nothing else."""

            messages = [{"role": "user", "content": prompt}]
            
            response = await chat_completion(messages, stream=False)
            content = response.choices[0].message.content.strip()
            
            # Parse JSON
            import json
            claims_list = json.loads(content)
            
            # Convert to Claim objects
            claims = []
            for claim_text in claims_list[:max_claims]:
                claims.append(Claim(
                    text=claim_text,
                    type=ClaimType.CONCEPTUAL,
                    metadata={"extraction_method": "llm"}
                ))
            
            logger.info(f"✅ Extracted {len(claims)} conceptual claims via LLM")
            return claims
        
        except Exception as e:
            logger.error(f"LLM claim extraction failed: {e}")
            return []
    
    def _deduplicate_claims(self, claims: List[Claim]) -> List[Claim]:
        """
        Deduplicate similar claims.
        
        Simple implementation: exact text match.
        Production: use semantic similarity.
        """
        seen_texts = set()
        deduplicated = []
        
        for claim in claims:
            normalized_text = claim.text.lower().strip()
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                deduplicated.append(claim)
        
        return deduplicated
