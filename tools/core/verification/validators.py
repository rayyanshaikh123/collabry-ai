"""
Deterministic Validators - No LLM calls.

This module contains validators that use deterministic algorithms (SymPy, regex, etc.)
to validate claims without relying on LLMs, preventing hallucination.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ClaimType(Enum):
    """Types of claims that can be extracted."""
    NUMERIC = "numeric"
    EQUATION = "equation"
    DATE = "date"
    DEFINITION = "definition"
    CONCEPTUAL = "conceptual"
    UNKNOWN = "unknown"


@dataclass
class Claim:
    """Represents an atomic claim extracted from an answer."""
    text: str
    type: ClaimType
    metadata: Dict[str, Any]
    
    def __repr__(self):
        return f"Claim(type={self.type.value}, text='{self.text[:50]}...')"


@dataclass
class ValidationResult:
    """Result of validating a claim."""
    valid: Optional[bool]  # None if not applicable
    reason: str
    computed_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    confidence: float = 1.0


@dataclass
class ConsistencyReport:
    """Report on logical consistency of claims."""
    contradictions: List[Tuple[Claim, Claim]]
    domain_mismatches: List[Dict[str, Any]]
    passed: bool
    details: str = ""


@dataclass
class MisinformationFlag:
    """Flag for detected misinformation patterns."""
    claim: Claim
    type: str  # viral_shortcut, overgeneralization, etc.
    confidence: float
    reason: str


class ArithmeticValidator:
    """Validate arithmetic claims using SymPy - NEVER LLM."""
    
    def __init__(self):
        try:
            from sympy import sympify
            self.sympify = sympify
            self.available = True
        except ImportError:
            logger.warning("SymPy not available. Arithmetic validation will be limited.")
            self.available = False
    
    def validate_claim(self, claim: Claim) -> ValidationResult:
        """
        Validate an arithmetic claim.
        
        Args:
            claim: Claim to validate
        
        Returns:
            ValidationResult with validation status
        """
        if claim.type != ClaimType.NUMERIC:
            return ValidationResult(
                valid=None,
                reason="Not a numeric claim"
            )
        
        if not self.available:
            return ValidationResult(
                valid=None,
                reason="SymPy not available for validation"
            )
        
        # Parse equation from claim text
        equation_parts = self._parse_equation(claim.text)
        
        if not equation_parts:
            return ValidationResult(
                valid=None,
                reason="Could not parse equation"
            )
        
        left_expr, right_expr = equation_parts
        
        try:
            # Evaluate both sides using SymPy
            left_value = self.sympify(left_expr).evalf()
            right_value = self.sympify(right_expr).evalf()
            
            # Check if they're equal (within floating point tolerance)
            is_valid = abs(float(left_value) - float(right_value)) < 1e-6
            
            return ValidationResult(
                valid=is_valid,
                reason=f"Computed {left_value}, expected {right_value}",
                computed_value=float(left_value),
                expected_value=float(right_value),
                confidence=0.95
            )
        
        except Exception as e:
            logger.error(f"SymPy evaluation error: {e}")
            return ValidationResult(
                valid=False,
                reason=f"Evaluation error: {str(e)}",
                confidence=0.5
            )
    
    def _parse_equation(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Parse equation from text.
        
        Examples:
            "15 × 17 = 255" -> ("15*17", "255")
            "2 + 3 = 5" -> ("2+3", "5")
        
        Returns:
            Tuple of (left_expression, right_expression) or None
        """
        # Replace common symbols with Python operators
        text = text.replace("×", "*").replace("÷", "/").replace("−", "-")
        
        # Try to find equation pattern
        equation_pattern = r'([^=]+)=([^=]+)'
        match = re.search(equation_pattern, text)
        
        if match:
            left = match.group(1).strip()
            right = match.group(2).strip()
            return (left, right)
        
        return None


class LogicalConsistencyChecker:
    """Check for contradictions using rule-based logic."""
    
    def check_consistency(
        self,
        claims: List[Claim],
        syllabus_context: Optional[Dict[str, Any]] = None
    ) -> ConsistencyReport:
        """
        Check logical consistency across claims.
        
        Args:
            claims: List of claims to check
            syllabus_context: Optional syllabus context for domain checking
        
        Returns:
            ConsistencyReport with findings
        """
        contradictions = []
        domain_mismatches = []
        
        # Check for direct contradictions
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                if self._are_contradictory(claim1, claim2):
                    contradictions.append((claim1, claim2))
        
        # Check domain consistency if syllabus context provided
        if syllabus_context:
            for claim in claims:
                mismatch = self._check_domain_mismatch(claim, syllabus_context)
                if mismatch:
                    domain_mismatches.append(mismatch)
        
        passed = len(contradictions) == 0 and len(domain_mismatches) == 0
        
        details = ""
        if contradictions:
            details += f"Found {len(contradictions)} contradictions. "
        if domain_mismatches:
            details += f"Found {len(domain_mismatches)} domain mismatches."
        
        return ConsistencyReport(
            contradictions=contradictions,
            domain_mismatches=domain_mismatches,
            passed=passed,
            details=details.strip()
        )
    
    def _are_contradictory(self, claim1: Claim, claim2: Claim) -> bool:
        """
        Check if two claims contradict each other.
        
        This is a simplified implementation. Production version would use
        more sophisticated NLP techniques or knowledge graphs.
        """
        # Simple keyword-based contradiction detection
        negation_patterns = [
            (r'\bnot\b', r'\bis\b'),
            (r'\bno\b', r'\byes\b'),
            (r'\bfalse\b', r'\btrue\b'),
            (r'\bincorrect\b', r'\bcorrect\b'),
        ]
        
        text1_lower = claim1.text.lower()
        text2_lower = claim2.text.lower()
        
        # Check for negation patterns
        for neg, pos in negation_patterns:
            if re.search(neg, text1_lower) and re.search(pos, text2_lower):
                # Check if they're talking about the same thing (simple overlap check)
                words1 = set(re.findall(r'\b\w+\b', text1_lower))
                words2 = set(re.findall(r'\b\w+\b', text2_lower))
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                if overlap > 0.5:
                    return True
        
        return False
    
    def _check_domain_mismatch(
        self,
        claim: Claim,
        syllabus_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if claim is in wrong domain (e.g., physics formula in biology).
        
        Returns:
            Mismatch details or None
        """
        # This is a placeholder - production version would use domain-specific
        # keyword dictionaries or ML models
        
        # Example: Check if physics terms appear in non-physics context
        physics_keywords = ["force", "velocity", "acceleration", "momentum", "energy"]
        biology_keywords = ["cell", "organism", "photosynthesis", "mitosis"]
        
        subject = syllabus_context.get("subject", "").lower()
        claim_text = claim.text.lower()
        
        if subject == "biology":
            for keyword in physics_keywords:
                if keyword in claim_text:
                    return {
                        "claim": claim,
                        "expected_domain": "biology",
                        "detected_domain": "physics",
                        "keyword": keyword
                    }
        
        return None


class MisinformationPatternDetector:
    """Pattern-based misinformation detection - NO LLM."""
    
    # Viral shortcut patterns (common exam "hacks" that are incorrect)
    VIRAL_SHORTCUTS = [
        "memorize all formulas in 5 minutes",
        "one trick to ace exams",
        "teachers don't want you to know",
        "secret formula for",
        "guaranteed 100% marks",
        "no need to study",
        "shortcut to solve all",
        "magic trick for",
    ]
    
    # Overgeneralization indicators
    OVERGENERALIZATION_WORDS = [
        "always", "never", "all", "none", "every", "no one",
        "everyone", "everything", "nothing", "100%", "impossible"
    ]
    
    def detect_patterns(self, claims: List[Claim]) -> List[MisinformationFlag]:
        """
        Detect misinformation patterns in claims.
        
        Args:
            claims: List of claims to check
        
        Returns:
            List of misinformation flags
        """
        flags = []
        
        for claim in claims:
            # Check viral shortcuts
            viral_flag = self._check_viral_shortcuts(claim)
            if viral_flag:
                flags.append(viral_flag)
            
            # Check overgeneralization
            overgen_flag = self._check_overgeneralization(claim)
            if overgen_flag:
                flags.append(overgen_flag)
        
        return flags
    
    def _check_viral_shortcuts(self, claim: Claim) -> Optional[MisinformationFlag]:
        """Check for viral shortcut patterns."""
        claim_text_lower = claim.text.lower()
        
        for pattern in self.VIRAL_SHORTCUTS:
            if pattern in claim_text_lower:
                return MisinformationFlag(
                    claim=claim,
                    type="viral_shortcut",
                    confidence=0.9,
                    reason=f"Detected viral shortcut pattern: '{pattern}'"
                )
        
        return None
    
    def _check_overgeneralization(self, claim: Claim) -> Optional[MisinformationFlag]:
        """Check for overgeneralization indicators."""
        claim_text_lower = claim.text.lower()
        
        # Count overgeneralization words
        count = sum(1 for word in self.OVERGENERALIZATION_WORDS if f" {word} " in f" {claim_text_lower} ")
        
        if count >= 2:  # Multiple overgeneralization words
            return MisinformationFlag(
                claim=claim,
                type="overgeneralization",
                confidence=0.7,
                reason=f"Contains {count} overgeneralization indicators"
            )
        
        return None
