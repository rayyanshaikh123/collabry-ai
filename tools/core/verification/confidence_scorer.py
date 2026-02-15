"""
Confidence Scorer - Deterministic mathematical formula for confidence calculation.
"""

import logging
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying a claim against verified knowledge."""
    status: str  # verified | partially_supported | unsupported | insufficient_data
    source: str = ""
    authority_level: str = None  # high | medium | low | None
    coverage: float = 0.0  # 0.0 - 1.0
    reason: str = ""


class ConfidenceScorer:
    """
    Deterministic confidence calculation with clear mathematical formula.
    
    Formula:
        confidence = 
            0.35 * verified_ratio +
            0.25 * authority_weight +
            0.15 * logical_consistency_score +
            0.10 * arithmetic_validity_score +
            0.10 * (1 - misinformation_score) +
            0.05 * verified_knowledge_coverage
    
    Range: 0.0 - 1.0 (converted to 0-100 for display)
    """
    
    # Authority level scores
    AUTHORITY_SCORES = {
        "high": 1.0,
        "medium": 0.6,
        "low": 0.3,
        None: 0.0
    }
    
    def calculate_confidence(
        self,
        verification_results: List[VerificationResult],
        arithmetic_valid: bool,
        logical_consistent: bool,
        misinformation_flags: List,
        verified_knowledge_coverage: float
    ) -> float:
        """
        Calculate confidence score using deterministic formula.
        
        Args:
            verification_results: List of verification results for claims
            arithmetic_valid: Whether arithmetic validation passed
            logical_consistent: Whether logical consistency check passed
            misinformation_flags: List of misinformation flags
            verified_knowledge_coverage: Coverage of verified knowledge (0.0-1.0)
        
        Returns:
            Confidence score (0-100)
        """
        total_claims = len(verification_results)
        
        if total_claims == 0:
            # No claims to verify - return moderate confidence
            return 50.0
        
        # 1. Verified Ratio (35%)
        verified_count = sum(
            1 for r in verification_results 
            if r.status == "verified"
        )
        verified_ratio = verified_count / total_claims
        
        # 2. Authority Weight (25%)
        authority_weight = sum(
            self.AUTHORITY_SCORES.get(r.authority_level, 0.0)
            for r in verification_results
        ) / total_claims
        
        # 3. Logical Consistency (15%)
        logical_consistency_score = 1.0 if logical_consistent else 0.0
        
        # 4. Arithmetic Validity (10%)
        arithmetic_validity_score = 1.0 if arithmetic_valid else 0.0
        
        # 5. Misinformation Score (10% - inverted)
        misinformation_score = sum(
            f.confidence for f in misinformation_flags
        ) / total_claims if total_claims > 0 else 0.0
        misinformation_score = min(misinformation_score, 1.0)  # Cap at 1.0
        
        # 6. Verified Knowledge Coverage (5%)
        # This component doesn't penalize heavily if verified knowledge is sparse
        coverage_score = verified_knowledge_coverage
        
        # FINAL CALCULATION
        confidence = (
            0.35 * verified_ratio +
            0.25 * authority_weight +
            0.15 * logical_consistency_score +
            0.10 * arithmetic_validity_score +
            0.10 * (1.0 - misinformation_score) +
            0.05 * coverage_score
        )
        
        # Convert to 0-100 scale and round
        confidence_percentage = round(confidence * 100, 1)
        
        logger.info(
            f"📊 Confidence: {confidence_percentage}% "
            f"(verified={verified_ratio:.2f}, authority={authority_weight:.2f}, "
            f"logic={logical_consistency_score:.2f}, arithmetic={arithmetic_validity_score:.2f}, "
            f"misinfo={misinformation_score:.2f}, coverage={coverage_score:.2f})"
        )
        
        return confidence_percentage
