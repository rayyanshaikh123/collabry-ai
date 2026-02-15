"""
Verification Arbitrator - Resolves conflicts between validators.

Priority Order:
1. ArithmeticValidator (highest - deterministic)
2. VerifiedKnowledgeRetrieval (high - authoritative)
3. LogicalConsistencyChecker (medium - rule-based)
4. MisinformationPatternDetector (low - pattern-based)
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
from core.verification.validators import Claim, ValidationResult, ConsistencyReport, MisinformationFlag
from core.verification.confidence_scorer import VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class FinalVerdict:
    """Final verdict after arbitration."""
    status: str  # verified | unsupported | conflict | insufficient_data
    reason: str
    confidence: float
    source: str


class VerificationArbitrator:
    """
    Resolves conflicts when different validators disagree.
    
    Example conflict:
    - claim_verification → "verified" (found in verified knowledge)
    - arithmetic_validator → False (15 × 17 ≠ 255)
    - Resolution: Arithmetic wins (deterministic > retrieval)
    """
    
    def arbitrate(
        self,
        claim: Claim,
        arithmetic_result: ValidationResult,
        verification_result: VerificationResult,
        consistency_result: ConsistencyReport,
        misinformation_flags: List[MisinformationFlag]
    ) -> FinalVerdict:
        """
        Arbitrate between potentially conflicting validation results.
        
        Args:
            claim: The claim being verified
            arithmetic_result: Result from arithmetic validator
            verification_result: Result from verified knowledge retrieval
            consistency_result: Result from logical consistency check
            misinformation_flags: Flags from misinformation detector
        
        Returns:
            FinalVerdict with final status and reasoning
        """
        
        # RULE 1: Arithmetic validator always wins for numeric claims
        if arithmetic_result.valid is not None:
            if not arithmetic_result.valid:
                logger.info(f"⚖️ Arbitration: Arithmetic validator wins (invalid)")
                return FinalVerdict(
                    status="unsupported",
                    reason=f"Arithmetic error: {arithmetic_result.reason}",
                    confidence=0.95,
                    source="arithmetic_validator"
                )
            else:
                logger.info(f"⚖️ Arbitration: Arithmetic validator confirms valid")
                # Arithmetic is valid, continue to other checks
        
        # RULE 2: Verified knowledge takes precedence for non-numeric claims
        if verification_result.status == "verified" and verification_result.authority_level == "high":
            # But check for logical contradictions
            if consistency_result.contradictions:
                logger.info(f"⚖️ Arbitration: Conflict detected despite verification")
                return FinalVerdict(
                    status="conflict",
                    reason="Verified by authoritative source but contradicts other claims",
                    confidence=0.5,
                    source="arbitrator"
                )
            
            logger.info(f"⚖️ Arbitration: High authority verification wins")
            return FinalVerdict(
                status="verified",
                reason=f"Verified by {verification_result.source}",
                confidence=0.9,
                source="verified_knowledge"
            )
        
        # RULE 3: Misinformation patterns override weak verification
        strong_misinformation = [f for f in misinformation_flags if f.confidence > 0.8]
        if strong_misinformation and verification_result.authority_level != "high":
            logger.info(f"⚖️ Arbitration: Misinformation pattern overrides weak verification")
            return FinalVerdict(
                status="unsupported",
                reason=f"Misinformation pattern detected: {strong_misinformation[0].type}",
                confidence=0.7,
                source="misinformation_detector"
            )
        
        # RULE 4: Default to verification result
        logger.info(f"⚖️ Arbitration: Default to verification result ({verification_result.status})")
        return FinalVerdict(
            status=verification_result.status,
            reason=verification_result.reason or f"Status: {verification_result.status}",
            confidence=0.6,
            source="verified_knowledge"
        )
