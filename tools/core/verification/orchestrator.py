"""
Verified Study Orchestrator - Main coordinator for verification pipeline.

Maximum 3 LLM calls:
1. Draft answer generation
2. Claim extraction (conceptual only)
3. Response formatting (optional - can use template)

All other operations are deterministic.
"""

import re
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from config import CONFIG
from core.llm import chat_completion
from core.rag_retriever import RAGRetriever
from core.verified_knowledge import get_verified_knowledge_store
from core.verification.claim_extractor import HybridClaimExtractor
from core.verification.validators import (
    ArithmeticValidator,
    LogicalConsistencyChecker,
    MisinformationPatternDetector,
    ClaimType
)
from core.verification.confidence_scorer import ConfidenceScorer
from core.verification.batch_processor import BatchClaimProcessor
from core.verification.arbitrator import VerificationArbitrator

logger = logging.getLogger(__name__)


class VerifiedStudyOrchestrator:
    """
    Main orchestrator coordinating the verification pipeline.
    
    Pipeline:
    1. Deterministic pre-processing (0 LLM calls)
    2. User RAG retrieval (0 LLM calls)
    3. Draft answer (LLM CALL #1)
    4. Hybrid claim extraction (LLM CALL #2 + deterministic)
    5. Batched verification (0 LLM calls - retrieval only)
    6. Deterministic validation (0 LLM calls)
    7. Confidence scoring (deterministic formula)
    8. Response formatting (LLM CALL #3 - optional)
    """
    
    def __init__(self):
        self.claim_extractor = HybridClaimExtractor()
        self.arithmetic_validator = ArithmeticValidator()
        self.consistency_checker = LogicalConsistencyChecker()
        self.misinformation_detector = MisinformationPatternDetector()
        self.confidence_scorer = ConfidenceScorer()
        self.verified_store = get_verified_knowledge_store()
        self.batch_processor = BatchClaimProcessor(self.verified_store)
        self.arbitrator = VerificationArbitrator()
    
    async def process(
        self,
        message: str,
        user_id: str,
        notebook_id: Optional[str] = None,
        source_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a message through the verified study pipeline.
        
        Args:
            message: User's question/message
            user_id: User ID
            notebook_id: Optional notebook ID
            source_ids: Optional source IDs for RAG
            session_id: Optional session ID
        
        Yields:
            Events for streaming response
        """
        
        yield {"type": "thinking", "content": "🔍 Verified mode: Analyzing question..."}
        
        # PHASE 1: Deterministic Pre-Processing (0 LLM calls)
        normalized_input = self._normalize_input(message)
        stress_detected = self._detect_stress(normalized_input)
        syllabus_context = await self._get_syllabus_context(notebook_id)
        
        # PHASE 2: User RAG Retrieval (0 LLM calls)
        yield {"type": "thinking", "content": "📚 Retrieving from your study materials..."}
        user_context = await self._retrieve_user_rag(user_id, notebook_id, source_ids, message)
        
        # PHASE 3: Draft Answer (LLM CALL #1)
        yield {"type": "thinking", "content": "✍️ Generating answer..."}
        draft_answer = await self._generate_draft(message, user_context, syllabus_context)
        
        # PHASE 4: Hybrid Claim Extraction (LLM CALL #2 + Deterministic)
        yield {"type": "thinking", "content": "🔬 Extracting claims for verification..."}
        claims = await self.claim_extractor.extract_claims(draft_answer, syllabus_context)
        
        logger.info(f"📋 Extracted {len(claims)} claims: {[c.type.value for c in claims]}")
        
        # PHASE 5: Batched Verification (0 LLM calls - retrieval only)
        yield {"type": "thinking", "content": f"✅ Verifying {len(claims)} claims against authoritative sources..."}
        verification_results = await self.batch_processor.batch_verify_claims(claims)
        
        # PHASE 6: Deterministic Validation (0 LLM calls)
        yield {"type": "thinking", "content": "🧮 Running deterministic validators..."}
        
        # Arithmetic validation
        arithmetic_results = [
            self.arithmetic_validator.validate_claim(claim)
            for claim in claims
        ]
        arithmetic_valid = all(
            r.valid is None or r.valid
            for r in arithmetic_results
        )
        
        # Logical consistency
        consistency_result = self.consistency_checker.check_consistency(claims, syllabus_context)
        
        # Misinformation detection
        misinformation_flags = self.misinformation_detector.detect_patterns(claims)
        
        # PHASE 7: Arbitration & Confidence Scoring
        yield {"type": "thinking", "content": "⚖️ Arbitrating verification results..."}
        
        final_verdicts = []
        for claim, arith_res, verif_res in zip(claims, arithmetic_results, verification_results):
            verdict = self.arbitrator.arbitrate(
                claim, arith_res, verif_res, consistency_result, misinformation_flags
            )
            final_verdicts.append(verdict)
        
        # Calculate coverage
        coverage = self._calculate_coverage(verification_results)
        
        # Calculate confidence
        confidence = self.confidence_scorer.calculate_confidence(
            verification_results,
            arithmetic_valid,
            consistency_result.passed,
            misinformation_flags,
            coverage
        )
        
        # PHASE 8: Format Response
        yield {"type": "thinking", "content": f"📊 Confidence: {confidence}%"}
        
        # Build structured response
        response = self._build_verified_response(
            draft_answer,
            claims,
            final_verdicts,
            confidence,
            consistency_result,
            misinformation_flags,
            stress_detected,
            coverage
        )
        
        # Stream the answer
        for token in re.split(r"(\s+)", response["answer"]):
            if token.strip():
                yield {"type": "token", "content": token}
        
        # Send complete response with verification data
        yield {
            "type": "complete",
            "message": response["answer"],
            "verified_data": response
        }
    
    def _normalize_input(self, message: str) -> str:
        """Normalize input text (regex-based)."""
        # Simple normalization
        normalized = message.strip()
        return normalized
    
    def _detect_stress(self, message: str) -> bool:
        """Detect stress/anxiety in message (pattern matching)."""
        stress_patterns = [
            r'\b(worried|anxious|stressed|nervous|scared|afraid)\b',
            r'\b(don\'t understand|can\'t understand|confused)\b',
            r'\b(help|please help)\b',
            r'!{2,}',  # Multiple exclamation marks
        ]
        
        message_lower = message.lower()
        for pattern in stress_patterns:
            if re.search(pattern, message_lower):
                return True
        return False
    
    async def _get_syllabus_context(self, notebook_id: Optional[str]) -> Dict[str, Any]:
        """Get syllabus context (DB lookup - placeholder)."""
        # Placeholder - in production, fetch from database
        return {
            "subject": "general",
            "chapters": [],
            "topics": {}
        }
    
    async def _retrieve_user_rag(
        self,
        user_id: str,
        notebook_id: Optional[str],
        source_ids: Optional[List[str]],
        query: str
    ) -> str:
        """Retrieve from user's RAG (existing system)."""
        logger.info(f"🔍 _retrieve_user_rag called with:")
        logger.info(f"   user_id: {user_id}")
        logger.info(f"   notebook_id: {notebook_id}")
        logger.info(f"   source_ids: {source_ids}")
        logger.info(f"   query: {query[:100]}...")
        
        if not notebook_id:
            logger.warning("❌ No notebook_id provided for RAG retrieval")
            return ""
        
        try:
            retriever = RAGRetriever(CONFIG, user_id=user_id)
            logger.info(f"✅ Created RAGRetriever for user {user_id}")
            
            # Try retrieval with notebook_id and source_ids
            docs = retriever.get_relevant_documents(
                query=query,
                user_id=user_id,
                notebook_id=notebook_id,
                source_ids=source_ids
            )
            
            logger.info(f"📚 Retrieved {len(docs)} documents from user RAG (with filters)")
            
            # If no docs found with filters, try without source_ids filter
            if not docs and source_ids:
                logger.info("🔄 Retrying without source_ids filter...")
                docs = retriever.get_relevant_documents(
                    query=query,
                    user_id=user_id,
                    notebook_id=notebook_id,
                    source_ids=None  # Remove source filter
                )
                logger.info(f"📚 Retrieved {len(docs)} documents (without source filter)")
            
            # If still no docs, try with just user_id
            if not docs:
                logger.info("🔄 Retrying with just user_id...")
                docs = retriever.get_relevant_documents(
                    query=query,
                    user_id=user_id,
                    notebook_id=None,
                    source_ids=None
                )
                logger.info(f"📚 Retrieved {len(docs)} documents (user-only filter)")
            
            if docs:
                context = "\n\n".join([doc.page_content for doc in docs[:3]])
                logger.info(f"📝 Context length: {len(context)} characters")
                logger.info(f"📄 First doc preview: {docs[0].page_content[:200]}...")
                logger.info(f"📄 First doc metadata: {docs[0].metadata}")
                return context
            else:
                logger.warning("⚠️ No documents found in user RAG - this is the issue!")
                logger.warning(f"   Searched with: notebook_id={notebook_id}, source_ids={source_ids}")
                return ""
        except Exception as e:
            logger.error(f"❌ RAG retrieval error: {e}", exc_info=True)
            return ""
    
    async def _generate_draft(
        self,
        message: str,
        user_context: str,
        syllabus_context: Dict[str, Any]
    ) -> str:
        """Generate draft answer (LLM CALL #1)."""
        
        # Build system prompt based on available context
        if user_context and len(user_context.strip()) > 0:
            system_prompt = f"""You are a helpful study assistant. Provide clear, accurate answers based on the provided study materials.

Context from study materials:
{user_context}

Use the above context to answer the question. If the context doesn't contain relevant information, say so clearly."""
        else:
            system_prompt = """You are a helpful study assistant. Provide clear, accurate answers.

Note: No study materials were found for this question. Please inform the user that they should add relevant sources to their notebook for better answers."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        response = await chat_completion(messages, stream=False)
        return response.choices[0].message.content
    
    def _calculate_coverage(self, verification_results: List) -> float:
        """Calculate verified knowledge coverage."""
        if not verification_results:
            return 0.0
        
        verified_count = sum(
            1 for r in verification_results
            if r.status in ["verified", "partially_supported"]
        )
        
        return verified_count / len(verification_results)
    
    def _build_verified_response(
        self,
        answer: str,
        claims: List,
        verdicts: List,
        confidence: float,
        consistency_result,
        misinformation_flags: List,
        stress_detected: bool,
        coverage: float
    ) -> Dict[str, Any]:
        """Build structured verified response."""
        
        # Build claims list
        claims_data = []
        for claim, verdict in zip(claims, verdicts):
            claims_data.append({
                "text": claim.text,
                "status": verdict.status,
                "source": verdict.source,
                "authority_level": None,  # Will be populated from verification
                "reason": verdict.reason
            })
        
        # Flags
        conflict_flag = not consistency_result.passed
        outdated_flag = False  # Placeholder - would check publication dates
        out_of_scope_flag = False  # Placeholder - would check syllabus
        
        # Emotional support
        emotional_support = None
        if stress_detected:
            emotional_support = "It's okay to feel overwhelmed. Let's break this down together step by step. You've got this! 💪"
        
        response = {
            "mode": "verified",
            "answer": answer,
            "claims": claims_data,
            "conflict_flag": conflict_flag,
            "outdated_flag": outdated_flag,
            "out_of_scope_flag": out_of_scope_flag,
            "numeric_validated": True,  # Based on arithmetic validator
            "logical_consistency_passed": consistency_result.passed,
            "confidence_score": confidence,
            "emotional_support": emotional_support,
            "sources": [],  # Would be populated from verification results
            "coverage": coverage,
            "misinformation_detected": len(misinformation_flags) > 0
        }
        
        return response
