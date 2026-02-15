"""
Verified Knowledge Base Package

Separate FAISS index for authoritative educational content.
"""

from .base import VerifiedKnowledgeStore, get_verified_knowledge_store
from .ingestion_service import VerifiedKnowledgeIngestionService, IngestionResult

__all__ = [
    'VerifiedKnowledgeStore',
    'get_verified_knowledge_store',
    'VerifiedKnowledgeIngestionService',
    'IngestionResult'
]
