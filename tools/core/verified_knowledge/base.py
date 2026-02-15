"""
Verified Knowledge Base - Separate FAISS index for authoritative educational content.

This module manages a completely separate vector store from the user RAG system,
ensuring no data mixing and maintaining strict authority levels for verification.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import pickle

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from core.embeddings import get_embedding_client
from config import Config

logger = logging.getLogger(__name__)


class VerifiedKnowledgeStore:
    """
    Manages the verified knowledge base with authoritative educational content.
    
    Metadata Schema:
    - authority_level: "high" | "medium" | "low"
    - source_type: "ncert" | "cbse" | "icse" | "textbook" | "research_paper" | "government"
    - publication_year: int
    - syllabus_version: str (e.g., "cbse_2023", "icse_2024")
    - exam_type: str (e.g., "class_10", "class_12", "jee", "neet")
    - subject: str
    - chapter: str (optional)
    - topic: str (optional)
    """
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize verified knowledge store.
        
        Args:
            index_path: Path to FAISS index (defaults to config)
        """
        self.index_path = index_path or Config.VERIFIED_FAISS_INDEX_PATH
        self.embeddings = get_embedding_client()
        self.vectorstore: Optional[FAISS] = None
        
        # Ensure directory exists
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing index or create new
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create a new empty one."""
        index_file = os.path.join(self.index_path, "index.faiss")
        
        if os.path.exists(index_file):
            try:
                self.vectorstore = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✅ Loaded verified knowledge index from {self.index_path}")
            except Exception as e:
                logger.warning(f"Failed to load verified index: {e}. Creating new index.")
                self._create_empty_index()
        else:
            logger.info("No existing verified knowledge index found. Creating new index.")
            self._create_empty_index()
    
    def _create_empty_index(self):
        """Create an empty FAISS index."""
        # Create with a dummy document to initialize
        dummy_doc = Document(
            page_content="Initialization document",
            metadata={
                "authority_level": "high",
                "source_type": "system",
                "is_dummy": True
            }
        )
        self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
        self._save_index()
        logger.info(f"✅ Created new verified knowledge index at {self.index_path}")
    
    def _save_index(self):
        """Save FAISS index to disk."""
        try:
            self.vectorstore.save_local(self.index_path)
            logger.info(f"💾 Saved verified knowledge index to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save verified index: {e}")
    
    def add_documents(
        self,
        documents: List[Document],
        authority_level: str = "medium",
        source_type: str = "textbook",
        **metadata_kwargs
    ) -> bool:
        """
        Add verified documents to the knowledge base.
        
        Args:
            documents: List of Document objects
            authority_level: "high" | "medium" | "low"
            source_type: Type of source
            **metadata_kwargs: Additional metadata fields
        
        Returns:
            bool: Success status
        """
        if not documents:
            logger.warning("No documents provided to add")
            return False
        
        # Validate authority level
        if authority_level not in ["high", "medium", "low"]:
            logger.error(f"Invalid authority_level: {authority_level}")
            return False
        
        # Enrich documents with metadata
        enriched_docs = []
        for doc in documents:
            # Merge metadata
            doc.metadata.update({
                "authority_level": authority_level,
                "source_type": source_type,
                **metadata_kwargs
            })
            enriched_docs.append(doc)
        
        try:
            # Add to vectorstore
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(enriched_docs, self.embeddings)
            else:
                # Remove dummy document if it exists
                self._remove_dummy_documents()
                self.vectorstore.add_documents(enriched_docs)
            
            self._save_index()
            logger.info(f"✅ Added {len(enriched_docs)} documents to verified knowledge base")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add documents to verified knowledge: {e}")
            return False
    
    def _remove_dummy_documents(self):
        """Remove initialization dummy documents."""
        # FAISS doesn't support direct deletion, so we'll filter during retrieval
        pass
    
    async def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[List[Document]]:
        """
        Batch retrieve documents for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of documents per query
            filter_metadata: Optional metadata filters
        
        Returns:
            List of document lists (one per query)
        """
        if not self.vectorstore:
            logger.warning("Vectorstore not initialized")
            return [[] for _ in queries]
        
        results = []
        
        for query in queries:
            try:
                # Retrieve documents
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    query,
                    k=top_k
                )
                
                # Filter out dummy documents
                filtered_docs = []
                for doc, score in docs_with_scores:
                    if doc.metadata.get("is_dummy"):
                        continue
                    
                    # Apply metadata filters if provided
                    if filter_metadata:
                        if all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                            # Add similarity score to metadata
                            doc.metadata["similarity"] = float(1 - score)  # Convert distance to similarity
                            filtered_docs.append(doc)
                    else:
                        doc.metadata["similarity"] = float(1 - score)
                        filtered_docs.append(doc)
                
                results.append(filtered_docs)
            
            except Exception as e:
                logger.error(f"Failed to retrieve for query '{query}': {e}")
                results.append([])
        
        return results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve documents for a single query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filters
        
        Returns:
            List of documents
        """
        import asyncio
        results = asyncio.run(self.batch_retrieve([query], top_k, filter_metadata))
        return results[0] if results else []
    
    def get_coverage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about verified knowledge coverage.
        
        Returns:
            Dict with coverage statistics
        """
        if not self.vectorstore:
            return {
                "total_documents": 0,
                "coverage": 0.0,
                "by_authority": {},
                "by_source_type": {}
            }
        
        # FAISS doesn't provide easy access to all documents
        # This is a placeholder - in production, maintain a separate metadata store
        return {
            "total_documents": "unknown",
            "coverage": 0.0,
            "message": "Coverage tracking requires separate metadata store"
        }


# Global instance
_verified_store: Optional[VerifiedKnowledgeStore] = None


def get_verified_knowledge_store() -> VerifiedKnowledgeStore:
    """Get or create global verified knowledge store instance."""
    global _verified_store
    if _verified_store is None:
        _verified_store = VerifiedKnowledgeStore()
    return _verified_store
