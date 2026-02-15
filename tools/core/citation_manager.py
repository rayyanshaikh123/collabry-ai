"""
Citation Manager - Tracks and manages source citations for AI responses.

This module ensures that AI-generated responses include proper citations
to source materials, improving transparency and reducing hallucinations.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class CitationManager:
    """
    Manages citations for a single AI response.
    
    Each citation links a piece of information in the response back to
    a specific source document or excerpt.
    """
    
    def __init__(self):
        """Initialize empty citation list."""
        self.citations: List[Dict[str, Any]] = []
        self._citation_map: Dict[str, int] = {}  # Maps content hash to citation ID
    
    def add_citation(
        self,
        source_id: str,
        source_name: str,
        excerpt: str,
        page: Optional[int] = None,
        source_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a citation and return its ID.
        
        Args:
            source_id: Unique identifier for the source document
            source_name: Human-readable name of the source
            excerpt: Text excerpt being cited (max 300 chars)
            page: Page number if applicable
            source_type: Type of source (pdf, text, audio, etc.)
            metadata: Additional metadata about the source
        
        Returns:
            Citation ID (1-indexed) to use in the response like [1], [2], etc.
        """
        # Truncate excerpt to reasonable length
        excerpt = excerpt[:300] if excerpt else ""
        
        # Check if we already have this exact citation
        content_key = f"{source_id}:{excerpt[:100]}"
        if content_key in self._citation_map:
            return self._citation_map[content_key]
        
        # Create new citation
        citation_id = len(self.citations) + 1
        
        citation = {
            "id": citation_id,
            "source_id": source_id,
            "source_name": source_name,
            "excerpt": excerpt,
            "page": page,
            "type": source_type,
            "metadata": metadata or {}
        }
        
        self.citations.append(citation)
        self._citation_map[content_key] = citation_id
        
        logger.debug(f"Added citation [{citation_id}] for source: {source_name}")
        return citation_id
    
    def add_from_document(self, doc: Any) -> int:
        """
        Add a citation from a LangChain Document object.
        
        Args:
            doc: LangChain Document with metadata
        
        Returns:
            Citation ID
        """
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        
        return self.add_citation(
            source_id=metadata.get("source_id", "unknown"),
            source_name=metadata.get("source", "Unknown Source"),
            excerpt=content,
            page=metadata.get("page"),
            source_type=metadata.get("type"),
            metadata=metadata
        )
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all citations.
        
        Returns:
            List of citation dictionaries
        """
        return self.citations
    
    def get_by_id(self, citation_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific citation by ID.
        
        Args:
            citation_id: Citation ID (1-indexed)
        
        Returns:
            Citation dictionary or None if not found
        """
        if 1 <= citation_id <= len(self.citations):
            return self.citations[citation_id - 1]
        return None
    
    def has_citations(self) -> bool:
        """Check if any citations have been added."""
        return len(self.citations) > 0
    
    def count(self) -> int:
        """Get total number of citations."""
        return len(self.citations)
    
    def format_for_prompt(self, docs: List[Any]) -> tuple[str, List[int]]:
        """
        Format documents with inline citations for LLM prompt.
        
        Args:
            docs: List of LangChain Documents
        
        Returns:
            Tuple of (formatted_context, list_of_citation_ids)
        """
        context_parts = []
        citation_ids = []
        
        for doc in docs:
            citation_id = self.add_from_document(doc)
            citation_ids.append(citation_id)
            
            # Format: content [citation_id]
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            context_parts.append(f"{content} [{citation_id}]")
        
        context = "\n\n".join(context_parts)
        return context, citation_ids
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export citations as a dictionary for API responses.
        
        Returns:
            Dictionary with citations list
        """
        return {
            "citations": self.citations,
            "count": len(self.citations)
        }


def extract_citation_numbers(text: str) -> List[int]:
    """
    Extract citation numbers from text like [1], [2], etc.
    
    Args:
        text: Text containing citations
    
    Returns:
        List of citation numbers found
    """
    import re
    matches = re.findall(r'\[(\d+)\]', text)
    return [int(m) for m in matches]


def validate_citations(text: str, citation_manager: CitationManager) -> bool:
    """
    Validate that all citations in text exist in the citation manager.
    
    Args:
        text: Text containing citations
        citation_manager: CitationManager instance
    
    Returns:
        True if all citations are valid, False otherwise
    """
    cited_ids = extract_citation_numbers(text)
    max_id = citation_manager.count()
    
    for cid in cited_ids:
        if cid < 1 or cid > max_id:
            logger.warning(f"Invalid citation [{cid}] - only {max_id} citations exist")
            return False
    
    return True
