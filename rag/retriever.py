"""
RAG Retriever - LangChain retriever with metadata filtering.

Creates retrievers filtered by user_id and notebook_id for secure multi-tenant RAG.
"""

from typing import Optional, List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rag.vectorstore import get_vectorstore, similarity_search


class FilteredRetriever(BaseRetriever):
    """Retriever with user and notebook filtering."""
    
    user_id: str
    notebook_id: Optional[str] = None
    source_ids: Optional[List[str]] = None
    k: int = 4
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> list[Document]:
        """Get documents relevant to query with filtering."""
        return similarity_search(
            query=query,
            user_id=self.user_id,
            notebook_id=self.notebook_id,
            source_ids=self.source_ids,
            k=self.k
        )


def get_retriever(
    user_id: str,
    notebook_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
    k: int = 4
) -> BaseRetriever:
    """
    Get retriever filtered by user and notebook.
    
    Args:
        user_id: User identifier (required for security)
        notebook_id: Optional notebook filter
        k: Number of results to return
    
    Returns:
        LangChain retriever instance
        
    Example:
        >>> retriever = get_retriever(user_id="user123", notebook_id="bio101")
        >>> docs = retriever.get_relevant_documents("photosynthesis")
    """
    return FilteredRetriever(
        user_id=user_id,
        notebook_id=notebook_id,
        source_ids=source_ids,
        k=k
    )


async def get_notebook_context(notebook_id: str, user_id: Optional[str] = None, k: int = 5) -> str:
    """
    Load notebook content for voice tutor context.
    
    Args:
        notebook_id: The notebook ID to load
        user_id: Optional user filter
        k: Number of chunks to retrieve
        
    Returns:
        Combined context string from notebook chunks
    """
    try:
        # Get relevant chunks from the notebook
        docs = similarity_search(
            query="Study content and key concepts",  # Generic query to get overview
            user_id=user_id or "system",
            notebook_id=notebook_id,
            k=k
        )
        
        if not docs:
            return ""
        
        # Combine document contents with metadata
        context_parts = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            chunk_text = doc.page_content
            
            # Add section header if available
            if 'section' in metadata:
                context_parts.append(f"## {metadata['section']}")
            elif 'page' in metadata:
                context_parts.append(f"## Page {metadata['page']}")
            
            context_parts.append(chunk_text)
            context_parts.append("")  # Empty line between chunks
        
        return "\n".join(context_parts)
        
    except Exception as e:
        # Log error but don't fail the whole session
        import logging
        logging.getLogger(__name__).error(f"Error loading notebook context: {e}")
        return ""
