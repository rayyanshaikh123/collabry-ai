"""
Search Sources Tool - RAG retrieval from user's documents.

This tool searches the user's uploaded documents and notes
for relevant information to answer their questions.
"""

from typing import Optional, List
from langchain_core.tools import tool
from rag.retriever import get_retriever
from langchain_core.documents import Document


def format_sources(docs: list[Document]) -> str:
    """Format retrieved documents by grouping chunks by their original source."""
    if not docs:
        return "I couldn't find any relevant information in your documents."
    
    # Group by source filename
    by_source = {}
    for doc in docs:
        source = doc.metadata.get("source") or doc.metadata.get("filename") or "Unknown Document"
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(doc.page_content.strip())
    
    formatted = "I found some relevant information in your documents:\n\n"
    for i, (source, chunks) in enumerate(by_source.items(), 1):
        formatted += f"### Document {i}: {source}\n"
        for chunk in chunks:
            formatted += f"- {chunk}\n\n"
    
    return formatted.strip()


@tool
def search_sources(
    query: str,
    user_id: str = "default",
    notebook_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
) -> str:
    """
    Search user's uploaded documents and notes for relevant information.
    
    Use this tool when the user asks about their notes, study materials,
    uploaded PDFs, or previously studied content.
    
    Examples of when to use:
    - "What did I learn about photosynthesis?"
    - "Find information about the French Revolution in my notes"
    - "What does my biology textbook say about DNA?"
    
    Args:
        query: The search query
        user_id: User identifier (optional, auto-injected)
        notebook_id: Optional notebook to search within (optional, auto-injected)
    
    Returns:
        Relevant text excerpts with source citations
    
    Example:
        search_sources(query="What is photosynthesis?")
    """
    try:
        # Get retriever with user and notebook filtering
        retriever = get_retriever(
            user_id=user_id,
            notebook_id=notebook_id,
            source_ids=source_ids,
            k=4
        )
        
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        
        # Format and return
        return format_sources(docs)
    
    except Exception as e:
        return f"Error searching documents: {str(e)}"
