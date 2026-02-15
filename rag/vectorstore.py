"""
Vector Store Client - Provider-agnostic vector database.

Supports multiple vector databases:
- FAISS (local, simple)
- Chroma (local/cloud, open source)
- Pinecone (cloud, production)
- Qdrant (local/cloud, open source)

All vector stores support metadata filtering by user_id and notebook_id.
"""

import os
import time
from typing import Optional, List, Dict, Any
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from core.embeddings import get_embedding_client
import logging

logger = logging.getLogger(__name__)


class VectorStoreConfig:
    """Configuration for vector store."""
    
    def __init__(self):
        self.provider = os.getenv("VECTOR_STORE", "faiss").lower()
        
        # FAISS config
        self.faiss_index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
        
        # Chroma config
        self.chroma_host = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        self.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        self.chroma_collection = os.getenv("CHROMA_COLLECTION", "documents")
        
        # Pinecone config
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-west-2")
        self.pinecone_index = os.getenv("PINECONE_INDEX", "study-assistant")
        
        # Qdrant config
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "documents")
    
    def __repr__(self):
        return f"VectorStoreConfig(provider={self.provider})"


def _create_faiss_store() -> VectorStore:
    """Create or load FAISS vector store."""
    from langchain_community.vectorstores import FAISS
    import os
    
    config = VectorStoreConfig()
    embeddings = get_embedding_client()
    
    # Check if index exists
    if os.path.exists(os.path.join(config.faiss_index_path, "index.faiss")):
        # Load existing index
        return FAISS.load_local(
            config.faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        # Create new empty index
        # Initialize with a dummy document
        dummy_doc = Document(
            page_content="Initialization document",
            metadata={"user_id": "system", "notebook_id": "system"}
        )
        store = FAISS.from_documents([dummy_doc], embeddings)
        
        # Save to disk
        os.makedirs(config.faiss_index_path, exist_ok=True)
        store.save_local(config.faiss_index_path)
        
        return store


def _create_chroma_store() -> VectorStore:
    """Create or load Chroma vector store."""
    from langchain_community.vectorstores import Chroma
    
    config = VectorStoreConfig()
    embeddings = get_embedding_client()
    
    return Chroma(
        collection_name=config.chroma_collection,
        embedding_function=embeddings,
        persist_directory=config.chroma_persist_dir
    )


def _create_pinecone_store() -> VectorStore:
    """Create Pinecone vector store."""
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    
    config = VectorStoreConfig()
    embeddings = get_embedding_client()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=config.pinecone_api_key)
    
    # Check if index exists, create if not
    if config.pinecone_index not in pc.list_indexes().names():
        pc.create_index(
            name=config.pinecone_index,
            dimension=1536,  # Adjust based on embedding model
            metric="cosine"
        )
    
    return PineconeVectorStore(
        index_name=config.pinecone_index,
        embedding=embeddings
    )


def _create_qdrant_store() -> VectorStore:
    """Create Qdrant vector store."""
    from langchain_community.vectorstores import Qdrant
    from qdrant_client import QdrantClient
    
    config = VectorStoreConfig()
    embeddings = get_embedding_client()
    
    client = QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key
    )
    
    return Qdrant(
        client=client,
        collection_name=config.qdrant_collection,
        embeddings=embeddings
    )


# Singleton instance
_vectorstore: Optional[VectorStore] = None


def get_vectorstore() -> VectorStore:
    """
    Get vector store client based on configuration.
    
    Returns:
        LangChain VectorStore instance
        
    Environment Variables:
        VECTOR_STORE: faiss | chroma | pinecone | qdrant
        
    Examples:
        # FAISS (local, simple)
        VECTOR_STORE=faiss
        FAISS_INDEX_PATH=./data/faiss_index
        
        # Chroma (local)
        VECTOR_STORE=chroma
        CHROMA_PERSIST_DIR=./data/chroma_db
        
        # Pinecone (cloud)
        VECTOR_STORE=pinecone
        PINECONE_API_KEY=...
        PINECONE_INDEX=study-assistant
        
        # Qdrant (cloud)
        VECTOR_STORE=qdrant
        QDRANT_URL=https://...
        QDRANT_API_KEY=...
    """
    global _vectorstore
    
    if _vectorstore is None:
        config = VectorStoreConfig()
        
        if config.provider == "faiss":
            _vectorstore = _create_faiss_store()
        
        elif config.provider == "chroma":
            _vectorstore = _create_chroma_store()
        
        elif config.provider == "pinecone":
            _vectorstore = _create_pinecone_store()
        
        elif config.provider == "qdrant":
            _vectorstore = _create_qdrant_store()
        
        else:
            raise ValueError(
                f"Unknown vector store provider: {config.provider}. "
                f"Supported: faiss, chroma, pinecone, qdrant"
            )
    
    return _vectorstore


def add_documents(
    documents: List[Document],
    user_id: str,
    notebook_id: str
) -> List[str]:
    """
    Add documents to vector store with user and notebook metadata.
    SECURITY FIX - Phase 3: Enhanced with input validation and sanitization.
    
    Args:
        documents: List of LangChain Document objects
        user_id: User identifier
        notebook_id: Notebook identifier
    
    Returns:
        List of document IDs
    """
    from core.validator import sanitize_user_input
    import re
    
    # SECURITY FIX - Validate inputs
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("user_id is required and must be a non-empty string")
    if not isinstance(notebook_id, str) or not notebook_id:
        raise ValueError("notebook_id is required and must be a non-empty string")
    if not documents:
        raise ValueError("documents list cannot be empty")
    
    # Validate user_id and notebook_id format (prevent injection)
    if not re.match(r'^[A-Za-z0-9_-]{1,50}$', user_id):
        raise ValueError(f"Invalid user_id format: {user_id}")
    if not re.match(r'^[A-Za-z0-9_-]{1,50}$', notebook_id):
        raise ValueError(f"Invalid notebook_id format: {notebook_id}")
    
    vectorstore = get_vectorstore()
    
    # SECURITY FIX - Sanitize and validate documents
    sanitized_documents = []
    for i, doc in enumerate(documents[:1000]):  # Limit to prevent DoS
        if not isinstance(doc, Document):
            logger.warning(f"🚨 Skipping non-Document object at index {i}")
            continue
            
        # Sanitize page content
        sanitized_content = sanitize_user_input(doc.page_content)
        if len(sanitized_content.strip()) < 10:  # Skip very short content
            logger.warning(f"🚨 Skipping document with insufficient content at index {i}")
            continue
        
        # Create sanitized document with security metadata
        sanitized_doc = Document(
            page_content=sanitized_content[:50000],  # Limit size to prevent DoS
            metadata={
                **doc.metadata,
                "user_id": user_id,
                "notebook_id": notebook_id,
                "added_at": str(int(time.time())),  # Timestamp for audit
                "content_length": len(sanitized_content)
            }
        )
        
        # Sanitize metadata keys and values
        clean_metadata = {}
        for key, value in sanitized_doc.metadata.items():
            # Validate metadata key
            if not isinstance(key, str) or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]{0,49}$', key):
                logger.warning(f"🚨 Skipping invalid metadata key: {key}")
                continue
                
            # Sanitize metadata value
            if isinstance(value, str):
                clean_value = sanitize_user_input(value)[:500]  # Limit metadata size
            elif isinstance(value, (int, float, bool)):
                clean_value = value
            else:
                clean_value = str(value)[:500]  # Convert to string and limit
                
            clean_metadata[key] = clean_value
        
        sanitized_doc.metadata = clean_metadata
        sanitized_documents.append(sanitized_doc)
    
    if not sanitized_documents:
        raise ValueError("No valid documents after sanitization")
    
    logger.info(f"✅ Adding {len(sanitized_documents)} sanitized documents for user {user_id}")
    
    try:
        # Add to vector store
        ids = vectorstore.add_documents(sanitized_documents)
        
        # Save FAISS index if using FAISS
        config = VectorStoreConfig()
        if config.provider == "faiss":
            vectorstore.save_local(config.faiss_index_path)
        
        return ids
        
    except Exception as e:
        logger.error(f"🚨 Error adding documents: {e}")
        raise ValueError(f"Failed to add documents: {str(e)}")


def similarity_search(
    query: str,
    user_id: str,
    notebook_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
    k: int = 4
) -> List[Document]:
    """
    Search for similar documents with metadata filtering.
    SECURITY FIX - Phase 3: Enhanced with source boundary validation.
    
    Args:
        query: Search query
        user_id: User identifier (required for isolation)
        notebook_id: Optional notebook filter
        source_ids: Optional list of source IDs to filter by
        k: Number of results to return
    
    Returns:
        List of relevant documents
    """
    from core.validator import validate_source_boundaries, sanitize_user_input
    
    # SECURITY FIX - Phase 3: Validate and sanitize inputs
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("user_id is required and must be a non-empty string")
    
    # Sanitize query to prevent injection attacks
    query = sanitize_user_input(query)
    if not query or len(query.strip()) < 3:
        logger.warning("🚨 Query too short or empty after sanitization")
        return []
    
    # Validate source boundaries if source_ids provided
    if source_ids:
        if not validate_source_boundaries(user_id, notebook_id, source_ids, user_id):
            logger.error(f"🚨 Source boundary validation failed for user {user_id}")
            return []
    
    # Limit result count to prevent DoS
    k = min(k, 100)
    
    vectorstore = get_vectorstore()
    
    # Build metadata filter with strict user isolation
    filter_dict = {"user_id": str(user_id)}
    if notebook_id:
        filter_dict["notebook_id"] = str(notebook_id)
    
    logger.debug(f"🔍 RAG Search: query='{query[:50]}...', filter={filter_dict}, source_ids={source_ids}")
    
    # Search with filter
    # Note: FAISS doesn't support native filtering, so we'll post-filter
    config = VectorStoreConfig()
    
    if config.provider == "faiss":
        # FAISS: search more results and filter in memory.
        # When filtering by source_ids, over-retrieve significantly.
        search_k = k * (100 if source_ids else 5)
        results = vectorstore.similarity_search(query, k=search_k)
        
        # Post-filter by metadata
        filtered = []
        for i, doc in enumerate(results):
            doc_user_id = str(doc.metadata.get("user_id"))
            doc_notebook_id = str(doc.metadata.get("notebook_id"))
            
            # Diagnostic: Only log for individual chunks if there's a problem later or in debug mode
            # logger.debug(f"Chunk {i}: user='{doc_user_id}', notebook='{doc_notebook_id}'")
            
            if doc_user_id != str(user_id):
                continue
            if notebook_id and doc_notebook_id != str(notebook_id):
                continue
            
            # Source filtering: Match either source_id or filename (source)
            if source_ids:
                # Harden: Coerce both filter and metadata to strings to avoid type mismatch (int vs str)
                str_source_ids = [str(sid) for sid in source_ids]
                doc_source_id = str(doc.metadata.get("source_id")) if doc.metadata.get("source_id") is not None else None
                doc_filename = str(doc.metadata.get("source")) if doc.metadata.get("source") is not None else None
                
                if doc_source_id not in str_source_ids and doc_filename not in str_source_ids:
                    continue
            
            filtered.append(doc)
            if len(filtered) >= k:
                break
        
        if not filtered and results:
            # Critical Diagnostic: Why did everything fail filtering?
            sample_metadata = results[0].metadata if results else {}
            logger.warning(
                f"⚠️ RAG Search: 0/{len(results)} chunks matched filters!\n"
                f"  Target user_id:     '{user_id}' (type={type(user_id).__name__})\n"
                f"  Target notebook_id: '{notebook_id}' (type={type(notebook_id).__name__})\n"
                f"  Target source_ids:  {source_ids}\n"
                f"  Sample metadata:    {sample_metadata}"
            )
            
            # FALLBACK: If we filtered by notebook/sources and got NOTHING, but we HAVE results for this user,
            # try returning results for this user that MATCH the source_ids (if provided).
            # We must NEVER return unselected sources if source_ids is set.
            logger.info(f"🔄 RAG Fallback: Checking user-wide matches for {user_id}...")
            fallback_filtered = []
            for doc in results:
                doc_user_id = str(doc.metadata.get("user_id"))
                doc_source_id = str(doc.metadata.get("source_id"))
                
                # Check user ownership always
                if doc_user_id != str(user_id):
                    continue
                    
                # If source_ids is provided, STRICTLY filter by it
                if source_ids and doc_source_id not in [str(sid) for sid in source_ids]:
                    continue
                    
                fallback_filtered.append(doc)
                if len(fallback_filtered) >= k:
                    break
            
            if fallback_filtered:
                logger.info(f"✅ RAG Fallback: Found {len(fallback_filtered)} matches across user documents.")
                return fallback_filtered
            
        logger.debug(f"✅ RAG Search Result: Found {len(filtered)} matches (after post-filtering {len(results)} chunks)")
        return filtered
    
    else:
        # Other stores support native filtering for user/notebook isolation.
        search_k = k * 50 if source_ids else k
        results = vectorstore.similarity_search(query, k=search_k, filter=filter_dict)
        if not source_ids:
            return results[:k]
        
        # Standardize source filtering across providers by post-filtering the search results
        filtered = []
        for doc in results:
            doc_source_id = doc.metadata.get("source_id")
            doc_filename = doc.metadata.get("source")
            if doc_source_id in source_ids or doc_filename in source_ids:
                filtered.append(doc)
            if len(filtered) >= k:
                break
        
        logger.debug(f"✅ RAG Search Result: Found {len(filtered)} matches")
        return filtered


def reset_vectorstore():
    """Reset vector store singleton. Useful for testing."""
    global _vectorstore
    _vectorstore = None
