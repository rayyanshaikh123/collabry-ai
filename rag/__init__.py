"""
RAG (Retrieval-Augmented Generation) Package.

Handles document ingestion, vector storage, and retrieval.
"""

from .vectorstore import get_vectorstore, VectorStoreConfig
from .retriever import get_retriever
from .ingest import ingest_document, chunk_text

__all__ = [
    'get_vectorstore',
    'VectorStoreConfig',
    'get_retriever',
    'ingest_document',
    'chunk_text'
]
