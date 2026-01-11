# core/rag_retriever.py
"""
RAG Pipeline with multi-user isolation using FAISS metadata filtering.

- User-isolated document retrieval via metadata['user_id']
- Each user's documents are tagged and filtered
- No cross-user document leakage
- Supports per-user document ingestion

Flow:
  1. Documents ingested with user_id in metadata
  2. Retrieval filters by user_id to ensure isolation
  3. Shared documents can have user_id="public"
"""
import os
from pathlib import Path
import logging
from typing import List, Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from core.embeddings import EmbeddingModel
from langchain_core.embeddings import Embeddings


class HuggingFaceCloudEmbeddings(Embeddings):
    """LangChain-compatible wrapper for Hugging Face Inference API embeddings."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self._model = EmbeddingModel(model_name=model_name, api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents."""
        return self._model.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        result = self._model.embed([text])
        return result[0] if result else []

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed documents."""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed query."""
        return self.embed_query(text)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import shutil
import tarfile
import io
from core.mongo_store import MongoMemoryStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global shared vector store (singleton pattern to prevent stale index issues)
_shared_vector_store = None
_shared_embeddings = None
_shared_config = None

class RAGRetriever:
    def __init__(self, config, user_id: Optional[str] = None):
        """
        Initialize RAG retriever with optional user isolation.
        Uses shared global vector store to prevent stale index issues.
        
        Args:
            config: Configuration dictionary
            user_id: User identifier for filtering (None = shared/public only)
        """
        global _shared_vector_store, _shared_embeddings, _shared_config
        
        self.config = config
        self.user_id = user_id
        self.documents_path = Path(config.get("documents_path", "documents"))
        self.faiss_index_path = config["faiss_index_path"]

        if not self.documents_path.exists():
            self.documents_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created documents directory at: {self.documents_path}")

        # Use shared embeddings and vector store
        if _shared_embeddings is None:
            _shared_embeddings = HuggingFaceCloudEmbeddings(
                model_name=config["embedding_model"],
                api_key=config.get("huggingface_api_key")
            )
            _shared_config = config
        
        self.embeddings = _shared_embeddings
        self.vector_store = _shared_vector_store
        
        # Load or create vector store if not already loaded
        if _shared_vector_store is None:
            self._load_or_create_vector_store()
            _shared_vector_store = self.vector_store
        else:
            self.vector_store = _shared_vector_store
        
        if user_id:
            logger.info(f"✓ RAG retriever initialized with user isolation: user_id={user_id}")

    def _load_or_create_vector_store(self):
        """Load FAISS index from disk if it exists, otherwise create it."""
        # Ensure directory for index path parent exists
        faiss_dir = str(self.faiss_index_path)

        # Try to restore FAISS index from local disk first
        if os.path.exists(faiss_dir):
            try:
                self.vector_store = FAISS.load_local(
                    self.faiss_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded FAISS index from {self.faiss_index_path}")
                return
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}. Rebuilding...")

        # If local index missing, attempt to fetch from MongoDB GridFS
        try:
            mongo = MongoMemoryStore(self.config.get('mongo_uri'), self.config.get('mongo_db'), collection_name=self.config.get('memory_collection', 'conversations'))
            archive_name = Path(self.faiss_index_path).name + ".tar.gz"
            data = mongo.load_faiss_index_archive(archive_name)
            if data:
                logger.info(f"Found FAISS archive in GridFS: {archive_name}, restoring to {faiss_dir}")
                # ensure parent removed then extracted
                if os.path.exists(faiss_dir):
                    shutil.rmtree(faiss_dir)
                os.makedirs(faiss_dir, exist_ok=True)
                with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tf:
                    tf.extractall(path=str(Path(faiss_dir).parent))
                try:
                    self.vector_store = FAISS.load_local(
                        self.faiss_index_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"Restored FAISS index from GridFS to {self.faiss_index_path}")
                    return
                except Exception as e:
                    logger.error(f"Failed to load restored FAISS index: {e}. Rebuilding...")
        except Exception:
            # Any failure to use Mongo should not block index creation
            logger.debug("GridFS restore attempt failed or not available")

        logger.info("Creating new FAISS index...")
        # Load documents
        loader = DirectoryLoader(
            str(self.documents_path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()

        if not documents:
            # No documents on disk — create an empty placeholder FAISS index so
            # the vector store is initialized and subsequent background
            # ingestion can add documents without causing errors.
            logger.info("No documents found on disk — creating empty FAISS index placeholder.")
            try:
                # Create a placeholder index with a single system document
                placeholder_text = ["placeholder"]
                placeholder_meta = [{"user_id": "system", "placeholder": True}]
                self.vector_store = FAISS.from_texts(
                    texts=placeholder_text,
                    embedding=self.embeddings,
                    metadatas=placeholder_meta,
                )
                # Persist locally so subsequent restarts have an index directory
                self.vector_store.save_local(self.faiss_index_path)
                logger.info(f"Created placeholder FAISS index at {self.faiss_index_path}")
            except Exception as e:
                logger.error(f"Failed to create placeholder FAISS index: {e}")
            return

        # Add default metadata (public documents accessible to all)
        for doc in documents:
            if "user_id" not in doc.metadata:
                doc.metadata["user_id"] = "public"

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        # Persist locally
        self.vector_store.save_local(self.faiss_index_path)
        logger.info(f"FAISS index created and saved to {self.faiss_index_path}")

        # Archive and store in Mongo GridFS (best-effort)
        try:
            archive_base = str(Path(self.faiss_index_path).name)
            tmp_archive = str(Path(self.faiss_index_path).with_suffix('.tar.gz'))
            # Create tar.gz of the index directory
            shutil.make_archive(base_name=str(Path(self.faiss_index_path)), format='gztar', root_dir=str(Path(self.faiss_index_path).parent), base_dir=str(Path(self.faiss_index_path).name))
            # Read bytes
            with open(tmp_archive, 'rb') as f:
                data = f.read()
            mongo = MongoMemoryStore(self.config.get('mongo_uri'), self.config.get('mongo_db'), collection_name=self.config.get('memory_collection', 'conversations'))
            archive_name = Path(self.faiss_index_path).name + ".tar.gz"
            saved = mongo.save_faiss_index_archive(archive_name, data)
            if saved:
                logger.info(f"FAISS index archived to GridFS as {archive_name}")
            # cleanup temporary archive file
            try:
                os.remove(tmp_archive)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Could not save FAISS archive to MongoDB GridFS: {e}")

    def as_retriever(self):
        """Return the vector store as a LangChain retriever with user filtering."""
        if self.vector_store:
            return self.vector_store.as_retriever(
                search_kwargs={"k": self.config.get("retrieval_top_k", 3)}
            )
        return None

    def get_relevant_documents(self, query: str, user_id: Optional[str] = None, session_id: Optional[str] = None, source_ids: Optional[List[str]] = None) -> List[Document]:
        """
        Retrieve relevant documents for a query with user, session, and source filtering.
        
        Args:
            query: Search query
            user_id: Override user_id for this query (defaults to instance user_id)
            session_id: Filter by session/notebook (optional, for better isolation)
            source_ids: Filter by specific source IDs (optional, for selected sources only)
            
        Returns:
            List of documents filtered by user_id, session_id, and optionally source_ids
        """
        if not self.vector_store:
            return []
        
        # Use provided user_id or instance user_id
        filter_user_id = user_id or self.user_id
        
        # Retrieve more docs than needed, then filter by user_id, session_id, and source_ids
        k = self.config.get("retrieval_top_k", 3)
        
        # Log total documents before search
        total_docs = self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else "unknown"
        logger.info(f"🔍 FAISS index has {total_docs} total documents")
        
        # When filtering by session/source, retrieve many more docs to ensure we find matches
        # Otherwise similarity search might not include the filtered docs in top results
        search_k = k * 50 if (session_id or source_ids) else k * 5
        all_docs = self.vector_store.similarity_search(query, k=search_k)  # Over-retrieve for filtering
        
        logger.info(f"🔍 Similarity search returned {len(all_docs)} documents (search_k={search_k}) before filtering")
        
        # Filter: user's docs + public docs, optionally by session and source
        filtered_docs = []
        for i, doc in enumerate(all_docs):
            doc_user = doc.metadata.get("user_id", "public")
            doc_session = doc.metadata.get("session_id", None)
            doc_source = doc.metadata.get("source_id", None)
            is_placeholder = doc.metadata.get("placeholder", False)
            
            # Skip placeholder documents
            if is_placeholder:
                logger.info(f"  Doc {i+1}: PLACEHOLDER - skipping")
                continue
            
            # Verbose logging for debugging
            logger.info(f"  Doc {i+1}: user={doc_user}, session={doc_session}, source={doc_source}")
            
            if filter_user_id is None:
                # No user context: only public docs
                if doc_user == "public":
                    filtered_docs.append(doc)
            else:
                # User context: user's docs + public docs
                if doc_user == filter_user_id or doc_user == "public":
                    # If session_id filtering is requested, apply it
                    if session_id and doc_session != session_id:
                        logger.info(f"    ❌ Skipping: session mismatch ({doc_session} != {session_id})")
                        continue
                    
                    # If source_ids filtering is requested, apply it
                    if source_ids and doc_source not in source_ids:
                        logger.info(f"    ❌ Skipping: source not in filter ({doc_source} not in {source_ids})")
                        continue
                    
                    logger.info(f"    ✅ MATCHED!")
                    filtered_docs.append(doc)
                else:
                    logger.info(f"    ❌ Skipping: user mismatch ({doc_user} != {filter_user_id})")
            
            if len(filtered_docs) >= k:
                break
        
        logger.info(f"Retrieved {len(filtered_docs)} documents (user={filter_user_id}, session={session_id}, sources={source_ids})")
        return filtered_docs[:k]
    
    def add_user_documents(
        self,
        documents: List[Document],
        user_id: str,
        save_index: bool = True
    ):
        """
        Add documents for a specific user with metadata tagging.
        Updates global shared vector store.
        
        Args:
            documents: List of LangChain Document objects
            user_id: User identifier to tag documents
            save_index: Whether to persist FAISS index after adding
        """
        global _shared_vector_store
        
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return
        
        logger.info(f"📥 Adding {len(documents)} documents for user: {user_id}")
        
        # Tag all documents with user_id (preserve existing metadata)
        for doc in documents:
            if "user_id" not in doc.metadata:
                doc.metadata["user_id"] = user_id
            logger.info(f"  Document: {doc.metadata.get('source', 'unknown')} ({len(doc.page_content)} chars)")
        
        # Split and add to vector store (preserves metadata in chunks)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        logger.info(f"  Split into {len(texts)} chunks")
        
        # Log metadata for debugging
        if texts:
            logger.info(f"  Sample chunk metadata: {texts[0].metadata}")
        
        # Get current index size before adding
        current_size = self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0
        logger.info(f"  FAISS index size BEFORE: {current_size} documents")
        
        self.vector_store.add_documents(texts)
        
        # Get new index size after adding
        new_size = self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0
        logger.info(f"  FAISS index size AFTER: {new_size} documents (+{new_size - current_size})")
        
        # Update global shared reference
        _shared_vector_store = self.vector_store
        
        if save_index:
            self.vector_store.save_local(self.faiss_index_path)
            logger.info(f"  ✅ Saved FAISS index to disk: {self.faiss_index_path}")
        
        logger.info(f"✅ Successfully added {len(texts)} document chunks for user: {user_id}")
    
    def delete_documents_by_metadata(
        self,
        user_id: str,
        source_id: Optional[str] = None,
        session_id: Optional[str] = None,
        save_index: bool = True
    ) -> int:
        """
        Delete documents from FAISS index by metadata filtering.
        
        Args:
            user_id: User identifier (required for security)
            source_id: Delete all docs with this source_id (optional)
            session_id: Delete all docs with this session_id (optional)
            save_index: Whether to persist FAISS index after deletion
            
        Returns:
            Number of documents deleted
        """
        global _shared_vector_store
        
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return 0
        
        # Get all documents from the docstore
        docstore = self.vector_store.docstore
        index_to_docstore_id = self.vector_store.index_to_docstore_id
        
        if not hasattr(docstore, '_dict'):
            logger.error("Docstore doesn't support direct access")
            return 0
        
        # Find document IDs to delete
        ids_to_delete = []
        for idx, docstore_id in index_to_docstore_id.items():
            doc = docstore._dict.get(docstore_id)
            if not doc:
                continue
            
            metadata = doc.metadata
            doc_user = metadata.get('user_id')
            doc_source = metadata.get('source_id')
            doc_session = metadata.get('session_id')
            
            # Security: only delete own documents
            if doc_user != user_id:
                continue
            
            # Match source_id or session_id
            should_delete = False
            if source_id and doc_source == source_id:
                should_delete = True
            if session_id and doc_session == session_id:
                should_delete = True
            
            if should_delete:
                ids_to_delete.append((idx, docstore_id))
        
        if not ids_to_delete:
            logger.info(f"No documents found to delete (user={user_id}, source={source_id}, session={session_id})")
            return 0
        
        # Delete from docstore and index
        for idx, docstore_id in ids_to_delete:
            # Remove from docstore
            if docstore_id in docstore._dict:
                del docstore._dict[docstore_id]
            # Remove from index mapping
            if idx in index_to_docstore_id:
                del index_to_docstore_id[idx]
        
        # Rebuild FAISS index with remaining documents
        remaining_docs = list(docstore._dict.values())
        if remaining_docs:
            # Create new FAISS index from remaining documents
            texts = [doc.page_content for doc in remaining_docs]
            metadatas = [doc.metadata for doc in remaining_docs]
            
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            _shared_vector_store = self.vector_store
        else:
            # No documents left, create empty index
            logger.warning("All documents deleted, creating empty FAISS index")
            self.vector_store = FAISS.from_texts(
                texts=["placeholder"],
                embedding=self.embeddings,
                metadatas=[{"user_id": "system", "placeholder": True}]
            )
            _shared_vector_store = self.vector_store
        
        if save_index:
            self.vector_store.save_local(self.faiss_index_path)
        
        logger.info(f"✓ Deleted {len(ids_to_delete)} document chunks (user={user_id}, source={source_id}, session={session_id})")
        return len(ids_to_delete)


def create_rag_retriever(config, user_id: Optional[str] = None):
    return RAGRetriever(config, user_id=user_id)
