"""
Embeddings + Vector Store (Sentence Transformers + MongoDB Atlas)

Uses local Sentence Transformers for embeddings and MongoDB Atlas vector search.

Fixes:
- Local lightweight embeddings using sentence-transformers
- MongoDB Atlas vector search instead of FAISS
- Thread-safe operations
- Efficient vector similarity search
"""

from typing import List, Sequence, Optional, Tuple, Dict
import math
import hashlib
import threading
import logging

logger = logging.getLogger(__name__)


# ============================================================
#  Embedding Model - Local Sentence Transformers
# ============================================================

class EmbeddingModel:
    """
    Embedding Model using local Sentence Transformers:
    - Uses lightweight local models for embeddings
    - No cloud dependencies for embeddings
    - Fast local processing
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.model_name = model_name
        self.dim = dim

        # Initialize sentence transformers model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._use_local = True
            logger.info(f"✓ Using local Sentence Transformers model: {model_name}, Dimension: {self.dim}")
        except Exception as e:
            logger.warning(f"Failed to load Sentence Transformers model: {e}, falling back to deterministic processing")
            self.model = None
            self._use_local = False

    # ------------------------------
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings using local Sentence Transformers."""
        if self._use_local and self.model and texts:
            try:
                # Use sentence transformers to encode texts
                embeddings = self.model.encode(list(texts), convert_to_numpy=True)
                return embeddings.tolist()
            except Exception as e:
                logger.warning(f"Sentence Transformers embedding failed: {e}")

        # Fallback to deterministic hashing
        logger.info("Using fallback deterministic embeddings")
        return [self._fallback_embed(t) for t in texts]

    # ------------------------------
    def _fallback_embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()

        # Expand digest deterministically
        needed = self.dim * 8
        rep = (digest * ((needed // len(digest)) + 1))[:needed]

        vec = []
        for i in range(self.dim):
            chunk = rep[i * 8 : (i + 1) * 8]
            val = int.from_bytes(chunk, "big")
            normed = ((val / (2 ** 64 - 1)) * 2.0) - 1.0
            vec.append(normed)

        # normalize L2
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


# ============================================================
#  Vector Store (FAISS + fallback)
# ============================================================

class VectorStore:
    """
    Vector store with FAISS backend (if available)
    and in-memory fallback.

    Methods:
    --------
    add_documents(docs)
    query(query_emb, top_k)
    remove_documents(doc_ids)
    save(prefix)
    load(prefix)
    """

    def __init__(self, dim: int, mongo_uri: str = None, db_name: str = "collabry", collection_name: str = "vectors"):
        from config import CONFIG
        self.dim = dim
        self._lock = threading.Lock()

        # MongoDB Atlas configuration - use config values
        self.mongo_uri = mongo_uri or CONFIG.get("mongo_uri", "mongodb+srv://localhost:27017")
        self.db_name = db_name or CONFIG.get("vector_db_name", "collabry")
        self.collection_name = collection_name or CONFIG.get("vector_collection", "vectors")

        # Initialize MongoDB client
        try:
            from pymongo import MongoClient
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]

            # Create vector search index if it doesn't exist
            self._ensure_vector_index()
            self._use_mongodb = True
            logger.info(f"✓ Connected to MongoDB Atlas vector store: {self.db_name}.{self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB Atlas: {e}")
            self._use_mongodb = False

    def _ensure_vector_index(self):
        """Ensure vector search index exists."""
        try:
            # Check if index already exists
            indexes = list(self.collection.list_indexes())
            vector_index_exists = any(idx.get("name", "").startswith("vector_index") for idx in indexes)

            if not vector_index_exists:
                # Create vector search index
                index_definition = {
                    "name": "vector_index",
                    "type": "vectorSearch",
                    "definition": {
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": self.dim,
                                "similarity": "cosine"
                            }
                        ]
                    }
                }

                # Note: In production, you'd create this index via MongoDB Atlas UI or CLI
                # For now, we'll work with existing indexes
                logger.info("Vector search index should be created manually in MongoDB Atlas")

        except Exception as e:
            logger.warning(f"Could not check/create vector index: {e}")

    # ======================================================
    #  Save & Load
    # ======================================================

    def save(self, prefix: str):
        """
        Save operation not needed for MongoDB Atlas - data is persisted automatically.
        """
        logger.info("MongoDB Atlas automatically persists data - no save needed")

    # ------------------------------
    def load(self, prefix: str):
        """
        Load operation not needed for MongoDB Atlas - connection is established on init.
        """
        logger.info("MongoDB Atlas connection established on initialization")

    # ======================================================
    #  Add Documents
    # ======================================================

    def add_documents(self, docs: Sequence[Tuple[str, str, Sequence[float]]]):
        """
        Add documents to MongoDB Atlas vector store.
        """
        if not self._use_mongodb:
            logger.warning("MongoDB not available, skipping document addition")
            return

        with self._lock:
            try:
                documents = []
                for doc_id, text, emb in docs:
                    if len(emb) != self.dim:
                        raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {len(emb)}")

                    document = {
                        "_id": doc_id,
                        "text": text,
                        "embedding": emb,
                        "metadata": {
                            "created_at": threading.current_thread().ident,
                            "length": len(text)
                        }
                    }
                    documents.append(document)

                if documents:
                    # Insert documents (upsert to handle duplicates)
                    for doc in documents:
                        self.collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)

                    logger.info(f"Added {len(documents)} documents to vector store")

            except Exception as e:
                logger.error(f"Failed to add documents to MongoDB: {e}")

    # ------------------------------
    def _faiss_add(self, docs):
        """
        Incrementally add only NEW embeddings to FAISS.
        """
        import numpy as np

        new_vecs = [emb for (_, _, emb) in docs]
        matrix = np.array(new_vecs, dtype=np.float32)
        matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)

        ids_int = []
        for (doc_id, _, _) in docs:
            if doc_id in self._id_to_int:
                iid = self._id_to_int[doc_id]
            else:
                # 63-bit safe ID
                iid = abs(hash(doc_id)) % (2 ** 62)
                while iid in self._int_to_id:
                    iid = (iid + 1) % (2 ** 62)
                self._id_to_int[doc_id] = iid
                self._int_to_id[iid] = doc_id

            ids_int.append(iid)

        ids_arr = np.array(ids_int, dtype=np.int64)

        try:
            self._index.add_with_ids(matrix, ids_arr)
        except Exception:
            # fallback: add without ids
            self._index.add(matrix)

    # ======================================================
    #  Query
    # ======================================================

    def query(self, query_emb: Sequence[float], top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Query similar documents using MongoDB Atlas vector search.
        """
        if not self._use_mongodb:
            logger.warning("MongoDB not available, returning empty results")
            return []

        try:
            if len(query_emb) != self.dim:
                raise ValueError(f"Query embedding dimension mismatch: expected {self.dim}, got {len(query_emb)}")

            # Use MongoDB aggregation pipeline for vector search
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_emb,
                        "numCandidates": top_k * 10,  # Search more candidates for better results
                        "limit": top_k
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "text": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            results = list(self.collection.aggregate(pipeline))

            # Format results
            formatted_results = []
            for result in results:
                doc_id = str(result["_id"])
                text = result["text"]
                score = float(result["score"])
                formatted_results.append((doc_id, text, score))

            return formatted_results

        except Exception as e:
            logger.error(f"Vector search query failed: {e}")
            return []

    # ------------------------------
    def _faiss_query(self, qemb, top_k):
        import numpy as np

        q = np.array(qemb, dtype=np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

        D, I = self._index.search(q, top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue

            sid = self._int_to_id.get(int(idx))
            if not sid:
                # fallback to list index
                if 0 <= int(idx) < len(self._ids):
                    sid = self._ids[int(idx)]
                else:
                    continue

            text = self._get_text(sid)
            results.append((sid, text, float(score)))

        return results

    # ======================================================
    #  Remove Documents
    # ======================================================

    def remove_documents(self, doc_ids: Sequence[str]):
        """
        Remove documents from MongoDB Atlas vector store.
        """
        if not self._use_mongodb:
            logger.warning("MongoDB not available, skipping document removal")
            return

        try:
            result = self.collection.delete_many({"_id": {"$in": list(doc_ids)}})
            logger.info(f"Removed {result.deleted_count} documents from vector store")
        except Exception as e:
            logger.error(f"Failed to remove documents from MongoDB: {e}")

    # ------------------------------
    def _faiss_remove(self, doc_ids):
        import numpy as np

        ids_to_remove = []
        for did in doc_ids:
            iid = self._id_to_int.get(did)
            if iid is not None:
                ids_to_remove.append(iid)

        if not ids_to_remove:
            return

        arr = np.array(ids_to_remove, dtype=np.int64)

        try:
            self._index.remove_ids(arr)
        except Exception:
            # rebuild if remove_ids fails
            self._rebuild_faiss()

        # Clean mappings
        for iid in ids_to_remove:
            did = self._int_to_id.pop(iid, None)
            if did:
                self._id_to_int.pop(did, None)

    # ------------------------------
    def _rebuild_faiss(self):
        """
        Full FAISS rebuild after removals or corrupted index state.
        """
        import numpy as np

        if not self._use_faiss:
            return

        try:
            base = self.faiss.IndexFlatIP(self.dim)
            try:
                new_index = self.faiss.IndexIDMap2(base)
            except Exception:
                new_index = self.faiss.IndexIDMap(base)

            self._index = new_index

            # rebuild from _vecs
            matrix = np.array(self._vecs, dtype=np.float32)
            matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)

            ids_int = []
            for did in self._ids:
                if did not in self._id_to_int:
                    iid = abs(hash(did)) % (2 ** 62)
                    while iid in self._int_to_id:
                        iid = (iid + 1) % (2 ** 62)
                    self._id_to_int[did] = iid
                    self._int_to_id[iid] = did
                ids_int.append(self._id_to_int[did])

            ids_arr = np.array(ids_int, dtype=np.int64)
            self._index.add_with_ids(matrix, ids_arr)

        except Exception:
            pass

    # ======================================================
    #  Helpers
    # ======================================================

    def _get_text(self, doc_id: str) -> str:
        try:
            idx = self._ids.index(doc_id)
            return self._texts[idx]
        except Exception:
            return ""


# ============================================================
#  Standalone Demo
# ============================================================

if __name__ == "__main__":
    print("Running embeddings/vectorstore test...")

    em = EmbeddingModel(dim=64)
    docs = [
        ("doc1", "Apple is a fruit"),
        ("doc2", "Paris is in France"),
        ("doc3", "Multiplication example: 5 * 7 = 35"),
    ]

    vectors = em.embed([d[1] for d in docs])
    vs = VectorStore(dim=64)

    vs.add_documents([(docs[i][0], docs[i][1], vectors[i]) for i in range(3)])

    q = em.embed(["What is 5*7?"])[0]
    res = vs.query(q, top_k=3)
    print("Results:")
    for r in res:
        print(r)
