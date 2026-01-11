# core/memory_retrieval.py
"""
Retrieval Memory using FAISS + SentenceTransformers embeddings.
Stores conversational knowledge + scrape/search results.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class RetrievalMemory:
    def __init__(self, config: Dict[str, Any]):
        self.path_prefix = Path(config.get("faiss_index_path", "memory/faiss_index"))
        self.top_k = int(config.get("retrieval_top_k", 3))
        self.checkpoint_interval = int(config.get("faiss_checkpoint_interval", 60))
        self._meta_path = str(self.path_prefix) + ".meta.json"
        self._index_path = str(self.path_prefix) + ".index"

        # setup embedder
        model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        if SentenceTransformer is None:
            raise RuntimeError("sentence_transformers not available")
        self.embedder = SentenceTransformer(model_name)

        # index & meta
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self.last_save = time.time()

        self._load_or_init()

    def _load_or_init(self):
        # initialize FAISS index + meta store
        if faiss is None:
            raise RuntimeError("faiss not available")

        if Path(self._index_path).exists() and Path(self._meta_path).exists():
            try:
                self.index = faiss.read_index(self._index_path)
                raw = json.loads(Path(self._meta_path).read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    self.meta = raw
                else:
                    self.meta = []
            except Exception:
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self):
        dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dim)
        self.meta = []

    def add(self, text: str, source: str = "conversation", meta: Dict[str, Any] = None):
        if not text or not text.strip():
            return
        meta = meta or {}
        vecs = self.embedder.encode([text], convert_to_numpy=True)
        vec = np.array(vecs).astype("float32")
        # faiss expects 2D array
        try:
            self.index.add(vec)
            idx = len(self.meta)
            self.meta.append({"text": text, "source": source, "time": time.time(), "meta": meta})
        except Exception:
            # fallback: ignore
            pass

        # autosave
        if self.checkpoint_interval > 0 and (time.time() - self.last_save) > self.checkpoint_interval:
            self.save()

    def query(self, text: str, top_k: int = None) -> List[str]:
        """
        Return top-k texts (strings)
        """
        top_k = top_k or self.top_k
        if self.index.ntotal == 0:
            return []
        vec = self.embedder.encode([text], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(vec, top_k)
        out = []
        for idx in I[0]:
            if idx < len(self.meta):
                out.append(self.meta[idx]["text"])
        return out

    def query_with_meta(self, text: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Return list of dicts: {"text", "source", "time", "meta", "score" (optional)}
        """
        top_k = top_k or self.top_k
        if self.index is None or (hasattr(self.index, "ntotal") and self.index.ntotal == 0):
            return []
        vec = self.embedder.encode([text], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(vec, top_k)
        out = []
        for j, idx in enumerate(I[0]):
            if idx < len(self.meta):
                rec = dict(self.meta[idx])
                rec["score"] = float(D[0][j]) if (D is not None and len(D) and len(D[0]) > j) else None
                out.append(rec)
        return out

    def save(self):
        try:
            faiss.write_index(self.index, self._index_path)
        except Exception:
            pass
        try:
            Path(self._meta_path).write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        self.last_save = time.time()
