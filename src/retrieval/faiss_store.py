"""
FAISS Vector Store — Retrieval Layer
======================================
Builds and queries a FAISS IndexFlatIP (inner product = cosine similarity
when vectors are L2-normalized) over document chunks.

Persists the index and chunk metadata to disk so re-indexing is not needed
on every application restart.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


class FAISSStore:
    """
    In-memory FAISS vector store with disk persistence.

    Attributes:
        index_dir: Directory where index files are saved.
        embedding_dim: Dimension of the embedding vectors (384 for all-MiniLM).
    """

    INDEX_FILE = "faiss.index"
    METADATA_FILE = "chunks_metadata.pkl"

    def __init__(self, index_dir: str = "data/index", embedding_dim: int = 384):
        """
        Initialize the FAISS store.

        Args:
            index_dir: Directory to save/load the FAISS index.
            embedding_dim: Embedding dimensionality (must match embedder).
        """
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.index: Optional[Any] = None
        self.chunks: List[Dict[str, Any]] = []   # Parallel list to index rows

    # ──────────────────────────────────────────────────────────────────────────
    # BUILD
    # ──────────────────────────────────────────────────────────────────────────

    def build(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """
        Build a new FAISS index from chunks and their embeddings.

        Args:
            chunks: List of chunk dicts (with text, source_file, page_num, etc.)
            embeddings: np.ndarray of shape (N, embedding_dim), L2-normalized.
        """
        assert len(chunks) == len(embeddings), (
            f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
        )

        print(f"[INFO] Building FAISS IndexFlatIP over {len(chunks)} chunks...")

        # IndexFlatIP → exact inner product search (= cosine when L2-normalized)
        self.index = self.faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks

        print(f"[INFO] FAISS index built. Total vectors: {self.index.ntotal}")
        self.save()

    # ──────────────────────────────────────────────────────────────────────────
    # SEARCH
    # ──────────────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-K most similar chunks to a query vector.

        Args:
            query_vector: np.ndarray of shape (1, embedding_dim) or (embedding_dim,).
            k: Number of results to return.

        Returns:
            List of result dicts, each containing:
              - All original chunk fields (text, source_file, page_num, …)
              - "similarity_score": float in [0, 1] (cosine similarity)
              - "rank": 1-indexed rank
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Call build() or load() first.")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype(np.float32)
        k = min(k, self.index.ntotal)

        distances, indices = self.index.search(query_vector, k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            result = dict(self.chunks[idx])
            result["similarity_score"] = float(max(0.0, min(1.0, dist)))  # clamp to [0,1]
            result["rank"] = rank + 1
            results.append(result)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        index_path = self.index_dir / self.INDEX_FILE
        meta_path = self.index_dir / self.METADATA_FILE

        self.faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"[INFO] Index saved → {index_path}")
        print(f"[INFO] Metadata saved → {meta_path}")

    def load(self) -> bool:
        """
        Load a previously saved FAISS index from disk.

        Returns:
            True if loaded successfully, False if files not found.
        """
        index_path = self.index_dir / self.INDEX_FILE
        meta_path = self.index_dir / self.METADATA_FILE

        if not index_path.exists() or not meta_path.exists():
            print("[INFO] No existing FAISS index found. Please run indexing first.")
            return False

        self.index = self.faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"[INFO] Loaded FAISS index: {self.index.ntotal} vectors, "
              f"{len(self.chunks)} chunks.")
        return True

    def is_loaded(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)


if __name__ == "__main__":
    # Smoke test with random vectors
    import numpy as np

    dim = 384
    store = FAISSStore(index_dir="data/index", embedding_dim=dim)

    # Fake chunks + embeddings
    fake_chunks = [
        {"chunk_id": f"c{i}", "text": f"Sample text {i}", "source_file": "test.pdf", "page_num": i}
        for i in range(10)
    ]
    fake_embeddings = np.random.randn(10, dim).astype(np.float32)
    # L2-normalize
    norms = np.linalg.norm(fake_embeddings, axis=1, keepdims=True)
    fake_embeddings /= norms

    store.build(fake_chunks, fake_embeddings)

    query_vec = np.random.randn(1, dim).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)

    results = store.search(query_vec, k=3)
    print("\nTop-3 results:")
    for r in results:
        print(f"  Rank {r['rank']}: {r['text']} | score={r['similarity_score']:.4f}")
