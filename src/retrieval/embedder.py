"""
Embedder — Retrieval Layer
===========================
Loads the sentence-transformers all-MiniLM-L6-v2 model locally (no internet
required after first download) and converts text into 384-dimensional vectors.

This is the same model used for both document indexing and query encoding,
ensuring the embedding space is consistent across retrieval.
"""

import numpy as np
from typing import List, Union
from pathlib import Path


class Embedder:
    """
    Wraps sentence-transformers to produce 384-dim dense embeddings.

    Usage:
        embedder = Embedder()
        vectors = embedder.embed(["Hello world", "RAG systems"])
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Load the embedding model.

        Args:
            model_name: HuggingFace model identifier. Cached locally after
                        first download. Works offline thereafter.
        """
        print(f"[INFO] Loading embedding model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"[INFO] Embedding model loaded. Dim = {self.embedding_dim}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode one or more text strings into dense vectors.

        Args:
            texts: A single string or list of strings.
            batch_size: Number of texts to encode per batch.
            show_progress: Show tqdm progress bar for large corpora.

        Returns:
            np.ndarray of shape (N, 384) with float32 values.
            If a single string is given, returns shape (1, 384).
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalize for cosine similarity via dot product
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string. Returns shape (1, 384).
        """
        return self.embed([query])

    def embed_chunks(
        self,
        chunks: List[dict],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of chunk dicts (from chunker.chunk_pages).

        Args:
            chunks: List of chunk dicts with "text" key.

        Returns:
            np.ndarray of shape (N_chunks, 384).
        """
        texts = [c["text"] for c in chunks]
        print(f"[INFO] Embedding {len(texts)} chunks...")
        embeddings = self.embed(texts, batch_size=batch_size, show_progress=show_progress)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    # Quick smoke test
    embedder = Embedder()
    vecs = embedder.embed(["What is RAG?", "Hallucination in LLMs is a problem."])
    print(f"Shape: {vecs.shape}")
    print(f"First vector norm: {np.linalg.norm(vecs[0]):.4f}  (should be ~1.0)")
    # Cosine similarity (since vectors are L2-normalized, dot product = cosine)
    cos_sim = float(np.dot(vecs[0], vecs[1]))
    print(f"Cosine similarity between the two sentences: {cos_sim:.4f}")
