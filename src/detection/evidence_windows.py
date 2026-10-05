"""
Evidence Windows — sentence-level premises for verification
============================================================
Verifying a claim against a whole 512-token chunk dilutes the signal: a small NLI
cross-encoder is unreliable on long premises, and BERTScore against a long chunk is
compressed into a narrow band for supported and unsupported claims alike.

Following the sentence-level strategy of SummaC (paper ref. [15]), every retrieved
chunk is split into short windows of `size` consecutive sentences, and each claim is
verified against the `top_n` windows most similar to it (by Sentence-BERT cosine).
NLI and BERTScore then take the best-supported window.
"""
from typing import List, Dict, Any
import numpy as np

from src.ingestion.chunker import split_into_sentences


class EvidenceWindows:
    def __init__(self, chunks: List[Dict[str, Any]], embedder, size: int = 2):
        self.windows: List[str] = []
        self.sources: List[Dict[str, Any]] = []
        for c in chunks:
            sents = [s for s in split_into_sentences(c["text"]) if len(s.split()) >= 3]
            if not sents:
                continue
            step_end = max(len(sents) - size + 1, 1)
            for i in range(step_end):
                self.windows.append(" ".join(sents[i:i + size]))
                self.sources.append({"source_file": c.get("source_file", ""),
                                     "page_num": c.get("page_num", 0)})
        self.embedder = embedder
        self.vecs = embedder.embed(self.windows) if self.windows else np.zeros((0, 1))

    def top(self, claim: str, n: int = 4) -> List[str]:
        """The n windows most semantically similar to the claim."""
        if not self.windows:
            return []
        sims = self.vecs @ np.asarray(self.embedder.embed_query(claim)).reshape(-1)
        return [self.windows[i] for i in np.argsort(-sims)[:n]]
