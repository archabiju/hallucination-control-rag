"""
Hallucination Engine — Detection Layer
========================================
Aggregates NLI entailment scores, BERTScore F1, and cosine similarity
into a unified Support Score for each generated sentence.

The Support Score drives the tri-state Decision Controller:

    Support Score = w_nli * NLI_entail + w_bert * BERTScore_F1 + w_cos * cosine_sim

Default weights (from config): 0.45 / 0.35 / 0.20

Classification:
  - SUPPORTED         → score >= high_threshold (default 0.75)
  - WEAKLY_SUPPORTED  → score >= medium_threshold (default 0.50)
  - HALLUCINATED      → score < medium_threshold

This is the aggregation layer that makes all three techniques work together,
each catching a different type of hallucination:
  - NLI:        logical contradictions ("X causes Y" when chunk says "Y prevents X")
  - BERTScore:  semantic drift (different topic, different meaning)
  - Cosine:     out-of-domain claims (nothing retrieved is close to the claim)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data class for a single sentence verdict
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SentenceVerdict:
    """Complete verdict for one generated sentence."""
    sentence_idx:    int
    sentence:        str

    # Component scores
    nli_score:       float   # NLI entailment probability
    bertscore_f1:    float   # BERTScore F1
    cosine_score:    float   # Cosine similarity to best chunk

    # Aggregated
    support_score:   float   # Weighted combination

    # Classification
    classification:  str     # SUPPORTED | WEAKLY_SUPPORTED | HALLUCINATED
    nli_label:       str     # ENTAILMENT | NEUTRAL | CONTRADICTION

    # Source attribution
    source_file:     str = ""
    source_page:     int = 0
    best_chunk_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination Engine
# ─────────────────────────────────────────────────────────────────────────────

class HallucinationEngine:
    """
    Combines NLI + BERTScore + Cosine into a final support score per sentence.
    """

    def __init__(
        self,
        weight_nli:       float = 0.45,
        weight_bertscore: float = 0.35,
        weight_cosine:    float = 0.20,
        high_threshold:   float = 0.75,
        medium_threshold: float = 0.50,
    ):
        """
        Args:
            weight_nli:       Weight for NLI entailment component.
            weight_bertscore: Weight for BERTScore F1 component.
            weight_cosine:    Weight for cosine similarity component.
            high_threshold:   Score >= this → SUPPORTED.
            medium_threshold: Score >= this (but < high) → WEAKLY_SUPPORTED.
                              Score < this → HALLUCINATED.
        """
        assert abs(weight_nli + weight_bertscore + weight_cosine - 1.0) < 0.01, \
            "Weights must sum to 1.0"

        self.w_nli       = weight_nli
        self.w_bert      = weight_bertscore
        self.w_cos       = weight_cosine
        self.high_thr    = high_threshold
        self.medium_thr  = medium_threshold

    def compute_support_score(
        self,
        nli_score:    float,
        bertscore_f1: float,
        cosine_score: float,
    ) -> float:
        """
        Compute the weighted aggregated support score.

        Args:
            nli_score:    Entailment probability from NLI detector.
            bertscore_f1: F1 score from BERTScore detector.
            cosine_score: Cosine similarity from FAISS / XAI layer.

        Returns:
            Float in [0, 1] — higher means better supported.
        """
        score = (
            self.w_nli  * nli_score +
            self.w_bert * bertscore_f1 +
            self.w_cos  * cosine_score
        )
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def classify(self, support_score: float) -> str:
        """
        Map a support score to a classification label.

        Returns:
            "SUPPORTED" | "WEAKLY_SUPPORTED" | "HALLUCINATED"
        """
        if support_score >= self.high_thr:
            return "SUPPORTED"
        elif support_score >= self.medium_thr:
            return "WEAKLY_SUPPORTED"
        else:
            return "HALLUCINATED"

    def analyze(
        self,
        sentences:    List[str],
        nli_results:  List[Dict[str, Any]],
        bert_results: List[Dict[str, Any]],
        attributions: List[Dict[str, Any]],
    ) -> List[SentenceVerdict]:
        """
        Produce a verdict for every generated sentence.

        Args:
            sentences:    Original generated sentences.
            nli_results:  Output from NLIDetector.score_sentences().
            bert_results: Output from BERTScoreDetector.score_sentences().
            attributions: Output from xai_layer.compute_chunk_attribution().

        Returns:
            List of SentenceVerdict objects.
        """
        verdicts = []

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Get NLI score for this sentence
            nli = self._get_by_idx(nli_results, i)
            nli_score = nli.get("nli_support_score", 0.0) if nli else 0.0
            nli_label = nli.get("label", "NEUTRAL") if nli else "NEUTRAL"

            # Get BERTScore for this sentence
            bert = self._get_by_idx(bert_results, i)
            bertscore_f1 = bert.get("bertscore_f1", 0.0) if bert else 0.0

            # Get cosine similarity from attribution
            attr = attributions[i] if i < len(attributions) else {}
            cosine_score = attr.get("attribution_score", 0.0)
            source_file  = attr.get("best_chunk_source", "")
            source_page  = attr.get("best_chunk_page", 0)
            best_chunk   = attr.get("best_chunk_text", "")

            # Aggregate
            support_score = self.compute_support_score(
                nli_score, bertscore_f1, cosine_score
            )
            classification = self.classify(support_score)

            verdicts.append(SentenceVerdict(
                sentence_idx=i,
                sentence=sentence,
                nli_score=nli_score,
                bertscore_f1=bertscore_f1,
                cosine_score=cosine_score,
                support_score=support_score,
                classification=classification,
                nli_label=nli_label,
                source_file=source_file,
                source_page=source_page,
                best_chunk_text=best_chunk,
            ))

        return verdicts

    def summarize(self, verdicts: List[SentenceVerdict]) -> Dict[str, Any]:
        """
        Produce a summary of all sentence verdicts for the decision controller.

        Returns:
            Dict with aggregate stats and per-classification counts.
        """
        if not verdicts:
            return {
                "overall_support_score": 0.0,
                "num_supported": 0,
                "num_weak": 0,
                "num_hallucinated": 0,
                "total_sentences": 0,
                "hallucination_ratio": 0.0,
                "supported_ratio": 0.0,
            }

        total = len(verdicts)
        n_supported = sum(1 for v in verdicts if v.classification == "SUPPORTED")
        n_weak      = sum(1 for v in verdicts if v.classification == "WEAKLY_SUPPORTED")
        n_hall      = sum(1 for v in verdicts if v.classification == "HALLUCINATED")
        avg_score   = np.mean([v.support_score for v in verdicts])

        return {
            "overall_support_score": round(float(avg_score), 4),
            "num_supported":         n_supported,
            "num_weak":              n_weak,
            "num_hallucinated":      n_hall,
            "total_sentences":       total,
            "hallucination_ratio":   round(n_hall / total, 4),
            "supported_ratio":       round(n_supported / total, 4),
        }

    @staticmethod
    def _get_by_idx(
        results: List[Dict[str, Any]],
        idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Get result dict for a given sentence_idx."""
        for r in results:
            if r.get("sentence_idx") == idx:
                return r
        return None


if __name__ == "__main__":
    engine = HallucinationEngine()

    # Simulate component outputs
    test_verdicts_input = [
        ("RAG uses retrieval for factual grounding.", 0.82, 0.79, 0.88),
        ("The moon is made of cheese.",               0.05, 0.12, 0.08),
        ("Hallucination detection uses NLI methods.", 0.62, 0.58, 0.70),
    ]

    print("Sentence Verdicts:")
    for i, (sent, nli, bert, cos) in enumerate(test_verdicts_input):
        score = engine.compute_support_score(nli, bert, cos)
        cls   = engine.classify(score)
        print(f"\n[{i}] '{sent}'")
        print(f"     NLI={nli:.2f} | BERTScore={bert:.2f} | Cosine={cos:.2f}")
        print(f"     → Support Score: {score:.4f}  |  Classification: {cls}")
