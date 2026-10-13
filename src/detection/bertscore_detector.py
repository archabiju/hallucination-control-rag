"""
BERTScore Detector — Detection Layer
======================================
Computes token-level semantic similarity between each generated sentence
and its source chunk using BERTScore. Unlike string-matching metrics
(ROUGE, BLEU), BERTScore uses contextual BERT embeddings so it captures
semantic meaning even when different words are used.

Score range: 0.0 → 1.0
  - F1 >= 0.75 → Strong semantic overlap (supported)
  - F1 0.55–0.75 → Moderate overlap (weak support)
  - F1 < 0.55 → Poor overlap (likely hallucinated)

Model: distilbert-base-uncased (lightweight, offline after first download)
"""

import numpy as np
from typing import List, Dict, Any, Optional


class BERTScoreDetector:
    """
    Computes BERTScore F1 between generated sentences and source chunks.
    Uses a lightweight distilBERT model for offline operation.
    """

    def __init__(
        self,
        lang: str = "en",
        model_type: str = "distilbert-base-uncased",
        device: Optional[str] = None,
    ):
        """
        Args:
            lang: Language code (default: "en").
            model_type: HuggingFace model for BERTScore computation.
            device: "cuda" or "cpu". Auto-detects if None.
        """
        self.lang = lang
        self.model_type = model_type
        self.device = device
        self._bert_score_available = False
        self.scorer = None

        try:
            import bert_score
            import torch
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._bert_score = bert_score
            self.scorer = bert_score.BERTScorer(
                model_type=self.model_type,
                lang=self.lang,
                device=device,
                rescale_with_baseline=False,
            )
            self._bert_score_available = True
            print(f"[INFO] BERTScore ready (cached model: {model_type})")
        except Exception as e:
            print(f"[WARN] bert-score init failed: {e}. Using cosine fallback.")
            self._bert_score_available = False

    def score_sentence(
        self,
        candidate: str,
        references: List[str],
    ) -> Dict[str, Any]:
        """
        Score a single generated sentence against a list of reference chunks.
        Returns the best (max F1) match.

        Args:
            candidate: Generated sentence to evaluate.
            references: List of retrieved chunk texts as references.

        Returns:
            Dict with:
              - bertscore_f1: Best F1 score across all references
              - bertscore_precision: Corresponding precision
              - bertscore_recall: Corresponding recall
              - best_reference_idx: Index of best-matching reference
        """
        if not candidate.strip() or not references:
            return {
                "bertscore_f1": 0.0,
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "best_reference_idx": 0,
            }

        if self._bert_score_available:
            return self._compute_bertscore(candidate, references)
        else:
            return self._cosine_fallback(candidate, references)

    def _compute_bertscore(
        self,
        candidate: str,
        references: List[str],
    ) -> Dict[str, Any]:
        """Compute BERTScore using the bert_score library."""
        try:
            best_f1 = 0.0
            best_p = 0.0
            best_r = 0.0
            best_ref_idx = 0

            # Batch inputs to prevent bert_score from reloading the model multiple times
            cands = [candidate] * len(references)
            if self.scorer is not None:
                P, R, F1 = self.scorer.score(cands, references, verbose=False)
            else:
                import torch
                device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                P, R, F1 = self._bert_score.score(
                    cands=cands,
                    refs=references,
                    lang=self.lang,
                    model_type=self.model_type,
                    device=device,
                    verbose=False,
                )

            for idx in range(len(references)):
                f1_val = float(F1[idx])
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_p = float(P[idx])
                    best_r = float(R[idx])
                    best_ref_idx = idx

            return {
                "bertscore_f1":        round(best_f1, 4),
                "bertscore_precision": round(best_p, 4),
                "bertscore_recall":    round(best_r, 4),
                "best_reference_idx":  best_ref_idx,
            }
        except Exception as e:
            print(f"[WARN] BERTScore computation failed: {e}. Using cosine fallback.")
            return self._cosine_fallback(candidate, references)

    def _cosine_fallback(
        self,
        candidate: str,
        references: List[str],
    ) -> Dict[str, Any]:
        """
        Lightweight fallback using TF-IDF-style cosine similarity.
        Works completely offline with no model downloads.
        """
        def word_vector(text: str) -> Dict[str, float]:
            words = text.lower().split()
            vec = {}
            for w in words:
                w = w.strip(".,!?;:\"'()")
                if len(w) > 2:
                    vec[w] = vec.get(w, 0) + 1
            # Normalize
            total = sum(vec.values()) or 1
            return {k: v / total for k, v in vec.items()}

        cand_vec = word_vector(candidate)
        best_f1 = 0.0
        best_ref_idx = 0

        for idx, ref in enumerate(references):
            ref_vec = word_vector(ref)

            # Compute cosine similarity between word frequency vectors
            all_words = set(cand_vec) | set(ref_vec)
            dot = sum(cand_vec.get(w, 0) * ref_vec.get(w, 0) for w in all_words)
            mag_c = (sum(v**2 for v in cand_vec.values()) ** 0.5) or 1e-9
            mag_r = (sum(v**2 for v in ref_vec.values()) ** 0.5) or 1e-9
            cos_sim = dot / (mag_c * mag_r)

            if cos_sim > best_f1:
                best_f1 = cos_sim
                best_ref_idx = idx

        return {
            "bertscore_f1":        round(best_f1, 4),
            "bertscore_precision": round(best_f1, 4),
            "bertscore_recall":    round(best_f1, 4),
            "best_reference_idx":  best_ref_idx,
        }

    def score_sentences(
        self,
        sentences: List[str],
        chunks: List[Dict[str, Any]],
        evidence=None,
        top_n: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Score all generated sentences against retrieved chunks or evidence windows.
        Batches the BERTScore computation across all sentences for fast CPU execution.

        Args:
            sentences: List of generated sentences.
            chunks: List of retrieved chunk dicts (with "text" key).
            evidence: EvidenceWindows instance (optional).
            top_n: Number of top windows to score against.

        Returns:
            List of BERTScore result dicts, one per sentence.
        """
        references = [c["text"] for c in chunks]

        all_cands = []
        all_refs = []
        pair_to_sent = []

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            refs = evidence.top(sentence, top_n) if (evidence is not None and evidence.windows) else references
            if not refs:
                continue
            for r_idx, r in enumerate(refs):
                all_cands.append(sentence)
                all_refs.append(r)
                pair_to_sent.append((i, sentence, r_idx))

        if not all_cands:
            return []

        if self._bert_score_available:
            try:
                if self.scorer is not None:
                    P, R, F1 = self.scorer.score(all_cands, all_refs, verbose=False, batch_size=16)
                else:
                    import torch
                    device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                    P, R, F1 = self._bert_score.score(
                        cands=all_cands,
                        refs=all_refs,
                        lang=self.lang,
                        model_type=self.model_type,
                        device=device,
                        verbose=False,
                        batch_size=16,
                    )

                best_per_sentence = {}
                for idx, (sent_idx, sentence, ref_idx) in enumerate(pair_to_sent):
                    f1_val = float(F1[idx])
                    p_val = float(P[idx])
                    r_val = float(R[idx])

                    if sent_idx not in best_per_sentence or f1_val > best_per_sentence[sent_idx]["bertscore_f1"]:
                        best_per_sentence[sent_idx] = {
                            "sentence": sentence,
                            "sentence_idx": sent_idx,
                            "bertscore_f1": round(f1_val, 4),
                            "bertscore_precision": round(p_val, 4),
                            "bertscore_recall": round(r_val, 4),
                            "best_reference_idx": ref_idx,
                        }

                return [best_per_sentence[k] for k in sorted(best_per_sentence.keys())]

            except Exception as e:
                print(f"[WARN] Batched BERTScore failed: {e}. Falling back to sequential.")

        # Fallback if batched failed or not available
        results = []
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            refs = evidence.top(sentence, top_n) if (evidence is not None and evidence.windows) else references
            res = self.score_sentence(sentence, refs)
            res["sentence"] = sentence
            res["sentence_idx"] = i
            results.append(res)
        return results


if __name__ == "__main__":
    detector = BERTScoreDetector()

    cand = "RAG combines retrieval with generation to produce grounded answers."
    refs = [
        "Retrieval-Augmented Generation uses a retriever to ground LLM outputs in documents.",
        "Climate change is a global environmental crisis.",
        "FAISS enables fast approximate nearest-neighbor search.",
    ]

    result = detector.score_sentence(cand, refs)
    print(f"\nCandidate: {cand}")
    print(f"BERTScore F1: {result['bertscore_f1']:.4f}")
    print(f"Best reference idx: {result['best_reference_idx']}")
    print(f"Best reference: {refs[result['best_reference_idx']]}")
