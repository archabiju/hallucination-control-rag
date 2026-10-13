"""
NLI Detector — Detection Layer
================================
Uses a DeBERTa-based NLI cross-encoder to determine whether each
generated sentence (hypothesis) is logically entailed, neutral, or
contradicted by the best-matching retrieved chunk (premise).

Model: cross-encoder/nli-deberta-v3-small
  - Offline, cached by HuggingFace after first download
  - Produces 3 logits: [CONTRADICTION, NEUTRAL, ENTAILMENT]
  - Much lighter than full DeBERTa-large while maintaining strong accuracy

NLI Labels:
  - ENTAILMENT   → The chunk logically supports the sentence  ✓
  - NEUTRAL      → The chunk neither supports nor contradicts  ~
  - CONTRADICTION → The chunk contradicts the sentence         ✗
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


# Label mapping for cross-encoder/nli-deberta-v3-small
# Index 0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT
NLI_LABELS = ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]
ENTAILMENT_IDX = 2
NEUTRAL_IDX = 1
CONTRADICTION_IDX = 0


class NLIDetector:
    """
    Sentence-pair NLI scorer using a local DeBERTa cross-encoder.

    For each (premise=chunk, hypothesis=sentence) pair, it outputs:
      - entail_prob: probability the chunk entails the sentence
      - neutral_prob: probability the chunk is neutral
      - contradict_prob: probability the chunk contradicts the sentence
      - label: ENTAILMENT | NEUTRAL | CONTRADICTION
      - nli_support_score: entail_prob (used as the NLI component of support score)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
    ):
        print(f"[INFO] Loading NLI model: {model_name}")
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, num_labels=3)
            self.model_name = model_name
            print("[INFO] NLI model loaded successfully.")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        except Exception as e:
            print(f"[WARN] Could not load NLI model '{model_name}': {e}")
            print("[WARN] Falling back to heuristic NLI scoring.")
            self.model = None
            self.model_name = "heuristic-fallback"

    def score_pair(
        self,
        premise: str,
        hypothesis: str,
    ) -> Dict[str, Any]:
        """
        Score a single (premise, hypothesis) pair.

        Args:
            premise: The source chunk text (retrieved document chunk).
            hypothesis: The generated sentence to verify.

        Returns:
            Dict with entail_prob, neutral_prob, contradict_prob, label, nli_support_score.
        """
        if self.model is None:
            return self._heuristic_score(premise, hypothesis)

        try:
            # CrossEncoder returns raw logits for [CONTRADICTION, NEUTRAL, ENTAILMENT]
            logits = self.model.predict(
                [(premise, hypothesis)],
                apply_softmax=True,
            )[0]  # shape: (3,)

            contradict_prob = float(logits[CONTRADICTION_IDX])
            neutral_prob    = float(logits[NEUTRAL_IDX])
            entail_prob     = float(logits[ENTAILMENT_IDX])

            label_idx = int(np.argmax(logits))
            label = NLI_LABELS[label_idx]

            return {
                "entail_prob":      round(entail_prob, 4),
                "neutral_prob":     round(neutral_prob, 4),
                "contradict_prob":  round(contradict_prob, 4),
                "label":            label,
                "nli_support_score": round(entail_prob, 4),
            }
        except Exception as e:
            print(f"[WARN] NLI scoring failed: {e}. Using heuristic.")
            return self._heuristic_score(premise, hypothesis)

    def score_pairs(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Score multiple (premise, hypothesis) pairs in a fast batch.
        """
        if not pairs:
            return []
        if self.model is None:
            return [self._heuristic_score(p, h) for p, h in pairs]

        try:
            logits = self.model.predict(
                pairs,
                batch_size=batch_size,
                apply_softmax=True,
                show_progress_bar=False,
            )
            results = []
            for item in logits:
                contradict_prob = float(item[CONTRADICTION_IDX])
                neutral_prob    = float(item[NEUTRAL_IDX])
                entail_prob     = float(item[ENTAILMENT_IDX])
                label_idx = int(np.argmax(item))
                results.append({
                    "entail_prob":       round(entail_prob, 4),
                    "neutral_prob":      round(neutral_prob, 4),
                    "contradict_prob":   round(contradict_prob, 4),
                    "label":             NLI_LABELS[label_idx],
                    "nli_support_score": round(entail_prob, 4),
                })
            return results
        except Exception as e:
            print(f"[WARN] Batch NLI scoring failed: {e}. Using heuristic.")
            return [self._heuristic_score(p, h) for p, h in pairs]

    def score_sentences(
        self,
        sentences: List[str],
        chunks: List[Dict[str, Any]],
        attributions: Optional[List[Dict[str, Any]]] = None,
        evidence=None,
        top_n: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Score multiple sentences against their best-matching chunks using fast batching.

        If `evidence` (an EvidenceWindows) is given, each sentence is instead verified
        against its `top_n` most similar short evidence windows and the window with the
        highest entailment probability is used (sentence-level NLI, SummaC-style).
        """
        all_pairs = []
        pair_to_sent = []
        valid_indices = set()

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            valid_indices.add(i)

            if evidence is not None and evidence.windows:
                premises = evidence.top(sentence, top_n)
            elif attributions and i < len(attributions):
                premises = [attributions[i]["best_chunk_text"]]
            elif chunks:
                premises = [chunks[0]["text"]]
            else:
                premises = []

            for p in premises:
                all_pairs.append((p, sentence))
                pair_to_sent.append((i, sentence))

        if not all_pairs:
            # Fallback for empty pairs
            return [{
                "sentence": sentence,
                "sentence_idx": i,
                "entail_prob": 0.0,
                "neutral_prob": 1.0,
                "contradict_prob": 0.0,
                "label": "NEUTRAL",
                "nli_support_score": 0.0,
            } for i, sentence in enumerate(sentences) if sentence.strip()]

        scored_pairs = self.score_pairs(all_pairs)

        # Aggregate: pick best entailment per sentence
        best_per_sentence = {}
        for (i, sentence), res in zip(pair_to_sent, scored_pairs):
            if i not in best_per_sentence or res["entail_prob"] > best_per_sentence[i]["entail_prob"]:
                res_copy = dict(res)
                res_copy["sentence"] = sentence
                res_copy["sentence_idx"] = i
                best_per_sentence[i] = res_copy

        return [best_per_sentence[i] for i in sorted(best_per_sentence.keys())]

    def _heuristic_score(
        self,
        premise: str,
        hypothesis: str,
    ) -> Dict[str, Any]:
        """
        Lightweight fallback when NLI model is unavailable.
        Uses word overlap ratio as a proxy for entailment.
        """
        premise_words = set(premise.lower().split())
        hyp_words = set(hypothesis.lower().split())

        if not hyp_words:
            overlap = 0.0
        else:
            common = hyp_words & premise_words
            # Remove stop words from overlap count
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "be",
                         "been", "being", "have", "has", "had", "do", "does",
                         "did", "will", "would", "could", "should", "may",
                         "might", "shall", "can", "need", "to", "of", "in",
                         "on", "at", "by", "for", "with", "this", "that",
                         "and", "or", "but", "not", "it", "its"}
            meaningful_common = common - stopwords
            meaningful_hyp = hyp_words - stopwords
            overlap = len(meaningful_common) / max(len(meaningful_hyp), 1)

        # Map overlap ratio to NLI probs
        entail_prob     = min(0.85, overlap * 0.9)
        contradict_prob = max(0.0, 0.1 - overlap * 0.1)
        neutral_prob    = 1.0 - entail_prob - contradict_prob

        if entail_prob >= 0.5:
            label = "ENTAILMENT"
        elif contradict_prob >= 0.3:
            label = "CONTRADICTION"
        else:
            label = "NEUTRAL"

        return {
            "entail_prob":      round(entail_prob, 4),
            "neutral_prob":     round(neutral_prob, 4),
            "contradict_prob":  round(contradict_prob, 4),
            "label":            label,
            "nli_support_score": round(entail_prob, 4),
        }


if __name__ == "__main__":
    detector = NLIDetector()

    # Test cases
    pairs = [
        (
            "RAG combines dense retrieval with a seq2seq language model to produce factual answers.",
            "RAG uses retrieval to ground language model outputs.",
        ),
        (
            "The Eiffel Tower is located in Paris, France.",
            "The Eiffel Tower is located in London.",
        ),
        (
            "Hallucinations occur when LLMs generate unsupported content.",
            "Climate change is caused by greenhouse gases.",
        ),
    ]

    for premise, hyp in pairs:
        result = detector.score_pair(premise, hyp)
        print(f"\nPremise : {premise[:60]}...")
        print(f"Hypothesis: {hyp}")
        print(f"Label: {result['label']} | Entail: {result['entail_prob']:.3f} | "
              f"Neutral: {result['neutral_prob']:.3f} | Contradict: {result['contradict_prob']:.3f}")
