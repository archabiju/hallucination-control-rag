"""
Evaluator — Evaluation Layer
===============================
Runs the full pipeline against a test set of QA pairs with ground-truth
hallucination labels, and produces a comprehensive evaluation report.

Test set format (JSON):
    [
        {
            "question": "What is RAG?",
            "expected_answer": "RAG combines retrieval with generation...",
            "claims": [
                {"text": "RAG uses a retriever.", "hallucinated": false},
                {"text": "RAG was invented in 2025.", "hallucinated": true}
            ]
        },
        ...
    ]

Usage:
    python evaluation/evaluator.py              ← run with built-in demo test set
    python evaluation/evaluator.py test.json    ← run with a custom test set
"""

import sys
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    precision_recall_f1,
    auroc,
    compute_confusion_matrix,
    plot_roc_curve,
    plot_support_score_distribution,
    plot_metrics_bar,
)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in demo test set (no external file needed)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_TEST_SET = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "claims": [
            {"text": "RAG combines retrieval systems with language models to produce grounded answers.", "hallucinated": False},
            {"text": "RAG retrieves relevant documents before generating a response.", "hallucinated": False},
            {"text": "RAG was developed by NASA for space exploration.", "hallucinated": True},
        ],
    },
    {
        "question": "How does hallucination detection work in RAG systems?",
        "claims": [
            {"text": "Hallucination detection identifies generated content not supported by retrieved evidence.", "hallucinated": False},
            {"text": "NLI entailment models check if retrieved chunks logically support generated sentences.", "hallucinated": False},
            {"text": "BERTScore measures semantic overlap between generated and source text.", "hallucinated": False},
            {"text": "Hallucination detection always requires internet access.", "hallucinated": True},
        ],
    },
    {
        "question": "What is FAISS used for?",
        "claims": [
            {"text": "FAISS is a library for efficient similarity search over dense vectors.", "hallucinated": False},
            {"text": "FAISS was developed by Facebook AI Research.", "hallucinated": False},
            {"text": "FAISS requires a PostgreSQL database to function.", "hallucinated": True},
            {"text": "FAISS supports billion-scale nearest neighbor search.", "hallucinated": False},
        ],
    },
    {
        "question": "What is Natural Language Inference?",
        "claims": [
            {"text": "NLI determines the logical relationship between a premise and hypothesis.", "hallucinated": False},
            {"text": "NLI classifies pairs as entailment, neutral, or contradiction.", "hallucinated": False},
            {"text": "NLI models can only work with English text.", "hallucinated": True},
            {"text": "DeBERTa is a state-of-the-art model for NLI tasks.", "hallucinated": False},
        ],
    },
    {
        "question": "What are the limitations of current RAG systems?",
        "claims": [
            {"text": "RAG systems can still hallucinate even with retrieved context.", "hallucinated": False},
            {"text": "Current systems lack integration between explainability and reliability.", "hallucinated": False},
            {"text": "All RAG systems achieve 100% accuracy on factual questions.", "hallucinated": True},
            {"text": "Most hallucination detection methods rely on external services.", "hallucinated": False},
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator class
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Evaluates the hallucination detection pipeline against ground-truth labels.
    """

    def __init__(
        self,
        output_dir: str = "evaluation/results",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_with_pipeline(
        self,
        test_set: List[Dict[str, Any]],
        pipeline=None,
    ) -> Dict[str, Any]:
        """
        Run each test question through the pipeline, compare verdicts to
        ground truth, and compute metrics.

        Args:
            test_set: List of test items (question + claims with labels).
            pipeline: RAGHallucinationPipeline instance (optional). If None,
                      uses the hallucination engine directly with mock scores.

        Returns:
            Full evaluation report dict.
        """
        all_y_true = []
        all_y_pred = []
        all_scores = []
        per_question_results = []

        print(f"\n{'='*60}")
        print(f"  Evaluation: {len(test_set)} questions, "
              f"{sum(len(t['claims']) for t in test_set)} total claims")
        print(f"{'='*60}\n")

        for q_idx, item in enumerate(test_set):
            question = item["question"]
            claims   = item["claims"]

            print(f"[{q_idx+1}/{len(test_set)}] {question}")

            q_y_true = []
            q_y_pred = []
            q_scores = []
            q_details = []

            if pipeline is not None:
                # Run through the actual pipeline
                result = pipeline.query(question)

                # Match pipeline verdicts to test claims by similarity
                for claim in claims:
                    gt_label = 1 if claim["hallucinated"] else 0
                    q_y_true.append(gt_label)

                    # Find best matching verdict
                    best_match = self._match_claim_to_verdict(
                        claim["text"], result.verdicts
                    )

                    if best_match:
                        pred_label = 1 if best_match.classification == "HALLUCINATED" else 0
                        support_score = best_match.support_score
                    else:
                        pred_label = 0
                        support_score = 0.5

                    q_y_pred.append(pred_label)
                    q_scores.append(1.0 - support_score)  # Invert: higher = more hallucinated

                    q_details.append({
                        "claim":      claim["text"],
                        "gt":         "HALLUCINATED" if gt_label else "SUPPORTED",
                        "pred":       "HALLUCINATED" if pred_label else "SUPPORTED",
                        "score":      round(support_score, 4),
                        "correct":    gt_label == pred_label,
                    })
            else:
                # Direct evaluation without pipeline (using test labels)
                for claim in claims:
                    gt_label = 1 if claim["hallucinated"] else 0
                    q_y_true.append(gt_label)

                    # Simulate a reasonable prediction for demo purposes
                    # Real evaluation would use pipeline.query() above
                    if claim["hallucinated"]:
                        pred_label = 1
                        support_score = 0.25  # Low support for hallucinated claims
                    else:
                        pred_label = 0
                        support_score = 0.82  # High support for true claims

                    q_y_pred.append(pred_label)
                    q_scores.append(1.0 - support_score)

                    q_details.append({
                        "claim":   claim["text"],
                        "gt":      "HALLUCINATED" if gt_label else "SUPPORTED",
                        "pred":    "HALLUCINATED" if pred_label else "SUPPORTED",
                        "score":   round(support_score, 4),
                        "correct": gt_label == pred_label,
                    })

            all_y_true.extend(q_y_true)
            all_y_pred.extend(q_y_pred)
            all_scores.extend(q_scores)

            q_metrics = precision_recall_f1(q_y_true, q_y_pred)
            per_question_results.append({
                "question": question,
                "num_claims": len(claims),
                "accuracy": q_metrics["accuracy"],
                "details": q_details,
            })

            correct = sum(1 for d in q_details if d["correct"])
            print(f"   → {correct}/{len(claims)} correct\n")

        return self._finalize_report(
            all_y_true, all_y_pred, all_scores, per_question_results,
            mode="full_pipeline" if pipeline is not None else "mock",
        )

    def evaluate_detection_layer(
        self,
        test_set: List[Dict[str, Any]],
        pipeline,
    ) -> Dict[str, Any]:
        """
        Evaluate ONLY the detection layer (real retrieval + real NLI +
        real BERTScore + cosine aggregation) directly against each
        ground-truth claim — this is the recommended, scientifically
        honest way to reproduce the paper's F1 / AUROC numbers.

        Why this instead of evaluate_with_pipeline(..., pipeline=pipeline):
        that method generates a fresh LLM answer for each question and then
        fuzzy-matches generated sentences back to the test set's claims by
        word overlap. But this test set's "hallucinated" claims are
        deliberately absurd statements (e.g. "FAISS requires a PostgreSQL
        database") that a real LLM will simply never generate — so there is
        often no generated sentence to match them to at all, and the match
        falls back to a meaningless default score. That is not a fair test
        of the detection layer. This method instead feeds every claim
        straight into the exact same NLI + BERTScore + cosine pipeline your
        live system uses to verify LLM output, scored against the real
        chunks retrieved for that question. No text is generated, so this
        also works even if Ollama is not running — it only needs the
        embedding, NLI, and BERTScore models.

        Args:
            test_set: List of test items (question + claims with labels).
            pipeline: A RAGHallucinationPipeline instance. Must have an
                      index already built (call pipeline.index_documents()
                      first).

        Returns:
            Full evaluation report dict (same shape as evaluate_with_pipeline).
        """
        from src.explainability.xai_layer import generate_xai_report

        pipeline._load_models()
        if not pipeline.faiss_store.is_loaded() and not pipeline.faiss_store.load():
            raise RuntimeError(
                "No FAISS index found. Call pipeline.index_documents() before "
                "evaluate_detection_layer()."
            )

        all_y_true, all_y_pred, all_scores = [], [], []
        per_question_results = []

        print(f"\n{'='*60}")
        print(f"  Detection-layer evaluation: {len(test_set)} questions, "
              f"{sum(len(t['claims']) for t in test_set)} total claims")
        print(f"  (real retrieval + real NLI + real BERTScore — no LLM generation)")
        print(f"{'='*60}\n")

        for q_idx, item in enumerate(test_set):
            question = item["question"]
            claims = item["claims"]
            claim_texts = [c["text"] for c in claims]

            print(f"[{q_idx+1}/{len(test_set)}] {question}")

            query_vec = pipeline.embedder.embed_query(question)
            retrieved = pipeline.faiss_store.search(query_vec, k=pipeline.top_k)

            q_y_true, q_y_pred, q_scores, q_details = [], [], [], []

            if not retrieved:
                # No evidence retrieved at all → every claim must be judged
                # HALLUCINATED, same as the live system's "insufficient
                # evidence" path.
                for claim in claims:
                    gt_label = 1 if claim["hallucinated"] else 0
                    q_y_true.append(gt_label)
                    q_y_pred.append(1)
                    q_scores.append(1.0)
                    q_details.append({
                        "claim": claim["text"], "gt": "HALLUCINATED" if gt_label else "SUPPORTED",
                        "pred": "HALLUCINATED", "score": 0.0,
                        "correct": gt_label == 1,
                    })
            else:
                chunk_texts = [c["text"] for c in retrieved]
                chunk_vecs = pipeline.embedder.embed(chunk_texts)

                xai_report = generate_xai_report(
                    query=question, chunks=retrieved, query_vector=query_vec,
                    chunk_vectors=chunk_vecs, sentences=claim_texts,
                    embedder=pipeline.embedder,
                )
                attributions = xai_report["attributions"]

                nli_results = pipeline.nli_detector.score_sentences(claim_texts, retrieved, attributions)
                bert_results = pipeline.bert_detector.score_sentences(claim_texts, retrieved)
                verdicts = pipeline.hall_engine.analyze(claim_texts, nli_results, bert_results, attributions)

                for claim, v in zip(claims, verdicts):
                    gt_label = 1 if claim["hallucinated"] else 0
                    pred_label = 1 if v.classification == "HALLUCINATED" else 0
                    q_y_true.append(gt_label)
                    q_y_pred.append(pred_label)
                    q_scores.append(1.0 - v.support_score)
                    q_details.append({
                        "claim": claim["text"],
                        "gt": "HALLUCINATED" if gt_label else "SUPPORTED",
                        "pred": "HALLUCINATED" if pred_label else "SUPPORTED",
                        "score": round(v.support_score, 4),
                        "correct": gt_label == pred_label,
                    })

            all_y_true.extend(q_y_true)
            all_y_pred.extend(q_y_pred)
            all_scores.extend(q_scores)

            q_metrics = precision_recall_f1(q_y_true, q_y_pred)
            per_question_results.append({
                "question": question,
                "num_claims": len(claims),
                "accuracy": q_metrics["accuracy"],
                "details": q_details,
            })

            correct = sum(1 for d in q_details if d["correct"])
            print(f"   → {correct}/{len(claims)} correct\n")

        return self._finalize_report(
            all_y_true, all_y_pred, all_scores, per_question_results,
            mode="detection_layer",
        )

    def _finalize_report(
        self,
        all_y_true: list,
        all_y_pred: list,
        all_scores: list,
        per_question_results: list,
        mode: str,
    ) -> Dict[str, Any]:
        """Aggregate metrics, print summary, generate plots, save JSON report."""
        overall = precision_recall_f1(all_y_true, all_y_pred)
        auc_roc = auroc(all_y_true, all_scores)
        cm = compute_confusion_matrix(all_y_true, all_y_pred)

        report = {
            "mode": mode,
            "overall_metrics": {
                "precision":  overall["precision"],
                "recall":     overall["recall"],
                "f1":         overall["f1"],
                "accuracy":   overall["accuracy"],
                "auroc":      auc_roc,
                "tp":         overall["tp"],
                "fp":         overall["fp"],
                "fn":         overall["fn"],
                "tn":         overall["tn"],
                "total":      overall["support"],
            },
            "confusion_matrix": cm,
            "per_question": per_question_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # ── Print summary ─────────────────────────────────────────────────
        print("=" * 60)
        print(f"  EVALUATION RESULTS  (mode={mode})")
        print("=" * 60)
        if mode == "mock":
            print("  WARNING: mock mode — these scores are HARDCODED (0.82 / 0.25),")
            print("  not produced by any real model. Do not report these numbers.")
        print(f"  Total claims evaluated: {overall['support']}")
        print(f"  Precision:  {overall['precision']:.4f}")
        print(f"  Recall:     {overall['recall']:.4f}")
        print(f"  F1-Score:   {overall['f1']:.4f}")
        print(f"  Accuracy:   {overall['accuracy']:.4f}")
        print(f"  AUC-ROC:    {auc_roc:.4f}")
        print(f"  TP={overall['tp']}  FP={overall['fp']}  "
              f"FN={overall['fn']}  TN={overall['tn']}")
        print("=" * 60)

        # ── Generate plots ────────────────────────────────────────────────
        try:
            plot_roc_curve(
                all_y_true, all_scores,
                str(self.output_dir / "roc_curve.png"),
            )
            plot_support_score_distribution(
                [1 - s for s in all_scores],  # Convert back to support scores
                all_y_true,
                str(self.output_dir / "score_distribution.png"),
            )
            plot_metrics_bar(
                {
                    "Precision": overall["precision"],
                    "Recall":    overall["recall"],
                    "F1":        overall["f1"],
                    "Accuracy":  overall["accuracy"],
                    "AUROC":     auc_roc,
                },
                title="Hallucination Detection — Overall Metrics",
                save_path=str(self.output_dir / "metrics_bar.png"),
            )
        except Exception as e:
            print(f"[WARN] Plotting failed: {e}")

        # ── Save JSON report ──────────────────────────────────────────────
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Full report saved: {report_path}")

        return report

    def _match_claim_to_verdict(self, claim_text: str, verdicts: list):
        """
        Match a test claim to the closest pipeline verdict by text overlap.
        """
        if not verdicts:
            return None

        claim_words = set(claim_text.lower().split())
        best_v = None
        best_overlap = 0.0

        for v in verdicts:
            v_words = set(v.sentence.lower().split())
            if not v_words:
                continue
            overlap = len(claim_words & v_words) / max(len(claim_words | v_words), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_v = v

        return best_v


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate the TrustRAG hallucination-detection pipeline against a labeled test set."
    )
    parser.add_argument(
        "test_set", nargs="?", default=None,
        help="Path to a custom test set JSON file (see module docstring for format). "
             "Omit to use the small built-in demo test set.",
    )
    parser.add_argument(
        "--mode", choices=["detection", "full", "mock"], default="detection",
        help=(
            "detection (default): real retrieval + real NLI + real BERTScore + cosine, "
            "scored directly against each ground-truth claim. No LLM/Ollama needed. "
            "This is the recommended mode and the one whose numbers you should report. "
            "full: real end-to-end generation via Ollama, then fuzzy-matches claims to "
            "generated sentences — slower and noisier, requires Ollama running. "
            "mock: hardcoded 0.82/0.25 scores, NO real model is run at all — for smoke-"
            "testing the reporting/plotting code only. Never report these numbers."
        ),
    )
    args = parser.parse_args()

    evaluator = Evaluator()

    if args.test_set:
        with open(args.test_set, "r", encoding="utf-8") as f:
            test_set = json.load(f)
        print(f"Loaded test set from: {args.test_set}")
    else:
        test_set = DEMO_TEST_SET
        print("Using built-in demo test set.")

    if args.mode == "mock":
        print("\n" + "!" * 60)
        print("  MOCK MODE — scores below are HARDCODED (0.82 / 0.25).")
        print("  No real model is being run. This exists only to smoke-test")
        print("  the reporting/plotting code. Do not cite these numbers.")
        print("!" * 60 + "\n")
        report = evaluator.evaluate_with_pipeline(test_set, pipeline=None)
    else:
        from src.pipeline import RAGHallucinationPipeline

        pipeline = RAGHallucinationPipeline()
        print("\n[INFO] Indexing corpus (reuses the existing index if one is already built)...")
        stats = pipeline.index_documents()
        print(f"[INFO] Index stats: {stats}\n")

        if args.mode == "full":
            report = evaluator.evaluate_with_pipeline(test_set, pipeline=pipeline)
        else:
            report = evaluator.evaluate_detection_layer(test_set, pipeline)
