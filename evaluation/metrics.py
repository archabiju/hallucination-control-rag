"""
Evaluation Metrics — Evaluation Layer
========================================
Standard metrics for evaluating:
  1. Hallucination Detection Quality   (Precision, Recall, F1, AUROC)
  2. Retrieval Quality                 (Recall@K, MRR, NDCG)
  3. Answer Quality                    (BERTScore, Provenance score)

All computations are offline using scikit-learn and numpy.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for offline use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hallucination Detection Metrics
# ─────────────────────────────────────────────────────────────────────────────

def precision_recall_f1(
    y_true: List[int],
    y_pred: List[int],
    positive_label: int = 1,
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 for binary hallucination classification.

    Args:
        y_true: Ground-truth labels. 1 = hallucinated, 0 = not hallucinated.
        y_pred: Predicted labels from the pipeline.
        positive_label: Which label counts as "positive" (default: 1 = hallucinated).

    Returns:
        Dict with precision, recall, f1, accuracy, support.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p == positive_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p != positive_label)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p != positive_label)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "support": len(y_true),
    }


def auroc(
    y_true:  List[int],
    y_scores: List[float],
) -> float:
    """
    Compute Area Under the ROC Curve (AUC-ROC).

    Args:
        y_true: Ground-truth binary labels.
        y_scores: Confidence scores from the hallucination engine
                  (higher score = more likely hallucinated).

    Returns:
        AUC-ROC value in [0, 1].
    """
    try:
        from sklearn.metrics import roc_auc_score
        return round(float(roc_auc_score(y_true, y_scores)), 4)
    except Exception:
        return 0.0


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str] = None,
) -> Dict[str, Any]:
    """Compute confusion matrix as a dict."""
    try:
        from sklearn.metrics import confusion_matrix
        labels = labels or [0, 1]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return {
            "matrix": cm.tolist(),
            "labels": labels,
        }
    except Exception:
        return {"matrix": [], "labels": []}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Retrieval Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(
    relevant_ids: List[str],
    retrieved_ids: List[str],
    k: int = 5,
) -> float:
    """
    Recall@K: fraction of relevant docs found in the top-K retrieved.

    Args:
        relevant_ids: Ground-truth relevant chunk/doc IDs.
        retrieved_ids: IDs retrieved by FAISS (in rank order).
        k: Cutoff rank.

    Returns:
        Float in [0, 1].
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits  = len(set(relevant_ids) & top_k)
    return round(hits / len(relevant_ids), 4)


def mrr(
    relevant_ids: List[str],
    retrieved_ids: List[str],
) -> float:
    """
    Mean Reciprocal Rank: 1/rank of the first relevant result.

    Args:
        relevant_ids: Ground-truth relevant IDs.
        retrieved_ids: Retrieved IDs in rank order.

    Returns:
        Float in [0, 1].
    """
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return round(1.0 / rank, 4)
    return 0.0


def ndcg_at_k(
    relevant_ids: List[str],
    retrieved_ids: List[str],
    k: int = 5,
) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    Args:
        relevant_ids: Ground-truth relevant IDs.
        retrieved_ids: Retrieved IDs in rank order.
        k: Cutoff rank.

    Returns:
        NDCG@K float in [0, 1].
    """
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    # DCG
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, doc_id in enumerate(top_k)
        if doc_id in relevant_set
    )

    # Ideal DCG
    ideal = sum(
        1.0 / np.log2(rank + 2)
        for rank in range(min(len(relevant_ids), k))
    )

    return round(dcg / ideal if ideal > 0 else 0.0, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true:   List[int],
    y_scores: List[float],
    save_path: Optional[str] = None,
) -> str:
    """
    Plot and save the ROC curve.

    Returns:
        Path to saved plot image.
    """
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#6366F1", lw=2,
                label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1, linestyle="--",
                label="Random classifier")
        ax.fill_between(fpr, tpr, alpha=0.08, color="#6366F1")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("Hallucination Detection — ROC Curve", fontsize=13)
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        save_path = save_path or "evaluation/results/roc_curve.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] ROC curve saved: {save_path}")
        return save_path
    except Exception as e:
        print(f"[WARN] Could not plot ROC: {e}")
        return ""


def plot_support_score_distribution(
    support_scores: List[float],
    labels:         List[int],
    save_path:      Optional[str] = None,
) -> str:
    """
    Plot histogram of support scores, coloured by ground-truth label.
    """
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        scores_hal   = [s for s, l in zip(support_scores, labels) if l == 1]
        scores_nohal = [s for s, l in zip(support_scores, labels) if l == 0]

        ax.hist(scores_nohal, bins=20, alpha=0.6, color="#22C55E",
                label="Not Hallucinated", range=(0, 1))
        ax.hist(scores_hal,   bins=20, alpha=0.6, color="#EF4444",
                label="Hallucinated",     range=(0, 1))

        ax.axvline(x=0.75, color="#6366F1", linestyle="--", lw=1.5, label="High threshold")
        ax.axvline(x=0.50, color="#F59E0B", linestyle="--", lw=1.5, label="Medium threshold")

        ax.set_xlabel("Support Score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Support Score Distribution by Label", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = save_path or "evaluation/results/score_distribution.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Score distribution plot saved: {save_path}")
        return save_path
    except Exception as e:
        print(f"[WARN] Could not plot distribution: {e}")
        return ""


def plot_metrics_bar(
    metrics: Dict[str, float],
    title:   str = "Evaluation Metrics",
    save_path: Optional[str] = None,
) -> str:
    """Plot a bar chart of evaluation metrics."""
    try:
        names  = list(metrics.keys())
        values = list(metrics.values())
        colors = ["#6366F1", "#22C55E", "#F59E0B", "#EF4444",
                  "#06B6D4", "#8B5CF6", "#EC4899"]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(names, values, color=colors[:len(names)], alpha=0.85,
                      edgecolor="white", linewidth=1.2)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", alpha=0.3)

        save_path = save_path or "evaluation/results/metrics_bar.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Metrics bar chart saved: {save_path}")
        return save_path
    except Exception as e:
        print(f"[WARN] Could not plot metrics: {e}")
        return ""
