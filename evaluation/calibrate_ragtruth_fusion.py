"""Calibrate TrustRAG fusion on RAGTruth train and evaluate once on test.

The search is deliberately constrained to NLI-major non-negative weights:
    w_nli >= w_bert >= w_cos, sum(weights) = 1

The held-out test labels never participate in weight or threshold selection.
Confidence intervals use a response-level bootstrap, preserving correlations
among sentences from the same generated response.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def arrays(rows: list[dict]):
    y = np.array([row["hallucinated"] for row in rows], dtype=int)
    x = np.array(
        [
            [row["nli_entailment"], row["bertscore_f1"], row["cosine"]]
            for row in rows
        ],
        dtype=float,
    )
    groups = np.array([row["response_id"] for row in rows])
    return x, y, groups


def metrics(y: np.ndarray, support: np.ndarray, threshold: float) -> dict:
    prediction = (support < threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "auroc": float(roc_auc_score(y, 1.0 - support)),
    }


def candidates(step: float):
    count = round(1.0 / step)
    for ni in range(count + 1):
        for bi in range(count - ni + 1):
            ci = count - ni - bi
            wn, wb, wc = ni / count, bi / count, ci / count
            if wn >= wb >= wc:
                yield np.array([wn, wb, wc])


def calibrate(x: np.ndarray, y: np.ndarray, step: float) -> tuple[np.ndarray, float, dict]:
    best = None
    for weight in candidates(step):
        support = x @ weight
        for threshold in np.arange(0.05, 0.951, 0.01):
            result = metrics(y, support, float(threshold))
            # Hallucination F1 is primary. Balanced accuracy prevents a
            # majority-class solution from winning ties.
            key = (result["f1"], result["balanced_accuracy"], result["precision"])
            if best is None or key > best[0]:
                best = (key, weight.copy(), float(threshold), result)
    assert best is not None
    return best[1], best[2], best[3]


def bootstrap(
    y: np.ndarray,
    support: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    values = []
    for _ in range(samples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[group] for group in drawn])
        # A bootstrap draw can exceptionally contain one class only.
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(metrics(y[idx], support[idx], threshold))
    return {
        name: {
            "mean": float(np.mean([row[name] for row in values])),
            "ci95_low": float(np.percentile([row[name] for row in values], 2.5)),
            "ci95_high": float(np.percentile([row[name] for row in values], 97.5)),
        }
        for name in values[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    default = ROOT / "evaluation" / "results" / "ragtruth"
    parser.add_argument("--train", type=Path, default=default / "train_per_claim.jsonl")
    parser.add_argument("--test", type=Path, default=default / "test_per_claim.jsonl")
    parser.add_argument("--output", type=Path, default=default / "calibrated_fusion_report.json")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()

    train_rows = load_rows(args.train)
    test_rows = load_rows(args.test)
    train_x, train_y, _ = arrays(train_rows)
    test_x, test_y, test_groups = arrays(test_rows)
    weight, threshold, train_metrics = calibrate(
        train_x, train_y, args.weight_step
    )
    test_support = test_x @ weight
    test_metrics = metrics(test_y, test_support, threshold)
    report = {
        "selection_data": display_path(args.train),
        "held_out_test_data": display_path(args.test),
        "constraint": "w_nli >= w_bertscore >= w_cosine >= 0; weights sum to 1",
        "weights": {
            "nli": float(weight[0]),
            "bertscore": float(weight[1]),
            "cosine": float(weight[2]),
        },
        "support_threshold": threshold,
        "train_metrics": train_metrics,
        "held_out_test_metrics": test_metrics,
        "held_out_response_bootstrap": bootstrap(
            test_y,
            test_support,
            test_groups,
            threshold,
            args.bootstrap_samples,
            args.seed,
        ),
        "warning": (
            "Do not change weights or threshold after reading held-out test "
            "results. Raw accuracy is secondary because supported sentences "
            "are the majority class."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
