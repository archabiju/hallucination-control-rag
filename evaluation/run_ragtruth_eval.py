"""Score the independently annotated RAGTruth benchmark.

This runner produces raw component scores only. Weight and threshold selection
must use the RAGTruth train split; the test split is held out for one final
evaluation. It intentionally reports class-balanced metrics because raw
accuracy is misleading on RAGTruth's imbalanced sentence labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import auroc, precision_recall_f1
from src.detection.bertscore_detector import BERTScoreDetector
from src.detection.nli_detector import NLIDetector
from src.ingestion.chunker import split_into_sentences
from src.retrieval.embedder import Embedder


def read_jsonl(path: Path, limit: int | None):
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            yield json.loads(line)


def evidence_windows(text: str, size: int) -> list[str]:
    sentences = [
        sentence
        for sentence in split_into_sentences(text)
        if len(sentence.split()) >= 3
    ]
    if not sentences:
        return [text]
    return [
        " ".join(sentences[i : i + size])
        for i in range(max(len(sentences) - size + 1, 1))
    ]


def evaluate_binary(y: np.ndarray, support: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import balanced_accuracy_score

    prediction = (support < threshold).astype(int)
    metrics = precision_recall_f1(y.tolist(), prediction.tolist())
    metrics["balanced_accuracy"] = round(
        float(balanced_accuracy_score(y, prediction)), 4
    )
    metrics["auroc"] = auroc(y.tolist(), (1.0 - support).tolist())
    return metrics


def score_record(
    record: dict,
    embedder: Embedder,
    nli: NLIDetector,
    bert: BERTScoreDetector,
    window_size: int,
    top_n: int,
) -> list[dict]:
    claims = [item["claim"] for item in record["claims"]]
    windows = evidence_windows(record["evidence"], window_size)
    claim_vectors = embedder.embed(claims)
    window_vectors = embedder.embed(windows)
    similarities = claim_vectors @ window_vectors.T
    selected = np.argsort(-similarities, axis=1)[:, : min(top_n, len(windows))]
    pair_claims = []
    pair_evidence = []
    pair_cosine = []
    for claim_index, window_ids in enumerate(selected):
        for window_id in window_ids:
            pair_claims.append(claims[claim_index])
            pair_evidence.append(windows[int(window_id)])
            pair_cosine.append(float(similarities[claim_index, int(window_id)]))

    pairs = list(zip(pair_evidence, pair_claims))
    probabilities = np.asarray(
        nli.model.predict(
            pairs,
            apply_softmax=True,
            batch_size=32,
            show_progress_bar=False,
        )
    )
    _, _, bert_f1 = bert._get_scorer().score(
        cands=pair_claims,
        refs=pair_evidence,
        batch_size=32,
    )
    width = selected.shape[1]
    probabilities = probabilities.reshape(len(claims), width, -1)
    bert_scores = bert_f1.detach().cpu().numpy().reshape(len(claims), width)
    cosine_scores = np.asarray(pair_cosine).reshape(len(claims), width)
    entail_index = nli.label_to_idx["ENTAILMENT"]
    neutral_index = nli.label_to_idx["NEUTRAL"]
    contradiction_index = nli.label_to_idx["CONTRADICTION"]

    rows = []
    for claim_index, claim in enumerate(record["claims"]):
        # Match existing TrustRAG behavior: use the candidate evidence window
        # with the greatest entailment probability.
        best = int(np.argmax(probabilities[claim_index, :, entail_index]))
        rows.append(
            {
                "response_id": record["response_id"],
                "source_id": record["source_id"],
                "task_type": record["task_type"],
                "source_dataset": record["source_dataset"],
                "model": record["model"],
                "claim": claim["claim"],
                "hallucinated": int(claim["hallucinated"]),
                "nli_entailment": float(
                    probabilities[claim_index, best, entail_index]
                ),
                "nli_neutral": float(
                    probabilities[claim_index, best, neutral_index]
                ),
                "nli_contradiction": float(
                    probabilities[claim_index, best, contradiction_index]
                ),
                "bertscore_f1": float(np.max(bert_scores[claim_index])),
                "cosine": float(np.max(cosine_scores[claim_index])),
                "best_evidence_window": windows[int(selected[claim_index, best])],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "evaluation" / "benchmarks" / "ragtruth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation" / "results" / "ragtruth")
    parser.add_argument("--max-responses", type=int)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument(
        "--nli-model",
        default="cross-encoder/nli-deberta-v3-small",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = list(
        read_jsonl(args.benchmark_dir / f"{args.split}.jsonl", args.max_responses)
    )
    print(f"Loading models for {len(records)} {args.split} responses")
    embedder = Embedder()
    nli = NLIDetector(args.nli_model)
    bert = BERTScoreDetector()
    if nli.model is None or not bert._bert_score_available:
        raise RuntimeError("RAGTruth evaluation requires the real NLI and BERTScore models")

    rows = []
    started = time.time()
    for index, record in enumerate(records, start=1):
        rows.extend(
            score_record(
                record,
                embedder,
                nli,
                bert,
                args.window_size,
                args.top_n,
            )
        )
        print(f"{index}/{len(records)} responses; {len(rows)} claims", flush=True)

    tag = args.split
    if args.max_responses is not None:
        tag += f"_first_{args.max_responses}"
    if args.window_size != 2 or args.top_n != 4:
        tag += f"_windows_{args.window_size}_top_{args.top_n}"
    if not args.nli_model.endswith("nli-deberta-v3-small"):
        tag += "_" + args.nli_model.rsplit("/", 1)[-1].replace("-", "_")
    output = args.output_dir / f"{tag}_per_claim.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    y = np.array([row["hallucinated"] for row in rows])
    signals = {
        "NLI entailment": np.array([row["nli_entailment"] for row in rows]),
        "BERTScore": np.array([row["bertscore_f1"] for row in rows]),
        "Cosine": np.array([row["cosine"] for row in rows]),
    }
    report = {
        "split": args.split,
        "responses": len(records),
        "claims": len(rows),
        "hallucinated_claims": int(y.sum()),
        "supported_claims": int((1 - y).sum()),
        "window_size": args.window_size,
        "top_n": args.top_n,
        "nli_model": args.nli_model,
        "seconds": round(time.time() - started, 1),
        "metrics_at_support_threshold_0.5": {
            name: evaluate_binary(y, score) for name, score in signals.items()
        },
    }
    report_path = args.output_dir / f"{tag}_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
