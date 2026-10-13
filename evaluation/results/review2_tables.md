# Review 2 — measured results

37 questions / 109 labelled claims (39 hallucinated) · 213-chunk index · weights (0.45, 0.35, 0.2) · thresholds 0.75/0.5

## Overall (positive class = hallucinated)

| P | R | F1 | AUC-ROC | Acc |
|---|---|---|---|---|
| 0.4314 | 0.5641 | 0.4889 | 0.6414 | 0.578 |

## Ablation

| Config | P | R | F1 @0.50 | AUC | best-F1 (thr) |
|---|---|---|---|---|---|
| NLI only | 0.4074 | 0.5641 | 0.4731 | 0.6035 | 0.5063 (0.06) |
| BERTScore only | 0.0 | 0.0 | 0.0 | 0.7866 | 0.6667 (0.81) |
| Cosine only | 0.3793 | 0.5641 | 0.4536 | 0.5641 | 0.5455 (0.66) |
| NLI + BERTScore | 0.42 | 0.5385 | 0.4719 | 0.652 | 0.5735 (0.92) |
| TrustRAG (all three) | 0.4314 | 0.5641 | 0.4889 | 0.6414 | 0.5574 (0.8) |

## Evidence windows vs whole-chunk premises (all three signals)

| Premise | P | R | F1 @0.50 | AUC | best-F1 (thr) |
|---|---|---|---|---|---|
| whole 512-token chunk | 0.5 | 0.3077 | 0.381 | 0.5991 | 0.5414 (0.81) |
| 2-sentence windows (shipped) | 0.4314 | 0.5641 | 0.4889 | 0.6414 | 0.5574 (0.8) |

## Tri-state distribution (% of claims)

| Claim type | Supported | Weak | Hallucinated |
|---|---|---|---|
| Supported claims | 45.7 | 12.9 | 41.4 |
| Hallucinated claims | 30.8 | 12.8 | 56.4 |

## Retrieval (doc-level, k=5)

Recall@5 0.8514 · MRR 0.8108 · NDCG@5 0.8053 · Hit@5 0.8919