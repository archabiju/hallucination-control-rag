# Review 2 — measured results

37 questions / 109 labelled claims (39 hallucinated) · 213-chunk index · weights (0.45, 0.35, 0.2) · thresholds 0.75/0.5

## Overall (positive class = hallucinated)

| P | R | F1 | AUC-ROC | Acc |
|---|---|---|---|---|
| 0.6667 | 0.2051 | 0.3137 | 0.7007 | 0.6789 |

## Ablation

| Config | P | R | F1 @0.50 | AUC | best-F1 (thr) |
|---|---|---|---|---|---|
| NLI only | 0.6429 | 0.2308 | 0.3396 | 0.5712 | 0.4384 (0.95) |
| BERTScore only | 0.0 | 0.0 | 0.0 | 0.7969 | 0.675 (0.81) |
| Cosine only | 0.3793 | 0.5641 | 0.4536 | 0.5733 | 0.5455 (0.66) |
| NLI + BERTScore | 0.5833 | 0.1795 | 0.2745 | 0.7381 | 0.6535 (0.91) |
| TrustRAG (all three) | 0.6667 | 0.2051 | 0.3137 | 0.7007 | 0.5957 (0.81) |

## Evidence windows vs whole-chunk premises (all three signals)

| Premise | P | R | F1 @0.50 | AUC | best-F1 (thr) |
|---|---|---|---|---|---|
| whole 512-token chunk | 0.48 | 0.3077 | 0.375 | 0.6092 | 0.5414 (0.82) |
| 2-sentence windows (shipped) | 0.6667 | 0.2051 | 0.3137 | 0.7007 | 0.5957 (0.81) |

## Tri-state distribution (% of claims)

| Claim type | Supported | Weak | Hallucinated |
|---|---|---|---|
| Supported claims | 87.1 | 7.1 | 5.7 |
| Hallucinated claims | 69.2 | 10.3 | 20.5 |

## Retrieval (doc-level, k=5)

Recall@5 0.8514 · MRR 0.8108 · NDCG@5 0.8053 · Hit@5 0.8919