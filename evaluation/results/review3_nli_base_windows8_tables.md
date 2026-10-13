# Review 2 — measured results

37 questions / 109 labelled claims (39 hallucinated) · 213-chunk index · weights (0.45, 0.35, 0.2) · thresholds 0.75/0.5

## Overall (positive class = hallucinated)

| P | R | F1 | AUC-ROC | Acc |
|---|---|---|---|---|
| 0.8043 | 0.9487 | 0.8706 | 0.9077 | 0.8991 |

## Ablation

| Config | P | R | F1 @0.50 | AUC | best-F1 (thr) |
|---|---|---|---|---|---|
| NLI only | 0.8043 | 0.9487 | 0.8706 | 0.948 | 0.8837 (0.76) |
| BERTScore only | 0.0 | 0.0 | 0.0 | 0.7925 | 0.6582 (0.81) |
| Cosine only | 0.3793 | 0.5641 | 0.4536 | 0.5733 | 0.5455 (0.66) |
| NLI + BERTScore | 0.8 | 0.9231 | 0.8571 | 0.9007 | 0.8837 (0.79) |
| TrustRAG (all three) | 0.8043 | 0.9487 | 0.8706 | 0.9077 | 0.8837 (0.72) |

## Evidence windows vs whole-chunk premises (all three signals)

| Premise | P | R | F1 @0.50 | AUC | best-F1 (thr) |
|---|---|---|---|---|---|
| whole 512-token chunk | 0.3645 | 1.0 | 0.5342 | 0.6412 | 0.5588 (0.4) |
| 2-sentence windows (shipped) | 0.8043 | 0.9487 | 0.8706 | 0.9077 | 0.8837 (0.72) |

## Tri-state distribution (% of claims)

| Claim type | Supported | Weak | Hallucinated |
|---|---|---|---|
| Supported claims | 87.1 | 0.0 | 12.9 |
| Hallucinated claims | 2.6 | 2.6 | 94.9 |

## Retrieval (doc-level, k=5)

Recall@5 0.8514 · MRR 0.8108 · NDCG@5 0.8053 · Hit@5 0.8919