# TrustRAG Final Review — Corrected Evidence

## Outcome

The main Review 2 problem was an implementation defect, not the NLI weight.
`cross-encoder/nli-deberta-v3-small` publishes:

- contradiction = 0
- entailment = 1
- neutral = 2

The old detector treated class 2 as entailment. It therefore used neutral
probability as NLI support. The detector now reads `id2label` from the
checkpoint, and a regression test protects both common NLI label orders.

Weights do not determine an individual component's AUROC. AUROC measures the
ranking produced by that component. After correcting the label mapping, NLI
AUROC is higher than BERTScore on the project benchmark, as expected from the
actual measured scores.

## Corrected project benchmark

Data: 37 questions, 109 author-labelled claims, 213 indexed chunks.

### Existing NLI-small, four evidence windows

| Configuration | Precision | Recall | F1 | Accuracy | AUROC |
|---|---:|---:|---:|---:|---:|
| NLI only | 0.805 | 0.846 | 0.825 | 0.872 | **0.935** |
| BERTScore only | 0.000 | 0.000 | 0.000 | 0.642 | 0.797 |
| Cosine only | 0.379 | 0.564 | 0.454 | 0.514 | 0.573 |
| NLI + BERTScore | 0.790 | 0.769 | 0.779 | 0.844 | 0.913 |
| Three-signal fusion | 0.795 | 0.795 | 0.795 | 0.853 | 0.897 |

### Eight evidence windows

Eight candidate evidence windows slightly improve the three-signal result:
precision 0.816, recall 0.795, F1 0.805, accuracy 0.862, and AUROC 0.900.

### Larger NLI model, eight evidence windows

`cross-encoder/nli-deberta-v3-base` gives the strongest project-benchmark
result:

| Configuration | Precision | Recall | F1 | Accuracy | AUROC |
|---|---:|---:|---:|---:|---:|
| NLI only | 0.804 | 0.949 | 0.871 | 0.899 | **0.948** |
| NLI + BERTScore | 0.800 | 0.923 | 0.857 | 0.890 | 0.901 |
| Three-signal fusion | 0.804 | 0.949 | 0.871 | 0.899 | 0.908 |

At a threshold chosen on these same 109 claims, accuracy can reach 0.908.
That number is a development result, not an unbiased final estimate, and
precision remains approximately 0.81. It must not be described as more than
90% precision.

## Larger independent benchmark

`evaluation/build_ragtruth_benchmark.py` converts official RAGTruth
hallucination-span annotations into sentence labels without inventing labels.
It creates:

- calibration: 800 responses, 5,911 claims, 605 hallucinated;
- held-out test: 400 responses, 3,031 claims, 232 hallucinated;
- approximately equal QA, summarization, and data-to-text response counts;
- no response or source overlap between calibration and test.

The NLI-major calibration constraint is:

`w_nli >= w_bertscore >= w_cosine >= 0`, with weights summing to one.

Calibration selected NLI = 0.60, BERTScore = 0.40, cosine = 0.00 and a support
threshold of 0.32. On the untouched RAGTruth test set:

| Metric | Estimate | Response-bootstrap 95% CI |
|---|---:|---:|
| Precision | 0.155 | 0.128–0.183 |
| Recall | 0.599 | 0.527–0.670 |
| F1 | 0.246 | 0.208–0.284 |
| Accuracy | 0.719 | 0.693–0.743 |
| Balanced accuracy | 0.664 | 0.628–0.699 |
| AUROC | 0.705 | 0.667–0.741 |

Raw BERTScore accuracy at threshold 0.5 is 0.924 only because 92.3% of test
sentences are supported and it predicts no hallucinations. Its hallucination
precision, recall, and F1 are all zero. This is why a “more than 90% correct”
claim based only on accuracy would be misleading.

## What can be claimed

1. Correcting the checkpoint label mapping raises project-benchmark NLI AUROC
   from the invalid 0.571 result to 0.935.
2. Evidence windows are much better than whole 512-token chunks on the
   project benchmark.
3. A larger NLI checkpoint improves recall and project-benchmark F1, at higher
   latency.
4. The present zero-shot detector does not generalize at 90% precision or F1
   on RAGTruth. Further model work is required.
5. The system is customizable by rebuilding its local index from an
   organization's approved PDF, DOCX, TXT, and Markdown documents. It answers
   from retrieved organizational evidence, but adding documents does not
   guarantee correctness.

## Work required before a strong final paper claim

1. Fine-tune or calibrate a faithfulness classifier on RAGTruth train only,
   using class weighting and response-grouped validation.
2. Freeze the model and decision threshold before evaluating RAGTruth test.
3. Obtain a second annotator for the organization-specific benchmark and
   report agreement.
4. Evaluate retrieval separately with MS MARCO qrels. The checked-in
   `msmarco_subset.txt` contains 1,000 passages, not 2,000.
5. Run generated-answer evaluation only when Ollama and the named generation
   model are actually available.
6. Compare against cited baselines on the same test split. Novelty does not
   establish superiority; measured paired results do.

Raw files:

- `evaluation/results/review3_corrected_nli_report.json`
- `evaluation/results/review3_windows8_report.json`
- `evaluation/results/review3_nli_base_windows8_report.json`
- `evaluation/results/ragtruth/calibrated_fusion_report.json`
