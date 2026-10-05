# Review 2 Runbook — TrustRAG

## Run the live demo
```bash
source .venv/bin/activate
streamlit run ui/app.py          # sidebar: click "Build Document Index" once (213 chunks)
```
Ollama is **not** installed on this Mac. Without it the generator uses a clearly-labelled fallback
(returns the retrieved passage), so retrieval, XAI, detection and the tri-state controller are all real
but the answer text is not LLM-written. For a full demo: install Ollama, `ollama pull mistral`
(paper uses Llama 3 8B: set `generation.model_name` in `config/config.yaml`).

## Reproduce the numbers (no Ollama needed, ~3 min)
```bash
python evaluation/run_review2_eval.py    # writes evaluation/results/review2_report.json + review2_tables.md
```
Inputs: `evaluation/test_set.json` (37 questions, 109 hand-labelled claims; rebuild with `evaluation/build_test_set.py`).

## Measured results (shipped config: DeBERTa-v3-small + 2-sentence evidence windows)
| | P | R | F1@0.50 | AUC | best-F1 |
|---|---|---|---|---|---|
| NLI only | .64 | .23 | .34 | .57 | .44 |
| BERTScore only | 0 | 0 | 0 | **.80** | .68 |
| Cosine only | .38 | .56 | .45 | .57 | .55 |
| NLI + BERTScore | .58 | .18 | .27 | .74 | .65 |
| **TrustRAG (all 3)** | .67 | .21 | .31 | .70 | .60 |
| TrustRAG, whole-chunk premises | .48 | .31 | .38 | .61 | .54 |

Tri-state: supported claims → 87% Supported / 7% Weak / 6% Hallucinated; hallucinated claims → 69% / 10% / 21%.
Retrieval (doc-level, k=5, with 2,000 MS MARCO distractors): Recall@5 .85, MRR .81, NDCG@5 .81.

## How this relates to the IEEE paper — be ready for this question
The paper reports F1 0.898 / AUC 0.951 on 250 documents / 9,800 chunks / 400 queries, with Llama 3 8B.
**This repo has not reproduced those figures**: the corpus here is 22 documents / 213 chunks and the benchmark
is 109 claims. The numbers above are what the code actually produces. Before submission, either run the
paper's benchmark through this pipeline and update the paper to the measured values, or describe the paper's
figures only if the experiment behind them exists. Do not present paper numbers as outputs of this demo.

Known differences between paper and code: chunk overlap 50 (paper: 64); S_COS uses the best-attributed chunk
(paper Eq. 21–22: mean of retrieved embeddings); weights 0.45/0.35/0.20 (paper leaves w1–w3 unspecified);
NLI/BERTScore verify against evidence windows (not in paper); generator default Mistral (paper: Llama 3 8B).

## Likely panel questions
- *Why is recall low at 0.75/0.50?* Hallucinated claims in the benchmark are near-misses of true statements
  (changed number, negation), so topical similarity stays high. Thresholds are untuned; AUC (0.70) shows the
  score ranks them correctly more often than not. Calibration/adaptive weights = future work (also in the paper).
- *Why does BERTScore alone beat the combination?* NLI-small is weak on these near-misses (AUC .57) and drags
  the weighted sum down; NLI+BERTScore (.74) > full (.70). DeBERTa-v3-base lifts NLI to .64 at ~3x latency.
- *What did evidence windows change?* AUC .61 -> .70; supported-claim pass rate 56% -> 87%.
