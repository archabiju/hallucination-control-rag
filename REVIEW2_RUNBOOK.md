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

## Corrected measured results (DeBERTa-v3-small + 2-sentence evidence windows)

The original Review 2 run used the wrong checkpoint label order: it treated
class 2 (neutral) as entailment. The checkpoint defines contradiction=0,
entailment=1, neutral=2. After fixing the detector to read `id2label` from the
checkpoint, the same 109 claims produce:

| | P | R | F1@0.50 | AUC | best-F1 |
|---|---|---|---|---|---|
| NLI only | .80 | .85 | .83 | **.94** | .85 |
| BERTScore only | 0 | 0 | 0 | **.80** | .68 |
| Cosine only | .38 | .56 | .45 | .57 | .55 |
| NLI + BERTScore | .79 | .77 | .78 | .91 | .85 |
| **TrustRAG (all 3)** | .79 | .79 | .79 | .90 | .84 |
| TrustRAG, whole-chunk premises | .38 | .95 | .54 | .68 | .58 |

Tri-state: supported claims → 86% Supported / 3% Weak / 11% Hallucinated; hallucinated claims → 13% / 8% / 80%.
Retrieval (doc-level, k=5): Recall@5 .85, MRR .81, NDCG@5 .81.

The checked-in MS MARCO file currently contains 1,000 passages, not 2,000.
The corrected raw output is `evaluation/results/review3_corrected_nli_report.json`.

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
- *Why did NLI improve?* The detector now reads the model's semantic label mapping. The old run accidentally
  used neutral probability as entailment. This was an implementation defect, not a weight-tuning improvement.
- *Why is the combined AUC below NLI alone?* Weights do not determine component AUROCs. On these 109
  claims NLI ranks the labels better than BERTScore and cosine, so adding weaker signals lowers AUC.
- *Can we claim more than 90% precision?* No. Corrected NLI-only precision is .80 and combined precision is .79
  on this small benchmark. The larger RAGTruth held-out evaluation must be completed before final claims.
