# TrustRAG Demo Guide

A run-based demo: one command regenerates the version record, sample outputs and evaluation metrics.

## 1. Run it

macOS / Linux:

```bash
source .venv/bin/activate
./run_demo.sh
```

Windows (PowerShell or CMD):

```bat
.venv\Scripts\activate
run_demo.bat
```

Runs in about 3 to 5 minutes. Ollama is not required: without it, the generator returns the retrieved passage (clearly labelled fallback), and retrieval, XAI, detection and the tri-state controller still run for real.

| Step | Command run by the script | Output |
|---|---|---|
| 1 | Version and config record | `demo/outputs/versions.txt` |
| 2 | `python src/pipeline.py` (build or reuse FAISS index) | `data/index/` |
| 3 | `python demo/demo_run.py` | `demo/outputs/sample_outputs.txt` and `.json` |
| 4 | `python evaluation/run_review2_eval.py --output-prefix demo_rerun` | `evaluation/results/demo_rerun_report.json`, `demo_rerun_tables.md`, plots |

Other entry points:

```bash
streamlit run ui/app.py                                # interactive dashboard
python evaluation/build_ragtruth_benchmark.py          # external RAGTruth benchmark (see README)
python evaluation/run_trustrag_opt_variability.py      # five-run retrieval index comparison
```

## 2. What the demo shows

1. **Live query.** The full pipeline retrieves the top-k chunks, generates an answer, attributes each sentence to evidence, scores it with NLI, BERTScore and cosine similarity, and applies the tri-state controller.
2. **Tri-state demonstration.** The same verification stack scores three answers for each sample question: grounded only (Answer with citations), grounded plus fabricated (Partial answer with warnings), and fabricated only (Blocked).
3. **Metrics.** 37 questions and 109 hand-labelled claims; the output tables match `evaluation/results/review3_corrected_nli_tables.md`.

## 3. Versions and configuration

Recorded automatically in `demo/outputs/versions.txt`. Key settings from `config/config.yaml`:

| Item | Value |
|---|---|
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 (384-d) |
| Vector index | FAISS IndexFlatIP, top_k 8 |
| NLI model | cross-encoder/nli-deberta-v3-small (labels read from `id2label`) |
| BERTScore model | distilbert-base-uncased |
| Fusion weights | NLI 0.45, BERTScore 0.35, cosine 0.20 |
| Thresholds | Supported at 0.75 or above; Weakly Supported at 0.50 or above; otherwise Hallucinated |

## 4. Suggested video (3 to 5 minutes)

1. (20 s) Title, problem statement: RAG answers can still contain unsupported sentences.
2. (30 s) Show the repo structure and `docs/trustrag_architecture.png`.
3. (60 s) Run `./run_demo.sh`; show the version record and the index step.
4. (90 s) Walk through the Part B output: the grounded answer is accepted, the mixed answer is partial with warnings, the fabricated answer is blocked.
5. (45 s) Show the regenerated metrics table and compare it with `review3_corrected_nli_tables.md`.
6. (30 s) Show `streamlit run ui/app.py`: ask one question and show the evidence attribution and detection log.
7. (15 s) Close with the limitations: small development benchmark, plus the external RAGTruth result.

## 5. Known limitations

- The 109-claim benchmark is a development set; the held-out RAGTruth evaluation is the external result.
- Without Ollama the answer text is the retrieved passage, not LLM-written.
- The paper's 9,800-chunk figures are not reproduced by this repository.
