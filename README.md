# 🛡️ Explainability-Driven Hallucination Detection Framework for Offline RAG

> **"We are not building a chatbot. We are building a knowledge system that retrieves documents, explains why they were selected, and only then generates answers."**

An offline, transparent, evidence-grounded Retrieval-Augmented Generation (RAG) system with intrinsic hallucination detection using NLI entailment, BERTScore, and XAI-based evidence attribution.

**No cloud APIs. No ChatGPT. No black-box answers.**

---

## 📐 System Architecture

```
User Query
    │
    ▼
┌───────────────────────┐
│  1. Document Store     │  PDFs, DOCX, TXT → chunked (512 tokens, 50 overlap)
│     └─ Embedder        │  all-MiniLM-L6-v2 (384-dim, offline)
│     └─ FAISS Index     │  IndexFlatIP (cosine similarity)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  2. Retriever          │  Top-K chunks via FAISS cosine search
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  3. XAI Layer ★        │  Cosine scores, chunk attribution,
│     (NOVELTY)          │  provenance scoring, word saliency
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  4. LLM Generator      │  Mistral 7B / LLaMA 3 via Ollama (local)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  5. Hallucination      │  NLI Entailment (DeBERTa) +
│     Detector           │  BERTScore F1 + Cosine Similarity
│                        │  → Aggregated Support Score per sentence
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  6. Decision           │  Score ≥ 0.75 → ✅ Cited Answer
│     Controller         │  Score ≥ 0.50 → ⚠️  Partial + Warnings
│                        │  Score < 0.50 → 🚫 Blocked + Clarify
└───────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Fully Offline** | All models run locally — no internet needed after setup |
| **Explainability-First** | XAI signals are the primary hallucination detection mechanism |
| **Tri-State Control** | System blocks hallucinated answers, not just flags them |
| **Evidence Attribution** | Every claim is traced back to a specific source chunk |
| **Provenance Scoring** | Measures what fraction of the answer is document-grounded |
| **Interactive Dashboard** | Streamlit UI with real-time charts and detection logs |

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Embeddings | `all-MiniLM-L6-v2` | Offline, fast, 80MB, 384-dim |
| Vector Search | `FAISS (CPU)` | In-memory, no server, blazing fast |
| LLM | `Mistral 7B` / `LLaMA 3` | Open-source, offline via Ollama |
| NLI | `DeBERTa NLI cross-encoder` | Best entailment accuracy, lightweight |
| Semantic Score | `BERTScore` | Token-level overlap, no external calls |
| UI | `Streamlit` + `Plotly` | Rich interactive dashboard |

---

## 📁 Project Structure

```
MAIN PROJECT/
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py    # PDF/DOCX/TXT loading
│   │   └── chunker.py            # 512-token sliding-window chunker
│   ├── retrieval/
│   │   ├── embedder.py           # all-MiniLM-L6-v2 embeddings
│   │   └── faiss_store.py        # FAISS index build/search/persist
│   ├── generation/
│   │   └── llm_generator.py      # Ollama local LLM wrapper
│   ├── explainability/
│   │   └── xai_layer.py          # ★ Core XAI: attribution, provenance, heatmaps
│   ├── detection/
│   │   ├── nli_detector.py       # DeBERTa NLI entailment scoring
│   │   ├── bertscore_detector.py # BERTScore semantic overlap
│   │   └── hallucination_engine.py # Aggregated support score + classification
│   ├── controller/
│   │   └── decision_controller.py # Tri-state output logic
│   └── pipeline.py               # End-to-end orchestration
├── ui/
│   └── app.py                    # Streamlit interactive dashboard
├── data/
│   ├── corpus/                   # ← Place your domain documents here
│   └── index/                    # FAISS index (auto-generated)
├── evaluation/
│   ├── metrics.py                # P/R/F1, AUROC, MRR, NDCG
│   └── evaluator.py             # Test set evaluation runner
├── config/
│   └── config.yaml              # All thresholds & model settings
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Guide

### Prerequisites

- Python 3.9+
- (Optional) Ollama for local LLM generation

### Step 1: Install Dependencies

```bash
cd "MAIN PROJECT"
pip install -r requirements.txt
```

### Step 2: Install Ollama (for LLM generation)

```bash
# Download from https://ollama.ai and install
# Then pull a model:
ollama pull mistral
# Start the server:
ollama serve
```

> **Note:** The system works without Ollama too — it has a fallback mode
> that uses extractive answers from retrieved chunks for testing.

### Step 3: Add Documents to Corpus

Place your domain PDF, DOCX, or TXT files in:
```
data/corpus/
```

### Step 4: Launch the Dashboard

```bash
streamlit run ui/app.py
```

### Step 5: Build Index & Query

1. Click **🔨 Build Index** in the sidebar
2. Type a question and click **🚀 Query**
3. View color-coded results, explainability charts, and detection logs

---

## 💻 CLI Usage

### Index documents
```bash
python src/pipeline.py
```

### Interactive query mode
```bash
python src/pipeline.py query
```

### Run evaluation
```bash
python evaluation/evaluator.py
```

---

## 🎯 Hallucination Detection Algorithm

For each generated answer, the system:

1. **Splits** the answer into individual sentences (claims)
2. **Attributes** each claim to its best-matching source chunk
3. **Scores** each claim using three independent detectors:

| Detector | What it catches | Score |
|----------|----------------|-------|
| **NLI Entailment** | Logical contradictions | P(entailment) |
| **BERTScore** | Semantic drift | F1 overlap |
| **Cosine Similarity** | Out-of-domain claims | cos(claim, chunk) |

4. **Aggregates** into a single support score:

```
support_score = 0.45 × NLI + 0.35 × BERTScore + 0.20 × Cosine
```

5. **Classifies** each claim:

| Support Score | Classification | Action |
|--------------|----------------|--------|
| ≥ 0.75 | `SUPPORTED` | ✅ Answer with `[source, page]` citation |
| ≥ 0.50 | `WEAKLY_SUPPORTED` | ⚠️ Partial answer with warning flag |
| < 0.50 | `HALLUCINATED` | 🚫 Suppressed or entire answer blocked |

---

## 📊 Evaluation Metrics

The evaluation module computes:

- **Precision / Recall / F1** for hallucination detection
- **AUC-ROC** for score discrimination quality
- **Recall@K / MRR / NDCG** for retrieval quality
- **Provenance score** for answer traceability

Run evaluation:
```bash
python evaluation/evaluator.py
```

Results and plots are saved to `evaluation/results/`.

---

## 🔧 Configuration

All thresholds and model settings are in `config/config.yaml`:

```yaml
detection:
  weight_nli:        0.45    # NLI entailment weight
  weight_bertscore:  0.35    # BERTScore weight
  weight_cosine:     0.20    # Cosine similarity weight
  high_support_threshold:    0.75   # ≥ this → full answer
  medium_support_threshold:  0.50   # ≥ this → partial answer
```

These can also be adjusted in real-time via the Streamlit sidebar sliders.

---

## 🔬 Research Contribution

> **Novelty:** This project introduces an explainability-driven hallucination
> detection mechanism that uses **internal** evidence attribution and confidence
> signals from a RAG pipeline. Unlike existing methods that depend on external
> validation models, the proposed approach enables transparent, offline, and
> evidence-grounded generation.

Most RAG systems: `Retrieve → Generate → (maybe) Check`
**This system:** `Retrieve → Score → Generate → Detect using those same scores → Decide`

The explainability signals are **not an afterthought** — they are built into
the pipeline from the start, making hallucination detection **intrinsic,
transparent, and offline-capable**.

---

## 📄 License

This project is developed as part of an M.Tech thesis at the Department of
Computer Science. For academic use only.
