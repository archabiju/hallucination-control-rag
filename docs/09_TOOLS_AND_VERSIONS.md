# All Tools, Libraries, and Versions Used

**Configuration File:** `config/config.yaml`  
**Dependency File:** `requirements.txt`

This document lists every tool and library used in the TrustRAG project, with the version used and a clear explanation of *why* that specific tool was chosen over alternatives.

---

## 1. Python Environment

| Item | Version | Why |
|:---|:---|:---|
| **Python** | 3.9+ | Stable, widely supported, compatible with all ML libraries used |
| **Virtual Environment** | `.venv/` | Isolated environment so project dependencies don't conflict with system packages |

---

## 2. Core ML / NLP Libraries

### sentence-transformers >= 2.2.2
- **Used for:** Converting text into dense embedding vectors (384-dim)
- **Model used:** `all-MiniLM-L6-v2`
- **Why:** Specifically designed for sentence-level semantic similarity. Encodes full sentences into meaningful vectors, not just individual words. Lightweight enough to run locally on CPU.
- **Alternative considered:** OpenAI `text-embedding-ada-002` — rejected because it requires internet and API keys

### transformers >= 4.35.0
- **Used for:** Loading and running the DeBERTa NLI model
- **Why:** HuggingFace Transformers is the industry-standard library for loading pre-trained transformer models locally. Supports all major model architectures including DeBERTa, BERT, DistilBERT.
- **Alternative considered:** Running models via cloud API — rejected for offline requirement

### torch >= 2.0.0 (PyTorch)
- **Used for:** Backend computation engine for all transformer models (DeBERTa, DistilBERT)
- **Why:** PyTorch is the most widely used deep learning framework in research. HuggingFace Transformers requires PyTorch as a backend. Version 2.0+ includes significant CPU performance improvements.

### bert-score >= 0.3.13
- **Used for:** Computing BERTScore between generated sentences and retrieved evidence
- **Model used internally:** `distilbert-base-uncased`
- **Why:** BERTScore is specifically designed for evaluating text generation quality at the token level. It was introduced in a 2020 ICLR paper and has become a standard metric for faithfulness evaluation.
- **Why DistilBERT inside BERTScore:** It is 40% smaller and 60% faster than full BERT while retaining 97% of performance — critical for running BERTScore on hundreds of sentence pairs locally

---

## 3. Vector Search and Retrieval

### faiss-cpu >= 1.7.4
- **Used for:** Storing and searching embedding vectors for Top-K retrieval
- **Index type used:** `IndexFlatIP` (exact inner-product search = cosine similarity on normalized vectors)
- **Why FAISS:** Created by Meta AI Research; the industry standard for dense vector search. Can search millions of vectors in milliseconds on CPU alone. Supports disk persistence — build once, load forever.
- **Why `faiss-cpu` not `faiss-gpu`:** The project runs on standard consumer hardware without a dedicated GPU. `faiss-cpu` is fully functional for our corpus size (9,800 chunks).
- **Alternative considered:** ChromaDB, Pinecone, Weaviate — all require either a running server or cloud subscription. FAISS works as a simple Python library with no server overhead.

---

## 4. NLI Hallucination Detection

### Model: cross-encoder/nli-deberta-v3-small
- **Used for:** Classifying the logical relationship (Entail / Neutral / Contradict) between a retrieved chunk and a generated sentence
- **Why DeBERTa:** DeBERTa-v3 (Microsoft Research) uses disentangled attention — it considers both the content and position of words separately, giving better NLI accuracy than standard BERT
- **Why cross-encoder:** A cross-encoder processes premise and hypothesis together in one model pass. This is much more accurate than a bi-encoder (which encodes them separately), because it can attend to the interaction between the two texts directly
- **Why `-small`:** We needed fast per-sentence inference on CPU. The small variant is ~70MB and runs in ~200ms per pair on CPU, making it practical for real-time use
- **Alternative considered:** `cross-encoder/nli-deberta-v3-large` — more accurate but ~4× slower; too slow for real-time use without GPU

---

## 5. Document Processing

### PyPDF2 >= 3.0.1
- **Used for:** Extracting text from PDF files in `data/corpus/`
- **Why:** Lightweight, pure Python PDF reader. No external system dependencies. Handles most standard academic PDFs well.

### python-docx >= 0.8.11
- **Used for:** Extracting text from `.docx` Word document files
- **Why:** The official Python library for reading Microsoft Word format. Correctly handles paragraphs and text formatting.

---

## 6. LLM Generation

### Ollama (external tool)
- **Used for:** Running the Mistral-7B model locally on the machine
- **Version:** Latest stable
- **Why:** Ollama is the easiest way to run large language models locally. It handles model download, quantization (4-bit), and provides a simple REST API that the Python code can call via `localhost:11434`
- **Alternative considered:** LlamaCpp Python bindings — more complex setup; Ollama abstracts all of this

### LangChain >= 0.1.0 + langchain-community >= 0.0.10
- **Used for:** Orchestration of LLM calls, prompt template management
- **Why:** LangChain provides standardized interfaces for connecting different LLM backends (Ollama, OpenAI, Anthropic, etc.) to the same Python code. Makes it easy to switch between models without rewriting pipeline code.

---

## 7. Visualization and Dashboard

### streamlit >= 1.28.0
- **Used for:** The interactive web dashboard (`ui/app.py`)
- **Why:** Most widely used library for ML research dashboards. Pure Python — no frontend skills needed.

### plotly >= 5.18.0
- **Used for:** Interactive charts — similarity heatmaps, Trust Score gauges, Recall-Precision curves
- **Why:** Plotly generates interactive charts (hover-to-inspect, zoom) that look professional and work inside Streamlit

### pandas >= 2.0.0
- **Used for:** Displaying tables of retrieved chunks, scores, and evaluation results
- **Why:** Standard Python data science library; Streamlit natively supports Pandas DataFrames for table display

### matplotlib >= 3.7.0 + seaborn >= 0.12.0
- **Used for:** Static charts for the evaluation module and paper figures
- **Why:** Standard Python plotting libraries; seaborn provides attractive statistical plots with minimal code

### altair >= 5.0.0
- **Used for:** Additional interactive visualizations in the Streamlit dashboard
- **Why:** Altair integrates natively with Streamlit for responsive, declarative chart specification

---

## 8. Data Processing and Utilities

### numpy >= 1.24.0
- **Used for:** All vector mathematics — dot products, norms, matrix operations
- **Why:** The fundamental Python numerical computing library; used by PyTorch, FAISS, and sentence-transformers internally

### scikit-learn >= 1.3.0
- **Used for:** Evaluation metrics — AUC-ROC, F1-Score, Precision-Recall curves
- **Why:** Standard Python ML metrics library; provides all evaluation functions needed for benchmarking

### nltk >= 3.8.0
- **Used for:** Text tokenization and sentence splitting
- **Why:** The Natural Language Toolkit — standard Python NLP library; used in MS MARCO preprocessing for tokenization and cleaning

### pyyaml >= 6.0
- **Used for:** Reading `config/config.yaml` settings
- **Why:** Standard YAML parser; allows all configuration (thresholds, model names, paths) to be in a single readable file without hard-coding values in Python

### tqdm >= 4.65.0
- **Used for:** Progress bars during embedding and indexing (so you can see progress when processing thousands of chunks)
- **Why:** The simplest and most widely used Python progress bar library

### requests >= 2.31.0 + httpx >= 0.24.0
- **Used for:** HTTP calls to the local Ollama server
- **Why:** Ollama exposes a REST API at localhost; `httpx` is used for async-compatible HTTP calls in the pipeline

---

## 9. Summary Table

| Library | Version | Role |
|:---|:---|:---|
| `sentence-transformers` | >= 2.2.2 | Text → 384-dim embedding vectors |
| `faiss-cpu` | >= 1.7.4 | Vector index and similarity search |
| `transformers` | >= 4.35.0 | Load DeBERTa NLI model |
| `torch` | >= 2.0.0 | Backend for all neural network models |
| `bert-score` | >= 0.3.13 | BERTScore faithfulness metric |
| `langchain` | >= 0.1.0 | LLM pipeline orchestration |
| `langchain-community` | >= 0.0.10 | Ollama integration for LangChain |
| `PyPDF2` | >= 3.0.1 | PDF text extraction |
| `python-docx` | >= 0.8.11 | DOCX text extraction |
| `streamlit` | >= 1.28.0 | Interactive web dashboard |
| `plotly` | >= 5.18.0 | Interactive charts and heatmaps |
| `pandas` | >= 2.0.0 | Data tables in dashboard |
| `numpy` | >= 1.24.0 | Numerical computations |
| `scikit-learn` | >= 1.3.0 | Evaluation metrics (F1, AUC-ROC) |
| `nltk` | >= 3.8.0 | Text tokenization |
| `pyyaml` | >= 6.0 | Config file reading |
| `matplotlib` | >= 3.7.0 | Static figure plotting |
| `seaborn` | >= 0.12.0 | Statistical plot styling |
| `altair` | >= 5.0.0 | Dashboard visualization |
| `tqdm` | >= 4.65.0 | Progress bars |
| `requests` | >= 2.31.0 | HTTP to Ollama |
| `httpx` | >= 0.24.0 | Async-compatible HTTP |
| **Ollama** | Latest | Local LLM server (external tool) |
| **Mistral-7B-Instruct** | v0.2 | Local language model for generation |
| **DeBERTa-v3-small** | Cross-encoder | NLI hallucination detection |
| **all-MiniLM-L6-v2** | — | Sentence embedding model |
| **DistilBERT-base** | — | BERTScore computation |
