# TrustRAG: Hallucination Control and Evidence Attribution in Offline RAG

## 📌 1. What Are You Building? (High-Level Overview)

You are building **TrustRAG**, a privacy-focused **Offline Retrieval-Augmented Generation (RAG)** system designed to eliminate **hallucinations** in Large Language Model (LLM) outputs.

### The Core Problem:
Standard LLMs often generate confident, fluent text that is **factually wrong** or **unsupported** by retrieved documents (hallucination). Existing solutions rely on expensive, online cloud APIs (like OpenAI GPT-4) for checking factual accuracy, which violates data privacy in sensitive domains (defense, medical, legal).

### Your Solution (TrustRAG):
An **entirely local/offline pipeline** that:
1. Retrieves top relevant evidence passages from a vector index (**Sentence-BERT + FAISS**).
2. Generates an answer using a local LLM (**Ollama**).
3. Evaluates every generated sentence using **3 explainability signals**:
   - **NLI (DeBERTa)**: Does the evidence *logically entail* the sentence?
   - **BERTScore**: Is there strong *token-level semantic similarity*?
   - **Cosine Similarity**: Do the dense embedding vectors align?
4. Computes a **Sentence Trust Score** and classifies each sentence into a **Tri-State**:
   - 🟢 **Supported** (Trust Score $\ge 0.70$)
   - 🟡 **Weakly Supported** ($0.45 \le$ Trust Score $< 0.70$)
   - 🔴 **Hallucinated** (Trust Score $< 0.45$)
5. Enforces **Response Suppression**: If the ratio of hallucinated sentences in a response is too high, the system automatically **suppresses** the answer to prevent spreading false information.

---

## 🗂️ 2. MS MARCO Dataset & Data Processing Guide

You have already downloaded the **MS MARCO** (Microsoft Machine Reading Comprehension) dataset in your root folder:
- `collection.tar.gz` (3.2+ million passage collection)
- `queries.tar.gz` (Evaluation and training queries)
- `qrels.dev.tsv` / `qrels.train.tsv` (Relevance judgments mapping queries to passage IDs)

### How Data Processing Works Step-by-Step:

#### **Step 1: Extract and Subset Passages**
Since `collection.tar.gz` is large (~1GB compressed, ~3GB uncompressed), we extract a clean subset of passages (e.g., top 1,000 to 10,000 passages) into `data/corpus/msmarco_subset.txt`.

Command to process data:
```bash
python process_msmarco.py --num_passages 1000
```

#### **Step 2: Vector Indexing (FAISS + Sentence-BERT)**
Once passages are placed in `data/corpus/`, run the indexing workflow to convert raw text passages into 384-dimensional dense vectors using `all-MiniLM-L6-v2` and build the local FAISS index (`data/index/`).

Command to index documents:
```bash
python src/pipeline.py
```

#### **Step 3: Run Interactive Queries & Validation**
After indexing, you can query the system or run the interactive web interface:

Command to query via CLI:
```bash
python src/pipeline.py query
```

Command to launch Streamlit Dashboard UI:
```bash
streamlit run ui/app.py
```

---

## 📐 3. End-to-End System Architecture

```
User Query
    │
    ▼
┌───────────────────────────────────────┐
│ 1. Local Semantic Retrieval (FAISS)   │  ◄── Uses Sentence-BERT
└──────────────────┬────────────────────┘
                   │ Top-k Passages
                   ▼
┌───────────────────────────────────────┐
│ 2. Local LLM Response Generation      │  ◄── Local LLM (Ollama)
└──────────────────┬────────────────────┘
                   │ Raw Answer
                   ▼
┌───────────────────────────────────────┐
│ 3. Multi-Signal XAI Verification      │
│   • NLI Entailment (DeBERTa-v3)       │
│   • BERTScore (Token Alignment)       │
│   • Cosine Similarity (SBERT)         │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 4. Trust Score & Tri-State Controller │
│   • Supported  /  Weakly  /  Hallucinated │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 5. Response-Level Control             │
│   • Retained (with citation badges)   │
│   • Suppressed (if Hallucination Ratio > T) │
└───────────────────────────────────────┘
```

---

## 🚀 4. Step-by-Step Execution Guide for Review 1

1. **Process Dataset**:
   ```bash
   python process_msmarco.py --num_passages 1000
   ```
2. **Build Index**:
   ```bash
   python src/pipeline.py
   ```
3. **Launch Demonstration Dashboard**:
   ```bash
   streamlit run ui/app.py
   ```

---

## 📌 Summary for viva / review:
> *"I have implemented an offline RAG framework called TrustRAG. Using the MS MARCO dataset, I extract document passages, generate SBERT embeddings, and build a local FAISS index. When an LLM generates a response, my multi-signal XAI layer evaluates logical entailment using NLI (DeBERTa), token alignment using BERTScore, and vector similarity using Cosine similarity. This computes a sentence-level Trust Score, categorizes claims into a tri-state model, and automatically suppresses answers that exceed the hallucination threshold."*
