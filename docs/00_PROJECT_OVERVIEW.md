# TrustRAG — Project Overview

**Full Title:** TrustRAG: Explainability-Driven Hallucination Detection and Control in Offline Retrieval-Augmented Generation

**Student:** Archa Biju  
**Guide:** Dr. Ramya G R  
**Institution:** Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore  
**Programme:** M.Tech Computer Science (S2 — 2025–26)

---

## What Is This Project About?

Large Language Models (LLMs) like GPT, Mistral, and LLaMA can generate very fluent and confident-sounding answers. But they sometimes "hallucinate" — meaning they make up facts that sound real but are not supported by any evidence.

**Retrieval-Augmented Generation (RAG)** is a technique that tries to fix this by first *retrieving* relevant documents, and then giving those documents as context to the LLM before it answers. The idea is: "Don't rely on what the model remembers — give it the actual source material to read before answering."

However, even with RAG, the LLM can still ignore the retrieved evidence and generate unsupported claims.

**This project builds a system called TrustRAG** that:
1. Runs an entire offline RAG pipeline with **no internet or cloud API**
2. After the LLM generates a response, **checks each sentence** of that response against the retrieved evidence
3. Uses **three verification methods** (NLI, BERTScore, and Cosine Similarity) to assign a **Trust Score** to each sentence
4. Based on that Trust Score, makes a **decision**: show the answer with citations, show a partial answer with warnings, or block the response entirely

---

## Why Fully Offline?

Most existing hallucination detection systems rely on:
- OpenAI GPT-4 API (requires internet, costs money, sends your data to a third party)
- Cloud-based fact-checkers (same problems)

This project is designed for **high-security, privacy-sensitive, or disconnected environments** — like hospitals, defense research labs, or enterprise systems that cannot send data outside.

---

## The Core Novelty

The **Explainability Layer (XAI Layer)** is the heart of this project.

Instead of treating the RAG pipeline as a black box, this layer captures **internal signals** that already exist during retrieval:
- How similar is each retrieved document chunk to the query? *(Cosine Similarity Score)*
- Which retrieved chunk supports which generated sentence? *(Chunk Attribution)*
- What fraction of the entire response is traceable back to a source? *(Provenance Score)*
- A word-level heat map showing which parts of the answer have the strongest evidence backing *(Similarity Heatmap)*

These signals are used **directly** to detect and suppress hallucinations — not as an afterthought.

---

## System at a Glance

```
User Query
   ↓
[Embedder] — converts query to a vector
   ↓
[FAISS Index] — finds the most relevant document chunks
   ↓
[XAI Layer] — measures how relevant each chunk is; maps chunks to sentences
   ↓
[Local LLM — Mistral-7B via Ollama] — generates a response based on evidence
   ↓
[Claim Segmenter] — splits the response into individual sentences
   ↓
[Hallucination Detector] — checks each sentence: NLI + BERTScore + Cosine
   ↓
[Trust Score] — combined score per sentence (0 to 1)
   ↓
[Tri-State Decision Controller]
   ├── Score ≥ 0.75 → ✅ Answer with inline citations
   ├── Score 0.50–0.74 → ⚠️ Partial answer with warnings
   └── Score < 0.50  → 🚫 Block response, show only verified evidence
```

---

## Documentation Guide

| File | What It Explains |
|:---|:---|
| `01_DATA_INGESTION.md` | How documents are loaded and split into chunks |
| `02_MSMARCO_PROCESSING.md` | How the MS MARCO dataset was downloaded and processed |
| `03_EMBEDDING_AND_FAISS.md` | How text is converted to vectors and indexed |
| `04_XAI_EXPLAINABILITY_LAYER.md` | The core novelty — how internal signals are captured |
| `05_LLM_GENERATION.md` | How the local LLM generates responses |
| `06_HALLUCINATION_DETECTION.md` | How NLI, BERTScore, and Cosine are used to detect hallucinations |
| `07_TRUST_SCORE_AND_CONTROLLER.md` | How the Trust Score is computed and the tri-state decision is made |
| `08_STREAMLIT_UI.md` | The interactive dashboard for exploring results |
| `09_TOOLS_AND_VERSIONS.md` | All tools, libraries, and versions used with reasons |
