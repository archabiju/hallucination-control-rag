# Step 4 — XAI Explainability Layer (Core Novelty)

**Module Location:** `src/explainability/`  
**File:** `xai_layer.py`

---

## Why Does This Layer Exist?

Most RAG systems work like a black box:
- Documents go in
- An answer comes out
- Nobody knows *why* the system produced that specific answer or which documents contributed to it

This project's **core research contribution** is making this process transparent. The XAI (Explainability) layer captures **internal signals that already exist** during retrieval and uses them to explain and verify the generated response.

The key insight: **You don't need to add an external fact-checker after the fact. The evidence quality signals are already there inside the pipeline — you just need to capture and use them.**

---

## What Are "Internal Signals"?

When the FAISS retrieval step runs, it already computes:
- How similar each retrieved chunk is to the query (cosine similarity score)
- Which chunks ranked highest (Top-K results)

The XAI layer **preserves and extends** these signals rather than discarding them after retrieval.

---

## Signal 1: Cosine Similarity Scores

### What It Is

After FAISS retrieves the Top-K chunks, the XAI layer records the **cosine similarity score** between the query and each retrieved chunk.

**Cosine similarity** measures the angle between two vectors in 384-dimensional space:
- Score of **1.0** = perfectly identical in meaning
- Score of **0.7–0.9** = highly relevant
- Score of **0.4–0.6** = somewhat related
- Score of **0.0–0.3** = mostly unrelated

### Formula

```
Cosine Similarity = (query_vector · chunk_vector) / (|query_vector| × |chunk_vector|)
```

Since all vectors are L2-normalized (length = 1.0), this simplifies to just the dot product:

```
Cosine Similarity = query_vector · chunk_vector
```

### Why This Signal Matters

If the highest similarity score is only 0.35, it tells us that even the "best" retrieved document isn't very relevant to the question. This is a red flag — the LLM may be generating an answer without solid evidence backing.

---

## Signal 2: Chunk Attribution (Provenance Mapping)

### The Problem It Solves

After the LLM generates a multi-sentence response, which retrieved chunk is responsible for each sentence?

**Example:**

Retrieved chunks:
- Chunk A: *"BERTScore measures semantic similarity using BERT token embeddings."*
- Chunk B: *"NLI determines whether a hypothesis is entailed by a premise."*
- Chunk C: *"FAISS supports billion-scale similarity search on CPUs."*

Generated response:
- Sentence 1: *"BERTScore evaluates token-level similarity between generated and reference text."*
- Sentence 2: *"Natural language inference checks logical entailment between texts."*
- Sentence 3: *"Vector databases allow fast offline retrieval without cloud APIs."*

The attribution map links each sentence back to its strongest evidence chunk:
- Sentence 1 → Chunk A (high cosine similarity)
- Sentence 2 → Chunk B (high cosine similarity)
- Sentence 3 → Chunk C (moderate cosine similarity)

### How It Works

```python
# For each generated sentence g_j:
# Compute cosine similarity with each retrieved chunk
A_ij = cosine(embedding(g_j), embedding(chunk_i))

# The chunk with the highest A_ij becomes the "cited source" for that sentence
best_chunk = argmax(A_ij)
```

### Why This Matters

This is what enables the system to produce **inline citations** like:
> *"BERTScore evaluates token-level similarity [Faithfulness_Review.pdf, p.4]"*

Without this attribution step, the system would not know which document to cite for each sentence.

---

## Signal 3: Provenance Score

### What It Is

The **Provenance Score** is a single number (0 to 1) that measures: *"Overall, what fraction of this response is traceable back to actual retrieved evidence?"*

It is computed as the **average attribution score** across all generated sentences:

```
Provenance Score = mean(A_ij for best chunk of each sentence)
```

### Interpreting the Score

| Provenance Score | Meaning |
|:---|:---|
| 0.85 – 1.00 | Excellent — nearly every sentence has strong evidence backing |
| 0.65 – 0.84 | Good — most sentences are grounded, some are weak |
| 0.50 – 0.64 | Borderline — significant parts of the response lack evidence |
| Below 0.50 | Poor — the response is mostly unsupported |

The system's reported Provenance Score from the paper is **0.936** — meaning 93.6% of generated content is strongly traceable to retrieved evidence.

---

## Signal 4: Similarity Heatmap

### What It Is

A word-level visualization that shows *how much evidence support* each part of the generated answer has.

Think of it like a colour-coded heat map:
- 🟢 **Green (high similarity):** This sentence/phrase is strongly backed by evidence
- 🟡 **Yellow (moderate similarity):** This part has some evidence but it's not very strong
- 🔴 **Red (low similarity):** This part has little to no evidence — potential hallucination

### How It's Used

The heatmap data is displayed in the **Streamlit dashboard** (`ui/app.py`) so the user can visually inspect the reliability of each part of the answer at a glance.

---

## Summary of XAI Signals

| Signal | Type | What It Measures | Used For |
|:---|:---|:---|:---|
| Cosine Similarity | Per-chunk float (0–1) | How relevant each chunk is to the query | Retrieval quality assessment |
| Chunk Attribution | Per-sentence mapping | Which chunk supports each sentence | Citation generation |
| Provenance Score | Single float (0–1) | Overall fraction of response that is evidence-backed | Response-level quality indicator |
| Similarity Heatmap | Per-word colour values | Word-level evidence strength | UI visualization |

---

## Current Status

- ✅ All four signals implemented in `xai_layer.py`
- ✅ Attribution mapping working: each sentence linked to its best supporting chunk
- ✅ Provenance score computed and available to the controller
- ✅ Heatmap data structure ready for Streamlit visualization
- ✅ Signals feed directly into the hallucination detection engine
