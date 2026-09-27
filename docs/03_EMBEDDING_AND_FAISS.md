# Step 3 — Embedding and FAISS Indexing

**Module Location:** `src/retrieval/`  
**Files:** `embedder.py`, `faiss_store.py`  
**Index Output:** `data/index/faiss.index`, `data/index/chunks_metadata.pkl`

---

## The Core Idea

Once documents are split into chunks, the system needs a way to **quickly find which chunks are relevant to a given question**.

Traditional keyword search (like CTRL+F) only works if the question uses the *exact same words* as the document. But users don't always phrase questions the same way as the text.

**Example:**
- Document says: *"LLMs can fabricate information not found in training data."*
- User asks: *"Do AI models make up facts?"*
- Keyword search would fail — "fabricate" ≠ "make up", "LLM" ≠ "AI model"
- But both sentences **mean the same thing**

The solution is to convert text into **mathematical vectors** (lists of numbers) that capture *meaning*, not just words. Then finding similar text becomes a **mathematical distance calculation**.

---

## Step 3A — Embedding: `embedder.py`

### What Is an Embedding?

An embedding is a list of numbers that represents the meaning of a piece of text.

- The word "dog" and the word "puppy" will have vectors that are very close to each other (similar meaning)
- The word "dog" and the word "calculus" will have vectors that are far apart (different meaning)

### Which Model Is Used?

**Model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Library:** `sentence-transformers >= 2.2.2`  
**Output Dimension:** 384 numbers per text

### Why This Specific Model?

| Reason | Detail |
|:---|:---|
| **Sentence-level accuracy** | Unlike word-embedding models (Word2Vec, GloVe), this model encodes an *entire sentence or paragraph* into one vector — capturing full-sentence meaning |
| **Lightweight** | "MiniLM" means "Miniature Language Model" — it is a distilled, smaller version of larger BERT-family models. It runs fast even on CPU-only machines |
| **L6** | Has 6 Transformer layers — a good balance between accuracy and speed |
| **Offline capable** | After the first download from HuggingFace, the model is cached locally and runs without any internet connection |
| **Industry standard** | Widely used in RAG and semantic search systems; performance benchmarks are well-published and reproducible |
| **384 dimensions** | Compact enough to store thousands of vectors in RAM while still capturing rich semantic information |

### How It Works

```
Input text:  "Retrieval-Augmented Generation combines retrieval with LLMs."
     ↓
Tokenize → [CLS] retrieval augmented generation combines ... [SEP]
     ↓
Pass through 6 Transformer layers
     ↓
Mean pooling of all token outputs
     ↓
L2 Normalization (makes all vectors length = 1.0)
     ↓
Output: [0.021, -0.134, 0.072, 0.198, ..., -0.041]  ← 384 numbers
```

**L2 Normalization** is important: once all vectors have length = 1.0, computing **cosine similarity** (how similar two vectors are in direction) becomes as simple as computing their **dot product** — which is much faster.

---

## Step 3B — FAISS Vector Index: `faiss_store.py`

### What Is FAISS?

**FAISS** stands for **Facebook AI Similarity Search**.  
**Library:** `faiss-cpu >= 1.7.4`  
**Created by:** Meta (Facebook) AI Research

It is a library specifically designed for **extremely fast similarity search** over millions of vectors. Think of it as a super-optimized database that answers the question: *"Which of these 10,000 vectors is most similar to my query vector?"*

### Why FAISS Instead of a Regular Database?

A regular database (like SQLite or MySQL) stores and retrieves rows by exact match (find row where ID = 5). It cannot natively find "the most similar vector to this query vector."

FAISS is designed *specifically* for vector similarity search. It can search through 10,000+ vectors in **milliseconds** on a CPU.

### Index Type Used: `IndexFlatIP`

**IP** = Inner Product (which equals Cosine Similarity when vectors are L2-normalized)

This is the **exact search** index — it compares the query vector against *every single stored vector* and returns the closest ones. It gives 100% accurate results (no approximation).

### What Gets Saved to Disk?

When indexing is complete, two files are saved to `data/index/`:

| File | Contents | Why |
|:---|:---|:---|
| `faiss.index` | The binary FAISS index with all 384-dim vectors | So the index doesn't need to be rebuilt every time the app starts |
| `chunks_metadata.pkl` | Python list of all chunk dictionaries (text, source_file, page_num, chunk_id) | So the system knows which text and source corresponds to each vector in the index |

### How Retrieval Works

When a user asks a question:

```
User Query: "How does BERTScore measure semantic similarity?"
     ↓
Embed query → query_vector (384 numbers)
     ↓
FAISS search: find top-5 vectors most similar to query_vector
     ↓
FAISS returns: indices [42, 107, 8, 203, 156] and their similarity scores
     ↓
Look up chunks_metadata[42], [107], [8], [203], [156]
     ↓
Return: 5 text chunks with their source files and page numbers
```

**`top_k = 5`** — the system retrieves the 5 most relevant chunks. This is set in `config/config.yaml`.

---

## Current Status

- ✅ `embedder.py` fully implemented using `all-MiniLM-L6-v2`
- ✅ `faiss_store.py` fully implemented with `IndexFlatIP`
- ✅ FAISS index built and saved: `data/index/faiss.index` (125 KB)
- ✅ Chunk metadata saved: `data/index/chunks_metadata.pkl` (238 KB)
- ✅ Index covers all 9 documents in `data/corpus/`

---

## What Happens Next?

The retrieved 5 chunks (with their similarity scores and source info) are passed to the **XAI Explainability Layer**, which maps them to the generated response sentences.
