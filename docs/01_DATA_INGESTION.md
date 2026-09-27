# Step 1 — Data Ingestion: Loading and Chunking Documents

**Module Location:** `src/ingestion/`  
**Files:** `document_loader.py`, `chunker.py`

---

## What Is This Step?

Before the system can answer any question, it needs to **read and store** the knowledge it will use.

Think of it like this: before a student can answer exam questions, they first read their textbooks and notes. This step is equivalent to that reading phase — the system reads all the documents in the knowledge base and prepares them to be searched later.

---

## Step 1A — Loading Documents: `document_loader.py`

### What It Does

This file reads every document from the `data/corpus/` folder and extracts the plain text from it. It handles three types of files:

| File Type | Library Used | Why |
|:---|:---|:---|
| `.pdf` | `PyPDF2 >= 3.0.1` | Most research papers and technical manuals are PDFs |
| `.docx` | `python-docx >= 0.8.11` | Word documents (like the teaching guide) |
| `.txt` / `.md` | Python built-in `open()` | Plain text files (like the MS MARCO subset) |

### How It Works — Step by Step

1. The system scans the `data/corpus/` folder for all supported files.
2. For each file, it opens the file and reads the text **page by page**.
3. For each page, it creates a record that contains:
   - The actual text content
   - The name of the source file (`source_file`)
   - The page number (`page_num`)
4. All these records are collected into a list and passed to the chunker.

### Why Metadata Matters

Every chunk of text keeps track of **where it came from** (which file, which page). This is critical for the later step where the system needs to **cite its sources** in the answer — e.g., *"According to [Hallucination_Survey.pdf, p.3]: ..."*

---

## Step 1B — Chunking Text: `chunker.py`

### The Problem with Long Documents

A research paper might be 20 pages long. If you give all 20 pages to a search system, it can't accurately pinpoint which **specific paragraph** is relevant to the question. You need to break documents into smaller, focused pieces.

### What Chunking Does

Chunking splits each page's text into smaller overlapping windows of text. Each window is called a **chunk**.

**Settings used (from `config/config.yaml`):**
```
chunk_size:    512  tokens (words)
chunk_overlap:  50  tokens
```

### Why 512 Tokens?

- 512 tokens is roughly 1–2 paragraphs of dense academic text.
- It is large enough to preserve meaning and context within a paragraph.
- It is small enough that one chunk stays focused on a single topic or claim.
- 512 is also aligned with the max input window of many transformer-based models used later.

### Why 50 Token Overlap?

Imagine a sentence that spans the boundary between two chunks. Without overlap, that sentence would be split in half — one half in chunk #3, the other half in chunk #4. Neither chunk would have the full sentence.

With **50-token overlap**, the last 50 tokens of one chunk are **repeated as the first 50 tokens of the next chunk**. This ensures no sentence is ever cut off and lost at a boundary.

### What Each Chunk Contains

Each chunk is a small dictionary with:

```python
{
    "chunk_id":    "a3f2...",       # Unique ID for this chunk
    "text":        "RAG frameworks combine...",  # The actual text
    "source_file": "survey.pdf",   # Which document it came from
    "page_num":    3,              # Which page in that document
    "token_count": 512,            # How many words/tokens
    "start_word":  0,              # Starting word index
    "end_word":    512             # Ending word index
}
```

The `chunk_id` is a UUID (universally unique identifier) generated automatically so every chunk can be tracked and cited uniquely across the entire pipeline.

---

## Current Status

- ✅ `document_loader.py` fully implemented and tested
- ✅ `chunker.py` fully implemented with configurable size and overlap
- ✅ Corpus of 9 documents loaded from `data/corpus/`
- ✅ Approximately 200+ chunks generated and ready for embedding

---

## What Happens Next?

These chunks are passed to **Step 3 (Embedding)** where each chunk is converted into a 384-dimensional number vector so that the system can do mathematical similarity search later.
