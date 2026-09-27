# Step 2 — MS MARCO Dataset: Download and Processing

**Script Location:** `process_msmarco.py` (project root)  
**Raw Data Location:** `dataset_raw/`  
**Processed Output:** `data/corpus/msmarco_subset.txt` and `data/msmarco_processed/`

---

## What Is MS MARCO?

**MS MARCO** stands for **Microsoft Machine Reading Comprehension**.

It is a large-scale, publicly available dataset created by Microsoft Research. It contains:
- **~8.8 million real passages** collected from web pages
- **~1 million real user questions** submitted to the Bing search engine
- Human-annotated answers for evaluation

It is considered the **industry standard benchmark** for testing information retrieval and question answering systems. This means researchers worldwide use it to fairly compare their systems against each other.

---

## Why MS MARCO for This Project?

| Reason | Explanation |
|:---|:---|
| **Real-world queries** | Questions are actual Bing search queries from real users — not artificial lab questions |
| **Diverse domains** | Passages cover healthcare, science, sports, general knowledge, and more |
| **Human-annotated** | Passages are labelled with relevance judgements — useful for measuring retrieval quality |
| **Industry benchmark** | Using MS MARCO makes our results comparable with published research papers |
| **Publicly available** | No licensing restrictions — freely downloadable for academic use |

---

## What Files Were Downloaded?

All raw files are stored in the `dataset_raw/` folder:

| File | Size | Contents |
|:---|:---|:---|
| `collection.tsv` | ~3 GB | The full MS MARCO passage collection (8.8M passages) |
| `collection.tar.gz` | ~1 GB | Compressed version of the collection |
| `queries.dev.tsv` | ~4.4 MB | All development set queries (questions) |
| `queries.tar.gz` | ~18 MB | Compressed version of queries |
| `qrels.dev.tsv` | ~1.2 MB | Relevance labels: which passages answer which query |
| `qrels.train.tsv` | ~10 MB | Training set relevance labels |

> **Note:** The `collection.tsv` is 3 GB. This is the raw unprocessed version. We only use a small subset of 10,000 passages for our system to keep it manageable on a local machine.

---

## What Does `process_msmarco.py` Do?

Since we cannot use all 8.8 million passages (that would need hundreds of GB of RAM), the script selects a manageable subset and converts it into a format our pipeline can use.

### Step-by-Step Process

**Step 1: Extract the queries file**
```
If queries.dev.tsv is not present, extract it from queries.tar.gz
```
The `queries.tar.gz` is a compressed archive. Python's `tarfile` library extracts it automatically.

**Step 2: Read the passage collection**
```
Open collection.tsv
Read the first N passages (default: 1,000 or 10,000)
```
Each line in `collection.tsv` is formatted as:
```
passage_id [TAB] passage_text
```

**Step 3: Write passages to the corpus folder**
```
Save selected passages as plain text files in data/corpus/
or as a single combined file: data/corpus/msmarco_subset.txt
```
Once in `data/corpus/`, these passages are treated exactly like your local PDF documents — they get chunked, embedded, and indexed into FAISS.

### Running the Script

```bash
# Process 1,000 passages (quick test)
python process_msmarco.py --num_passages 1000

# Process 10,000 passages (used in evaluation)
python process_msmarco.py --num_passages 10000
```

---

## How MS MARCO Fits Into the Pipeline

```
dataset_raw/collection.tsv
         ↓
   process_msmarco.py
         ↓
data/corpus/msmarco_subset.txt
         ↓
   document_loader.py  (reads as plain text)
         ↓
   chunker.py          (splits into 512-token chunks)
         ↓
   embedder.py         (converts to 384-dim vectors)
         ↓
   faiss_store.py      (stores in FAISS index)
```

The MS MARCO passages become part of the **searchable knowledge base** — exactly the same pipeline as the local PDF/DOCX documents.

---

## Current Status

- ✅ All raw MS MARCO files downloaded to `dataset_raw/`
- ✅ `process_msmarco.py` implemented and working
- ✅ A subset (`msmarco_subset.txt`) placed in `data/corpus/` for indexing
- ⚠️ `data/msmarco_processed/` is currently empty — planned to store passages individually for evaluation benchmarking in Semester 3
