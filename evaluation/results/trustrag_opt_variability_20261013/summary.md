# TrustRAG-Opt repeated measurement

These numbers were measured from the saved index and the saved 37-question test set.
They are not the 9,800-chunk / 400-query figures in the draft.

## What was confirmed

- Runs: 5
- Queries per run: 37 (the existing `evaluation/test_set.json`)
- The same 37 queries were reused, unchanged, across every run and every configuration.
- There is no 400-query evaluation set in this repository, so a 400-query reuse cannot be confirmed.
- Warm-up: 1000 discarded searches per configuration per run, then 500 timed searches per query.
- Query embeddings were computed once and reused. Timed retrieval is index search only.
- Approximate indexes were rebuilt at the start of every run.
- Corpus actually measured: 213 chunks, dimension 384.

## Retrieval

Agreement is the fraction of the FlatIP top-5 ids that the configuration also returned.
Latency is in milliseconds. The ± value is the sample standard deviation across runs.
The 95% confidence interval of the mean is in `summary.json`.

| Configuration | Agreement@5 | Mean latency (ms) | p95 latency (ms) | Index footprint (MB) |
|---|---:|---:|---:|---:|
| FlatIP | 1.0000 ± 0.0000 | 0.0054 ± 0.0000 | 0.0062 ± 0.0002 | 0.312 |
| IVF-PQ (nbits=7) | 0.8919 ± 0.0000 | 0.0085 ± 0.0003 | 0.0106 ± 0.0014 | 0.379 |
| HNSW ef=16 | 0.9676 ± 0.0000 | 0.0047 ± 0.0002 | 0.0057 ± 0.0004 | 0.367 |
| HNSW ef=32 | 1.0000 ± 0.0000 | 0.0060 ± 0.0001 | 0.0071 ± 0.0003 | 0.367 |
| HNSW ef=64 | 1.0000 ± 0.0000 | 0.0085 ± 0.0002 | 0.0101 ± 0.0005 | 0.367 |
| HNSW ef=128 | 1.0000 ± 0.0000 | 0.0158 ± 0.0004 | 0.0192 ± 0.0009 | 0.367 |
| Hybrid FlatIP+BM25 | 0.5135 ± 0.0000 | 0.0764 ± 0.0010 | 0.1225 ± 0.0014 | 0.868 |

p95 is computed inside each run over all timed searches (queries × repetitions), then the mean and standard deviation of those five run-level percentiles are reported.

### Memory procedure

- Index footprint: size of the FAISS index written with `faiss.write_index`. The same write path is used for every configuration. Hybrid adds the byte size of the in-process BM25 postings.
- Runtime memory: a fresh Python process loads only that index file. RSS is read with `ps -o rss` after import, after `faiss.read_index`, and after 100 searches. Incremental RSS is load minus import.
- FAISS was restricted to one thread (`OMP_NUM_THREADS=1`).

| Configuration | Footprint (MB) | Incremental RSS after load (MB) | RSS after 100 searches (MB) |
|---|---:|---:|---:|
| FlatIP | 0.312 | 0.250 | 68.297 |
| IVF-PQ (nbits=7) | 0.379 | 0.438 | 68.781 |
| HNSW ef=16 | 0.367 | 0.391 | 68.625 |
| HNSW ef=32 | 0.367 | 0.359 | 68.906 |
| HNSW ef=64 | 0.367 | 0.375 | 68.609 |
| HNSW ef=128 | 0.367 | 0.359 | 68.688 |
| Hybrid FlatIP+BM25 | 0.868 | 0.250 | 68.547 |

HNSW footprint is one index shared by every `efSearch` setting. `efSearch` is a query-time parameter.

## End-to-end

End-to-end is the sum of that run's mean search latency and NLI verification of the question's labelled claims against the retrieved chunks.
Search and NLI were timed in separate processes. Loading FAISS and PyTorch together aborts with OMP Error #15.
Generation is omitted because Ollama is not running. The draft's 5.20 s generation figure was not reused.

| Configuration | End-to-end mean (ms) | End-to-end p95 (ms) | NLI mean (ms) |
|---|---:|---:|---:|
| FlatIP | 605.4130 ± 21.8105 | 725.3470 ± 46.9880 | 605.4076 ± 21.8105 |
| IVF-PQ (nbits=7) | 609.9268 ± 46.7333 | 721.2629 ± 71.5775 | 609.9183 ± 46.7331 |
| HNSW ef=16 | 609.7801 ± 42.9240 | 731.1618 ± 50.4187 | 609.7754 ± 42.9241 |
| HNSW ef=32 | 612.8327 ± 37.8886 | 714.5957 ± 49.6466 | 612.8267 ± 37.8886 |
| HNSW ef=64 | 619.5851 ± 41.4691 | 735.7427 ± 49.0169 | 619.5766 ± 41.4691 |
| HNSW ef=128 | 623.4146 ± 38.2108 | 738.5427 ± 31.9678 | 623.3987 ± 38.2108 |
| Hybrid FlatIP+BM25 | 629.5213 ± 35.7039 | 759.8125 ± 67.6246 | 629.4449 ± 35.7031 |

NLI device: mps:0. The model was loaded once, warmed with 8 pairs, then each run repeated the same queries.

## Deviations from the draft protocol

The draft describes 9,800 chunks and 400 queries. This repository's saved index has 213 chunks and evaluation/test_set.json has 37 questions. Those 37 questions were reused for every run. IVF-PQ with 8 bits did not train (RuntimeError: Error in void faiss::Clustering::train_encoded(idx_t, const uint8_t *, const Index *, Index &, const float *) at /Users/runner/work/faiss/faiss/faiss/Clustering.cpp:72: Error: 'nx >= static_cast<idx_t>(k)' failed: Number of training points (213) should be at least as large as number of clusters (256)). Measured IVF-PQ uses 7 bits; nlist=128, m=8, nprobe=16. Hybrid BM25 is an in-process inverted index because Elasticsearch is not part of this environment. Latency was timed with perf_counter_ns around each search, which is what makes a 95th percentile defined; the draft's timeit loop would have collapsed those repetitions into one average. End-to-end excludes generation. FAISS and PyTorch were not loaded in the same process: doing so aborts with OMP Error #15 because each ships its own OpenMP runtime. Query embedding and NLI ran in torch-only processes; index search ran in a FAISS-only process. End-to-end latency is the sum of those two measurements.

Raw per-query rows are in `run_XX/per_query.jsonl`. Every timed latency sample is in `run_XX/latency_*.npy`.
The earlier Review 2 file `evaluation/results/review2_report.json` was not modified.
