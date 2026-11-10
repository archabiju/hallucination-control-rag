"""
Repeated TrustRAG-Opt retrieval measurement on the saved index and test set.

Reads data/index and evaluation/test_set.json. Does not modify them, and does
not overwrite evaluation/results/review2_*.

Paper protocol, applied to the data that actually exists:
  K = 5
  FlatIP, IVF-PQ (nlist=128, m=8, 8 bits, nprobe=16), HNSW (M=32,
  efConstruction=200, efSearch in {16, 32, 64, 128}), hybrid alpha=0.7
  1000 discarded warm-up searches per configuration per run, then 500 timed
  searches per query. Embeddings are cached. FAISS is pinned to one thread.

The saved corpus has 213 chunks, not 9,800, and the saved query set has 37
questions, not 400. IVF-PQ at 8 bits cannot be trained on 213 vectors because
each codebook has 256 centroids; that failure is recorded and the nearest
trainable width (7 bits) is measured.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "evaluation" / "results" / "trustrag_opt_variability_20261013"
INDEX_SRC = ROOT / "data" / "index" / "faiss.index"
META_SRC = ROOT / "data" / "index" / "chunks_metadata.pkl"
TEST_SRC = ROOT / "evaluation" / "test_set.json"

N_RUNS = 5
WARMUP = 1000
REPS = 500
K = 5
ALPHA = 0.7
NLIST = 128
PQ_M = 8
PQ_BITS_PAPER = 8
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF = (16, 32, 64, 128)
BM25_K1 = 1.5
BM25_B = 0.75


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rss_bytes() -> int:
    """Current resident set size. macOS `ps` reports RSS in 1024-byte units."""
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(out.strip().split()[0]) * 1024


def machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    for key, cmd in (
        ("cpu", ["sysctl", "-n", "machdep.cpu.brand_string"]),
        ("mem_bytes", ["sysctl", "-n", "hw.memsize"]),
        ("ncpu", ["sysctl", "-n", "hw.ncpu"]),
    ):
        try:
            info[key] = subprocess.check_output(cmd, text=True).strip()
        except Exception as exc:
            info[key] = f"unavailable: {exc}"
    return info


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """In-process Okapi BM25 inverted index. Elasticsearch is not installed."""

    def __init__(self, texts: list[str], k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        docs = [tokenize(t) for t in texts]
        self.N = len(docs)
        self.dl = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if self.N else 0.0
        df: Counter = Counter()
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            tf = Counter(doc)
            df.update(tf.keys())
            for term, freq in tf.items():
                postings[term].append((i, freq))
        self.postings = dict(postings)
        self.idf = {
            term: math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }
        self.norm = self.k1 * (1.0 - self.b + self.b * self.dl / self.avgdl)

    def structure_bytes(self) -> int:
        n_postings = sum(len(v) for v in self.postings.values())
        # Two int32s per posting plus a rough allowance for the term dictionary.
        return n_postings * 8 + sum(len(t) for t in self.postings) + 8 * len(self.idf)

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float64)
        for term in tokenize(query):
            posts = self.postings.get(term)
            if not posts:
                continue
            idf = self.idf[term]
            for doc_id, freq in posts:
                scores[doc_id] += idf * (freq * (self.k1 + 1.0)) / (freq + self.norm[doc_id])
        return scores


def t_critical_95(df: int) -> float:
    try:
        from scipy.stats import t as student_t
        return float(student_t.ppf(0.975, df))
    except Exception:
        # Two-sided 95% critical values. This study uses df = 4 (5 runs).
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 9: 2.262, 29: 2.045}
        return table.get(df, 1.96)


def across_runs(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    half = t_critical_95(len(a) - 1) * std / math.sqrt(len(a)) if len(a) > 1 else 0.0
    return {
        "n_runs": len(a),
        "values": [float(v) for v in a],
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "t_critical_95": t_critical_95(max(len(a) - 1, 1)),
    }


def load_saved_vectors():
    import faiss
    import pickle

    index = faiss.read_index(str(INDEX_SRC))
    if index.ntotal == 0:
        raise RuntimeError(f"Saved index is empty: {INDEX_SRC}")
    vectors = np.ascontiguousarray(index.reconstruct_n(0, index.ntotal).astype(np.float32))
    with open(META_SRC, "rb") as f:
        chunks = pickle.load(f)
    if len(chunks) != index.ntotal:
        raise RuntimeError(f"Metadata has {len(chunks)} chunks but the index has {index.ntotal}")
    return vectors, chunks, int(index.d), int(index.ntotal)


def try_train_ivfpq(vectors: np.ndarray, nbits: int):
    import faiss

    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFPQ(
        quantizer, vectors.shape[1], NLIST, PQ_M, nbits, faiss.METRIC_INNER_PRODUCT
    )
    index.train(vectors)
    return index


def build_flat(vectors: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_hnsw(vectors: np.ndarray):
    import faiss

    index = faiss.IndexHNSWFlat(vectors.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(vectors)
    return index


def as_query(query_vector: np.ndarray) -> np.ndarray:
    q = np.ascontiguousarray(query_vector, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    return q


def search_ids(index, query_vector: np.ndarray, k: int) -> np.ndarray:
    _, ids = index.search(as_query(query_vector), k)
    return ids[0]


def agreement(approx_ids: np.ndarray, exact_ids: np.ndarray) -> float:
    exact = {int(i) for i in exact_ids if int(i) >= 0}
    approx = {int(i) for i in approx_ids if int(i) >= 0}
    if not exact:
        return 0.0
    return len(exact & approx) / float(len(exact))


def time_searches(search_fn, n_reps: int) -> np.ndarray:
    samples = np.empty(n_reps, dtype=np.float64)
    for i in range(n_reps):
        t0 = time.perf_counter_ns()
        search_fn()
        t1 = time.perf_counter_ns()
        samples[i] = (t1 - t0) / 1e6
    return samples


def write_index(index, path: Path) -> int:
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    return path.stat().st_size


def probe_rss(index_path: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--rss-probe", str(index_path)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr[-2000:], "returncode": proc.returncode}
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def rss_probe_main(index_path: str) -> None:
    import faiss

    faiss.omp_set_num_threads(1)
    rss_import = rss_bytes()
    index = faiss.read_index(index_path)
    rss_load = rss_bytes()
    q = np.random.default_rng(0).standard_normal((1, index.d)).astype(np.float32)
    q /= np.linalg.norm(q) + 1e-12
    k = min(K, index.ntotal)
    for _ in range(100):
        index.search(q, k)
    rss_after = rss_bytes()
    print(json.dumps({
        "rss_after_import_bytes": rss_import,
        "rss_after_index_load_bytes": rss_load,
        "rss_after_100_searches_bytes": rss_after,
        "incremental_rss_after_load_bytes": rss_load - rss_import,
        "ntotal": int(index.ntotal),
        "d": int(index.d),
        "probe": "fresh Python process, faiss.read_index, then 100 searches; RSS via ps",
    }))


def percentile_95(samples: np.ndarray) -> float:
    return float(np.percentile(samples, 95, method="linear"))


def measure_config(name: str, search_one, exact_ids: list[np.ndarray], query_count: int) -> dict:
    for i in range(WARMUP):
        search_one(i % query_count)
    per_query = []
    samples = np.empty((query_count, REPS), dtype=np.float64)
    for qi in range(query_count):
        ids = search_one(qi)
        samples[qi] = time_searches(lambda qi=qi: search_one(qi), REPS)
        ids_after = search_one(qi)
        per_query.append({
            "query_index": qi,
            "agreement_at_5": agreement(ids, exact_ids[qi]),
            "retrieved_ids": [int(x) for x in ids],
            "flatip_ids": [int(x) for x in exact_ids[qi]],
            "latency_ms_mean": float(samples[qi].mean()),
            "latency_ms_p95": percentile_95(samples[qi]),
            "ids_stable_across_timed_loop": bool(np.array_equal(ids, ids_after)),
        })
    return {
        "config": name,
        "per_query": per_query,
        "latency_ms_mean": float(samples.mean()),
        "latency_ms_p95": percentile_95(samples),
        "agreement_at_5": float(np.mean([r["agreement_at_5"] for r in per_query])),
        "unstable_queries": sum(not r["ids_stable_across_timed_loop"] for r in per_query),
        "samples": samples,
    }


def gold_source_recall(retrieved_ids: list[int], relevant: list[str], chunks: list[dict]) -> float:
    from evaluation.metrics import recall_at_k

    ranked = []
    for i in retrieved_ids:
        if i < 0 or i >= len(chunks):
            continue
        ranked.append(chunks[i]["source_file"])
    return recall_at_k(relevant, ranked, K)


def run_retrieval(run_id: int, vectors, chunks, queries, query_vectors, ivf_bits: int, run_dir: Path) -> dict:
    import faiss

    faiss.omp_set_num_threads(1)
    np.random.seed(1000 + run_id)
    n_queries = len(queries)
    texts = [c["text"] for c in chunks]

    t0 = time.perf_counter()
    flat = build_flat(vectors)
    flat_build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    hnsw = build_hnsw(vectors)
    hnsw_build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ivf = try_train_ivfpq(vectors, ivf_bits)
    ivf.add(vectors)
    ivf.nprobe = 16
    ivf_build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    bm25 = BM25Index(texts)
    bm25_build_s = time.perf_counter() - t0

    exact_ids = [search_ids(flat, query_vectors[i], K) for i in range(n_queries)]
    flat_path = run_dir / "flatip.index"
    hnsw_path = run_dir / "hnsw.index"
    ivf_path = run_dir / "ivfpq.index"
    flat_bytes = write_index(flat, flat_path)
    hnsw_bytes = write_index(hnsw, hnsw_path)
    ivf_bytes = write_index(ivf, ivf_path)
    bm25_bytes = bm25.structure_bytes()

    def flat_search(qi: int):
        return search_ids(flat, query_vectors[qi], K)

    def ivf_search(qi: int):
        return search_ids(ivf, query_vectors[qi], K)

    def hnsw_search(qi: int):
        return search_ids(hnsw, query_vectors[qi], K)

    def hybrid_search(qi: int):
        ntotal = vectors.shape[0]
        scores, ids = flat.search(as_query(query_vectors[qi]), ntotal)
        dense = np.full(ntotal, -1e9, dtype=np.float64)
        valid = ids[0] >= 0
        dense[ids[0][valid]] = scores[0][valid]
        sparse = bm25.score(queries[qi]["question"])
        peak = float(sparse.max()) if sparse.size else 0.0
        sparse_n = sparse / peak if peak > 0 else sparse
        fused = ALPHA * dense + (1.0 - ALPHA) * sparse_n
        top = np.argpartition(-fused, K - 1)[:K]
        order = np.argsort(-fused[top])
        return top[order].astype(np.int64)

    jobs = [
        ("FlatIP", flat_search, flat_build_s, flat_bytes, flat_path, 0),
        (f"IVF-PQ (nbits={ivf_bits})", ivf_search, ivf_build_s, ivf_bytes, ivf_path, 0),
    ]
    for ef in HNSW_EF:
        jobs.append((f"HNSW ef={ef}", hnsw_search, hnsw_build_s, hnsw_bytes, hnsw_path, ef))
    jobs.append(("Hybrid FlatIP+BM25", hybrid_search, flat_build_s + bm25_build_s, flat_bytes + bm25_bytes, flat_path, 0))

    configs = []
    for name, fn, build_s, footprint, index_path, ef in jobs:
        if ef:
            hnsw.hnsw.efSearch = ef
        print(f"  run {run_id}  {name}  warmup {WARMUP} + {REPS} reps x {n_queries} queries", flush=True)
        measured = measure_config(name, fn, exact_ids, n_queries)
        samples = measured.pop("samples")
        np.save(run_dir / f"latency_{_slug(name)}.npy", samples)
        rss = probe_rss(index_path)
        for row in measured["per_query"]:
            qi = row["query_index"]
            row["question"] = queries[qi]["question"]
            row["gold_source_recall_at_5"] = gold_source_recall(
                row["retrieved_ids"], queries[qi]["relevant_sources"], chunks
            )
        measured.update({
            "build_seconds": build_s,
            "index_footprint_bytes": footprint,
            "faiss_file_bytes": index_path.stat().st_size,
            "bm25_structure_bytes": bm25_bytes if name.startswith("Hybrid") else 0,
            "runtime_memory": rss,
            "memory_procedure": {
                "index_footprint": "serialized FAISS file size; hybrid adds in-process BM25 posting-list bytes",
                "runtime_memory": "RSS of a fresh Python process after faiss.read_index minus RSS after import, then again after 100 searches",
                "threads": "OMP_NUM_THREADS=1 and faiss.omp_set_num_threads(1)",
            },
        })
        configs.append(measured)
        print(
            f"    mean {measured['latency_ms_mean']:.4f} ms  "
            f"p95 {measured['latency_ms_p95']:.4f} ms  "
            f"agreement {measured['agreement_at_5']:.4f}  "
            f"footprint {footprint / 1e6:.3f} MB",
            flush=True,
        )

    # Drop bulky per-query vectors from the compact run summary? Keep them; 37 rows is small.
    # Latency samples live in the npy files.
    payload = {
        "run": run_id,
        "configs": configs,
        "bm25_build_seconds": bm25_build_s,
        "hnsw_build_seconds": hnsw_build_s,
        "ivf_build_seconds": ivf_build_s,
        "flat_build_seconds": flat_build_s,
    }
    (run_dir / "retrieval.json").write_text(json.dumps(payload, indent=2))
    with open(run_dir / "per_query.jsonl", "w") as f:
        for cfg in configs:
            for row in cfg["per_query"]:
                f.write(json.dumps({"run": run_id, "config": cfg["config"], **row}) + "\n")
    return payload


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def embed_queries_main(out_path: str) -> None:
    """Torch-only process. Must not import faiss (duplicate OpenMP runtimes abort)."""
    queries = json.loads(TEST_SRC.read_text())
    from src.retrieval.embedder import Embedder

    embedder = Embedder()
    vectors = np.ascontiguousarray(embedder.embed([q["question"] for q in queries]))
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    print(f"saved {vectors.shape} query embeddings -> {path}")


def end_to_end_main() -> None:
    """Torch-only NLI timing. Retrieval latency is taken from the FAISS-only runs.

    Adding the two is the end-to-end figure. They are not timed in one process:
    importing FAISS and PyTorch together aborts with OMP Error #15.
    """
    import pickle
    from src.detection.nli_detector import NLIDetector

    print("Loading NLI model for end-to-end passes", flush=True)
    with open(META_SRC, "rb") as f:
        chunks = pickle.load(f)
    queries = json.loads(TEST_SRC.read_text())
    texts = [c["text"] for c in chunks]
    detector = NLIDetector()
    if detector.model is None:
        raise RuntimeError("NLI model failed to load; end-to-end verification was not measured.")
    device = "unknown"
    try:
        device = str(next(detector.model.model.parameters()).device)
    except Exception:
        pass
    warmup_pairs = [(texts[0], queries[0]["claims"][0]["text"]) for _ in range(8)]
    detector.model.predict(warmup_pairs, apply_softmax=True, show_progress_bar=False)
    print(f"NLI warmup done on {device}", flush=True)

    for run_id in range(1, N_RUNS + 1):
        run_dir = OUT / f"run_{run_id:02d}"
        payload = json.loads((run_dir / "retrieval.json").read_text())
        by_config = {}
        for cfg in payload["configs"]:
            rows = []
            retrieval_by_q = {row["query_index"]: row["latency_ms_mean"] for row in cfg["per_query"]}
            ids_by_q = {row["query_index"]: row["retrieved_ids"] for row in cfg["per_query"]}
            for qi, query in enumerate(queries):
                claims = [c["text"] for c in query["claims"]]
                ids = [i for i in ids_by_q[qi] if i >= 0]
                pairs = [(texts[i], claim) for claim in claims for i in ids]
                t0 = time.perf_counter_ns()
                if pairs:
                    detector.model.predict(pairs, apply_softmax=True, show_progress_bar=False)
                t1 = time.perf_counter_ns()
                nli_ms = (t1 - t0) / 1e6
                ret_ms = float(retrieval_by_q[qi])
                rows.append({
                    "query_index": qi,
                    "question": query["question"],
                    "n_pairs": len(pairs),
                    "retrieval_ms": ret_ms,
                    "nli_ms": nli_ms,
                    "end_to_end_ms": ret_ms + nli_ms,
                    "generation_ms": None,
                })
            e2e = np.array([r["end_to_end_ms"] for r in rows])
            nli = np.array([r["nli_ms"] for r in rows])
            summary = {
                "config": cfg["config"],
                "per_query": rows,
                "end_to_end_ms_mean": float(e2e.mean()),
                "end_to_end_ms_p95": percentile_95(e2e),
                "nli_ms_mean": float(nli.mean()),
                "nli_ms_p95": percentile_95(nli),
                "generation_measured": False,
                "composition": "per-query mean FAISS search latency from this run, plus NLI timed in a separate process",
            }
            by_config[cfg["config"]] = summary
            print(
                f"  e2e run {run_id} {cfg['config']}: mean {summary['end_to_end_ms_mean']:.1f} ms "
                f"(NLI {summary['nli_ms_mean']:.1f} ms)",
                flush=True,
            )
        out = {
            "run": run_id,
            "nli_device": device,
            "warmup_pairs": 8,
            "nli_model_loaded_once": True,
            "configs": by_config,
        }
        (run_dir / "end_to_end.json").write_text(json.dumps(out, indent=2))
    print("end-to-end passes written", flush=True)


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def write_summary(meta: dict, retrieval_runs: list[dict], e2e_runs: list[dict]) -> None:
    config_names = [c["config"] for c in retrieval_runs[0]["configs"]]
    rows = []
    for name in config_names:
        per = [next(c for c in run["configs"] if c["config"] == name) for run in retrieval_runs]
        mean_lat = across_runs([c["latency_ms_mean"] for c in per])
        p95 = across_runs([c["latency_ms_p95"] for c in per])
        agree = across_runs([c["agreement_at_5"] for c in per])
        gold = across_runs([
            float(np.mean([q["gold_source_recall_at_5"] for q in c["per_query"]]))
            for c in per
        ])
        footprint = per[0]["index_footprint_bytes"]
        rss = per[0]["runtime_memory"]
        rows.append({
            "config": name,
            "latency_ms": mean_lat,
            "p95_ms": p95,
            "agreement_at_5": agree,
            "gold_source_recall_at_5": gold,
            "index_footprint_bytes": footprint,
            "index_footprint_mb": footprint / (1024 * 1024),
            "runtime_memory": rss,
            "build_seconds_mean": float(np.mean([c["build_seconds"] for c in per])),
        })

    e2e_rows = []
    if e2e_runs and "error" not in e2e_runs[0]:
        for name in config_names:
            vals = [run["configs"][name]["end_to_end_ms_mean"] for run in e2e_runs]
            p95s = [run["configs"][name]["end_to_end_ms_p95"] for run in e2e_runs]
            nli = [run["configs"][name]["nli_ms_mean"] for run in e2e_runs]
            e2e_rows.append({
                "config": name,
                "end_to_end_ms": across_runs(vals),
                "end_to_end_p95_ms": across_runs(p95s),
                "nli_ms": across_runs(nli),
            })

    summary = {
        "meta": meta,
        "retrieval": rows,
        "end_to_end": e2e_rows,
        "generation": {
            "measured": False,
            "reason": "Ollama is not running on this machine, so generation latency was not measured and was not copied from the draft.",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    def cell(stat: dict) -> str:
        return f"{stat['mean']:.4f} ± {stat['std']:.4f}"

    lines = [
        "# TrustRAG-Opt repeated measurement",
        "",
        "These numbers were measured from the saved index and the saved 37-question test set.",
        "They are not the 9,800-chunk / 400-query figures in the draft.",
        "",
        "## What was confirmed",
        "",
        f"- Runs: {meta['n_runs']}",
        f"- Queries per run: {meta['n_queries']} (the existing `evaluation/test_set.json`)",
        f"- The same {meta['n_queries']} queries were reused, unchanged, across every run and every configuration.",
        "- There is no 400-query evaluation set in this repository, so a 400-query reuse cannot be confirmed.",
        f"- Warm-up: {meta['warmup_searches_per_config_per_run']} discarded searches per configuration per run, then {meta['timed_repetitions_per_query']} timed searches per query.",
        "- Query embeddings were computed once and reused. Timed retrieval is index search only.",
        "- Approximate indexes were rebuilt at the start of every run.",
        f"- Corpus actually measured: {meta['n_chunks']} chunks, dimension {meta['dimension']}.",
        "",
        "## Retrieval",
        "",
        "Agreement is the fraction of the FlatIP top-5 ids that the configuration also returned.",
        "Latency is in milliseconds. The ± value is the sample standard deviation across runs.",
        "The 95% confidence interval of the mean is in `summary.json`.",
        "",
        "| Configuration | Agreement@5 | Mean latency (ms) | p95 latency (ms) | Index footprint (MB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {cell(row['agreement_at_5'])} | {cell(row['latency_ms'])} | "
            f"{cell(row['p95_ms'])} | {row['index_footprint_mb']:.3f} |"
        )
    lines += [
        "",
        "p95 is computed inside each run over all timed searches (queries × repetitions), then the mean and standard deviation of those five run-level percentiles are reported.",
        "",
        "### Memory procedure",
        "",
        "- Index footprint: size of the FAISS index written with `faiss.write_index`. The same write path is used for every configuration. Hybrid adds the byte size of the in-process BM25 postings.",
        "- Runtime memory: a fresh Python process loads only that index file. RSS is read with `ps -o rss` after import, after `faiss.read_index`, and after 100 searches. Incremental RSS is load minus import.",
        "- FAISS was restricted to one thread (`OMP_NUM_THREADS=1`).",
        "",
        "| Configuration | Footprint (MB) | Incremental RSS after load (MB) | RSS after 100 searches (MB) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        mem = row["runtime_memory"]
        if "incremental_rss_after_load_bytes" in mem:
            inc = mem["incremental_rss_after_load_bytes"] / (1024 * 1024)
            after = mem["rss_after_100_searches_bytes"] / (1024 * 1024)
            lines.append(f"| {row['config']} | {row['index_footprint_mb']:.3f} | {inc:.3f} | {after:.3f} |")
        else:
            lines.append(f"| {row['config']} | {row['index_footprint_mb']:.3f} | n/a | n/a |")
    lines += [
        "",
        "HNSW footprint is one index shared by every `efSearch` setting. `efSearch` is a query-time parameter.",
        "",
        "## End-to-end",
        "",
        "End-to-end is the sum of that run's mean search latency and NLI verification of the question's labelled claims against the retrieved chunks.",
        "Search and NLI were timed in separate processes. Loading FAISS and PyTorch together aborts with OMP Error #15.",
        "Generation is omitted because Ollama is not running. The draft's 5.20 s generation figure was not reused.",
        "",
    ]
    if not e2e_rows:
        lines.append("End-to-end NLI was not measured. See `summary.json`.")
    else:
        lines += [
            "| Configuration | End-to-end mean (ms) | End-to-end p95 (ms) | NLI mean (ms) |",
            "|---|---:|---:|---:|",
        ]
        for row in e2e_rows:
            lines.append(
                f"| {row['config']} | {cell(row['end_to_end_ms'])} | {cell(row['end_to_end_p95_ms'])} | {cell(row['nli_ms'])} |"
            )
        lines.append("")
        lines.append(f"NLI device: {e2e_runs[0].get('nli_device', 'unknown')}. The model was loaded once, warmed with 8 pairs, then each run repeated the same queries.")
    lines += [
        "",
        "## Deviations from the draft protocol",
        "",
        meta["deviations"],
        "",
        "Raw per-query rows are in `run_XX/per_query.jsonl`. Every timed latency sample is in `run_XX/latency_*.npy`.",
        "The earlier Review 2 file `evaluation/results/review2_report.json` was not modified.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines))


def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "run.log"
    log_f = open(log_path, "w", buffering=1)
    class Tee:
        def write(self, data):
            sys.__stdout__.write(data)
            log_f.write(data)
        def flush(self):
            sys.__stdout__.flush()
            log_f.flush()
    sys.stdout = Tee()
    sys.stderr = Tee()

    query_embed_path = OUT / "query_embeddings.npy"
    if not query_embed_path.exists():
        print("Embedding the saved queries in a torch-only process", flush=True)
        embed_proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--embed-queries", str(query_embed_path)],
            env=os.environ.copy(),
            check=False,
        )
        if embed_proc.returncode != 0:
            raise RuntimeError(f"Query embedding process failed with code {embed_proc.returncode}")
    else:
        print(f"Reusing cached query embeddings {query_embed_path}", flush=True)
    query_vectors = np.load(query_embed_path)

    print("Loading saved FlatIP index (read-only)", flush=True)
    vectors, chunks, dim, ntotal = load_saved_vectors()
    norms = np.linalg.norm(vectors, axis=1)
    queries = json.loads(TEST_SRC.read_text())
    if len(queries) != len(query_vectors):
        raise RuntimeError("Query embedding count does not match the test set")
    print(f"Chunks {ntotal}  dim {dim}  queries {len(queries)}  mean L2 norm {norms.mean():.4f}", flush=True)

    ivf_bits = PQ_BITS_PAPER
    ivf_error = None
    print(f"Attempting paper IVF-PQ: nlist={NLIST}, m={PQ_M}, nbits={PQ_BITS_PAPER}", flush=True)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trial = try_train_ivfpq(vectors, PQ_BITS_PAPER)
            del trial
            for w in caught:
                print(f"IVF warning: {w.message}", flush=True)
    except Exception as exc:
        ivf_error = f"{type(exc).__name__}: {exc}"
        ivf_bits = 7
        print(f"Paper IVF-PQ failed: {ivf_error}", flush=True)
        print(f"Measuring IVF-PQ with nbits={ivf_bits} instead. nlist, m, and nprobe are unchanged.", flush=True)

    meta = {
        "timestamp_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_runs": N_RUNS,
        "n_queries": len(queries),
        "n_chunks": ntotal,
        "dimension": dim,
        "k": K,
        "warmup_searches_per_config_per_run": WARMUP,
        "timed_repetitions_per_query": REPS,
        "queries_reused_across_runs": True,
        "query_file": str(TEST_SRC.relative_to(ROOT)),
        "query_file_sha256": sha256(TEST_SRC),
        "saved_index_sha256": sha256(INDEX_SRC),
        "saved_index_modified": False,
        "mean_vector_l2_norm": float(norms.mean()),
        "faiss_version": faiss.__version__,
        "numpy_version": np.__version__,
        "machine": machine_info(),
        "threads": 1,
        "ivf_paper_nbits": PQ_BITS_PAPER,
        "ivf_measured_nbits": ivf_bits,
        "ivf_paper_error": ivf_error,
        "hybrid": "in-process Okapi BM25 inverted index, alpha=0.7, k1=1.5, b=0.75. Elasticsearch was not used.",
        "generation_measured": False,
        "deviations": (
            f"The draft describes 9,800 chunks and 400 queries. This repository's saved index has {ntotal} chunks "
            f"and evaluation/test_set.json has {len(queries)} questions. Those 37 questions were reused for every run. "
            f"IVF-PQ with {PQ_BITS_PAPER} bits did not train"
            + (f" ({ivf_error}). Measured IVF-PQ uses {ivf_bits} bits; nlist={NLIST}, m={PQ_M}, nprobe=16." if ivf_error
               else f". Measured IVF-PQ uses {ivf_bits} bits.")
            + " Hybrid BM25 is an in-process inverted index because Elasticsearch is not part of this environment. "
            "Latency was timed with perf_counter_ns around each search, which is what makes a 95th percentile defined; "
            "the draft's timeit loop would have collapsed those repetitions into one average. "
            "End-to-end excludes generation. "
            "FAISS and PyTorch were not loaded in the same process: doing so aborts with OMP Error #15 "
            "because each ships its own OpenMP runtime. Query embedding and NLI ran in torch-only processes; "
            "index search ran in a FAISS-only process. End-to-end latency is the sum of those two measurements."
        ),
    }
    (OUT / "protocol.json").write_text(json.dumps(meta, indent=2))

    retrieval_runs = []
    for run_id in range(1, N_RUNS + 1):
        run_dir = OUT / f"run_{run_id:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== retrieval run {run_id}/{N_RUNS} ===", flush=True)
        retrieval_runs.append(run_retrieval(run_id, vectors, chunks, queries, query_vectors, ivf_bits, run_dir))

    print("\n=== end-to-end (torch-only process) ===", flush=True)
    e2e_proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--end-to-end"],
        env=os.environ.copy(),
        check=False,
    )
    if e2e_proc.returncode != 0:
        e2e_runs = [{"error": f"end-to-end process exited {e2e_proc.returncode}"}]
        print(e2e_runs[0]["error"], flush=True)
        (OUT / "end_to_end_error.txt").write_text(e2e_runs[0]["error"])
    else:
        e2e_runs = []
        for run_id in range(1, N_RUNS + 1):
            e2e_path = OUT / f"run_{run_id:02d}" / "end_to_end.json"
            if e2e_path.exists():
                e2e_runs.append(json.loads(e2e_path.read_text()))
        if not e2e_runs:
            e2e_runs = [{"error": "end-to-end process exited 0 but wrote no results"}]

    meta["timestamp_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / "protocol.json").write_text(json.dumps(meta, indent=2))
    write_summary(meta, retrieval_runs, e2e_runs)
    print(f"\nWrote {OUT / 'summary.md'}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--rss-probe":
        rss_probe_main(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == "--embed-queries":
        embed_queries_main(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "--end-to-end":
        end_to_end_main()
    else:
        main()
