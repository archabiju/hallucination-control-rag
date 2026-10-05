"""
Review-2 evaluation — runs the REAL retrieval + NLI + BERTScore + cosine stack on
evaluation/test_set.json and writes every number the paper's Tables 2-4 report.

    python evaluation/run_review2_eval.py

Nothing here is simulated: each claim is scored by the same modules the live pipeline
uses. Output: evaluation/results/review2_report.json, review2_tables.md, review2_*.png.
No LLM / Ollama needed (the claims are given, not generated).
"""
import sys, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import RAGHallucinationPipeline
from src.detection.evidence_windows import EvidenceWindows
from src.explainability.xai_layer import generate_xai_report
from evaluation.metrics import (precision_recall_f1, auroc, recall_at_k, mrr, ndcg_at_k,
                                plot_roc_curve, plot_support_score_distribution)

OUT = ROOT / "evaluation" / "results"
OUT.mkdir(parents=True, exist_ok=True)
TEST = json.loads((ROOT / "evaluation" / "test_set.json").read_text())

p = RAGHallucinationPipeline()
p.index_documents()
p._load_models()
if not p.faiss_store.is_loaded():
    p.faiss_store.load()

rows, retr = [], []
t0 = time.time()
for item in TEST:
    qtext, claims = item["question"], item["claims"]
    texts = [c["text"] for c in claims]
    qv = p.embedder.embed_query(qtext)
    chunks = p.faiss_store.search(qv, k=p.top_k)
    ranked_src = [c["source_file"] for c in chunks]
    rel = item["relevant_sources"]
    retr.append({
        "recall@5": recall_at_k(rel, ranked_src, 5),
        "mrr": mrr(rel, ranked_src),
        "ndcg@5": ndcg_at_k(rel, ranked_src, 5),
        "hit@5": float(bool(set(rel) & set(ranked_src[:5]))),
    })
    cv = p.embedder.embed([c["text"] for c in chunks])
    xai = generate_xai_report(query=qtext, chunks=chunks, query_vector=qv, chunk_vectors=cv,
                              sentences=texts, embedder=p.embedder)
    # chunk-level baseline (whole 512-token chunk as premise)
    nli_c = p.nli_detector.score_sentences(texts, chunks, xai["attributions"])
    bs_c = p.bert_detector.score_sentences(texts, chunks)
    verd_c = p.hall_engine.analyze(texts, nli_c, bs_c, xai["attributions"])
    # sentence-level evidence windows (the shipped configuration)
    ev = EvidenceWindows(chunks, p.embedder, p.win_size)
    nli = p.nli_detector.score_sentences(texts, chunks, xai["attributions"], evidence=ev, top_n=p.win_top_n)
    bs = p.bert_detector.score_sentences(texts, chunks, evidence=ev, top_n=p.win_top_n)
    verd = p.hall_engine.analyze(texts, nli, bs, xai["attributions"])
    prov = xai["provenance"]["provenance_score"]
    for c, v, vc in zip(claims, verd, verd_c):
        rows.append({"q": qtext, "claim": c["text"], "y": int(c["hallucinated"]),
                     "nli": v.nli_score, "bert": v.bertscore_f1, "cos": v.cosine_score,
                     "support": v.support_score, "cls": v.classification, "prov": prov,
                     "support_chunk": vc.support_score, "nli_chunk": vc.nli_score,
                     "bert_chunk": vc.bertscore_f1})
    print(f"  {qtext[:60]:60s} done")
elapsed = time.time() - t0

y = np.array([r["y"] for r in rows])
S = {k: np.array([r[k] for r in rows]) for k in ("nli", "bert", "cos", "support", "support_chunk")}
w = (p.w_nli, p.w_bert, p.w_cos)

def combo(wn, wb, wc):
    tot = wn + wb + wc
    return (wn * S["nli"] + wb * S["bert"] + wc * S["cos"]) / tot

def evaluate(score, thr=0.5):
    """score = support (high = grounded). Positive class = hallucinated."""
    pred = (score < thr).astype(int)
    m = precision_recall_f1(y.tolist(), pred.tolist())
    m["auroc"] = auroc(y.tolist(), (1 - score).tolist())
    # threshold-free fairness for single signals: best-F1 operating point
    best = max(((precision_recall_f1(y.tolist(), (score < t).astype(int).tolist()), t)
                for t in np.arange(0.05, 0.96, 0.01)), key=lambda z: z[0]["f1"])
    m["best_f1"], m["best_f1_thr"] = best[0]["f1"], round(float(best[1]), 2)
    return m

configs = {
    "NLI only": S["nli"], "BERTScore only": S["bert"], "Cosine only": S["cos"],
    "NLI + BERTScore": combo(w[0], w[1], 0), "TrustRAG (all three)": S["support"],
}
baseline_chunk = evaluate(S["support_chunk"])   # same three signals, whole-chunk premises
ablation = {k: evaluate(v) for k, v in configs.items()}
overall = ablation["TrustRAG (all three)"]

cls_dist = {}
for name, lab in (("Supported claims", 0), ("Hallucinated claims", 1)):
    sel = [r["cls"] for r in rows if r["y"] == lab]
    n = max(len(sel), 1)
    cls_dist[name] = {k: round(100 * sel.count(k) / n, 1)
                      for k in ("SUPPORTED", "WEAKLY_SUPPORTED", "HALLUCINATED")}

# response-level blocking: block question if > 50% of its claims are HALLUCINATED
blk = {0: [], 1: []}
for item in TEST:
    cs = [r for r in rows if r["q"] == item["question"]]
    frac = sum(r["cls"] == "HALLUCINATED" for r in cs) / len(cs)
    mix = np.mean([r["y"] for r in cs])
    blk[1 if mix >= 0.5 else 0].append(frac > 0.5)

report = {
    "meta": {"questions": len(TEST), "claims": len(rows), "hallucinated": int(y.sum()),
             "supported": int((1 - y).sum()), "corpus_chunks": p.faiss_store.num_chunks,
             "weights_nli_bert_cos": w, "thresholds": [p.high_thr, p.medium_thr],
             "eval_seconds": round(elapsed, 1), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
    "overall": overall,
    "ablation": ablation,
    "chunk_level_baseline": baseline_chunk,
    "tri_state_distribution_pct": cls_dist,
    "retrieval": {k: round(float(np.mean([r[k] for r in retr])), 4) for k in retr[0]},
    "mean_provenance": round(float(np.mean([r["prov"] for r in rows])), 4),
    "mean_bertscore_f1_supported": round(float(S["bert"][y == 0].mean()), 4),
    "per_claim": rows,
}
(OUT / "review2_report.json").write_text(json.dumps(report, indent=2))

plot_roc_curve(y.tolist(), (1 - S["support"]).tolist(), str(OUT / "review2_roc.png"))
plot_support_score_distribution(S["support"].tolist(), y.tolist(), str(OUT / "review2_score_dist.png"))

md = ["# Review 2 — measured results\n",
      f"{len(TEST)} questions / {len(rows)} labelled claims ({int(y.sum())} hallucinated) · "
      f"{p.faiss_store.num_chunks}-chunk index · weights {w} · thresholds {p.high_thr}/{p.medium_thr}\n",
      "## Overall (positive class = hallucinated)\n", "| P | R | F1 | AUC-ROC | Acc |", "|---|---|---|---|---|",
      f"| {overall['precision']} | {overall['recall']} | {overall['f1']} | {overall['auroc']} | {overall['accuracy']} |\n",
      "## Ablation\n", "| Config | P | R | F1 @0.50 | AUC | best-F1 (thr) |", "|---|---|---|---|---|---|"]
for k, m in ablation.items():
    md.append(f"| {k} | {m['precision']} | {m['recall']} | {m['f1']} | {m['auroc']} | {m['best_f1']} ({m['best_f1_thr']}) |")
bc = baseline_chunk
md += ["\n## Evidence windows vs whole-chunk premises (all three signals)\n", "| Premise | P | R | F1 @0.50 | AUC | best-F1 (thr) |", "|---|---|---|---|---|---|",
       f"| whole 512-token chunk | {bc['precision']} | {bc['recall']} | {bc['f1']} | {bc['auroc']} | {bc['best_f1']} ({bc['best_f1_thr']}) |",
       f"| 2-sentence windows (shipped) | {overall['precision']} | {overall['recall']} | {overall['f1']} | {overall['auroc']} | {overall['best_f1']} ({overall['best_f1_thr']}) |"]
md += ["\n## Tri-state distribution (% of claims)\n", "| Claim type | Supported | Weak | Hallucinated |", "|---|---|---|---|"]
for k, d in cls_dist.items():
    md.append(f"| {k} | {d['SUPPORTED']} | {d['WEAKLY_SUPPORTED']} | {d['HALLUCINATED']} |")
r = report["retrieval"]
md += ["\n## Retrieval (doc-level, k=5)\n", f"Recall@5 {r['recall@5']} · MRR {r['mrr']} · NDCG@5 {r['ndcg@5']} · Hit@5 {r['hit@5']}"]
(OUT / "review2_tables.md").write_text("\n".join(md))
print("\n".join(md))
