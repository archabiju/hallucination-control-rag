"""
TrustRAG end-to-end demo (works with or without Ollama).

    python demo/demo_run.py

Part A  Live query through the full pipeline (retrieve -> generate -> XAI -> detect -> control).
        Without Ollama the generator returns the retrieved passage (clearly labelled fallback).
Part B  Tri-state demonstration. For sample questions from evaluation/test_set.json the same
        verification stack scores three candidate answers built from the hand-labelled claims:
          1. grounded claims only       -> expected ANSWER_WITH_CITATIONS
          2. grounded + fabricated      -> expected PARTIAL_ANSWER
          3. fabricated claims only     -> expected BLOCKED
        No numbers are hard-coded: every score comes from the real NLI, BERTScore and cosine modules.

Outputs: demo/outputs/sample_outputs.json and demo/outputs/sample_outputs.txt
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):          # Windows consoles default to cp1252 and reject emojis
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

from src.pipeline import RAGHallucinationPipeline
from src.ingestion.chunker import get_sentences
from src.detection.evidence_windows import EvidenceWindows
from src.explainability.xai_layer import generate_xai_report

OUT = ROOT / "demo" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

LIVE_QUERIES = [
    "What is hallucination in large language models and how does RAG help reduce it?",
]
N_TRISTATE_QUESTIONS = 2

_lines = []


def log(text=""):
    print(text)
    _lines.append(text)


def verify_answer(p, question, answer):
    """Steps 1, 3-6 of RAGHallucinationPipeline.query for a supplied answer text."""
    qv = p.embedder.embed_query(question)
    chunks = p.faiss_store.search(qv, k=p.top_k)
    sentences = [s for s in get_sentences(answer) if len(s.split()) >= 3] or [answer]
    cv = p.embedder.embed([c["text"] for c in chunks])
    xai = generate_xai_report(query=question, chunks=chunks, query_vector=qv,
                              chunk_vectors=cv, sentences=sentences, embedder=p.embedder)
    ev = EvidenceWindows(chunks, p.embedder, p.win_size) if p.use_windows else None
    nli = p.nli_detector.score_sentences(sentences, chunks, xai["attributions"],
                                         evidence=ev, top_n=p.win_top_n)
    bs = p.bert_detector.score_sentences(sentences, chunks, evidence=ev, top_n=p.win_top_n)
    verdicts = p.hall_engine.analyze(sentences, nli, bs, xai["attributions"])
    summary = p.hall_engine.summarize(verdicts)
    prov = xai["provenance"]["provenance_score"]
    resp = p.controller.decide(verdicts, summary, answer, retrieved_chunks=chunks,
                               provenance_score=prov)
    return verdicts, resp


def show_verdicts(verdicts):
    for v in verdicts:
        log(f"   [{v.classification:16s}] support={v.support_score:.2f} "
            f"(NLI {v.nli_score:.2f} | BERT {v.bertscore_f1:.2f} | cos {v.cosine_score:.2f})")
        log(f"      \"{v.sentence[:110]}\"")


def main():
    p = RAGHallucinationPipeline()
    stats = p.index_documents()
    p._load_models()
    log(f"Index: {stats}")
    results = {"index": stats, "live": [], "tristate": []}

    log("\n" + "=" * 72)
    log("PART A - LIVE QUERY (full pipeline)")
    log("=" * 72)
    for q in LIVE_QUERIES:
        r = p.query(q)
        log(f"\nQuestion: {q}")
        if r.error:
            log(f"ERROR: {r.error}")
            continue
        log(r.display())
        log("\nPer-sentence verdicts:")
        show_verdicts(r.verdicts)
        results["live"].append({"question": q, "state": r.response.state,
                                "overall_score": r.response.overall_score,
                                "response": r.response.to_dict()})

    log("\n" + "=" * 72)
    log("PART B - TRI-STATE DEMONSTRATION (answers built from labelled claims)")
    log("=" * 72)
    test = json.loads((ROOT / "evaluation" / "test_set.json").read_text(encoding="utf-8"))
    for item in test[:N_TRISTATE_QUESTIONS]:
        q = item["question"]
        good = [c["text"] for c in item["claims"] if not c["hallucinated"]]
        bad = [c["text"] for c in item["claims"] if c["hallucinated"]]
        scenarios = [("1. Grounded answer", good),
                     ("2. Grounded + fabricated", good[:2] + bad[:2]),
                     ("3. Fabricated answer", bad)]
        for name, claims in scenarios:
            if not claims:
                continue
            verdicts, resp = verify_answer(p, q, " ".join(claims))
            log(f"\nQ: {q}\n{name}  ->  {resp.state}  (grounding {resp.overall_score:.0%})")
            show_verdicts(verdicts)
            results["tristate"].append({"question": q, "scenario": name, "state": resp.state,
                                        "overall_score": resp.overall_score,
                                        "verdicts": [v.to_dict() for v in verdicts]})

    (OUT / "sample_outputs.json").write_text(json.dumps(results, indent=2, default=str),
                                             encoding="utf-8")
    (OUT / "sample_outputs.txt").write_text("\n".join(_lines), encoding="utf-8")
    log(f"\nSaved: {OUT / 'sample_outputs.json'}\n       {OUT / 'sample_outputs.txt'}")


if __name__ == "__main__":
    main()
