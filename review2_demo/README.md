# Review 2 — Demo Materials

Everything needed to showcase this project at Review 2, in one place.

## Important — the corpus grew

12 new domain-knowledge documents were added to `data/corpus/` (covering RAG, embeddings,
FAISS, NLI, BERTScore, XAI, hallucination, tri-state control, chunking, prompt engineering,
Ollama/LLMs, and evaluation metrics) so the system can answer far more than the original 5
demo questions — 82 chunks → 94 chunks, 9 documents → 21.

**Before your live demo: click "🔨 Build Document Index" in the Streamlit sidebar once**
(or run the pipeline's indexing step) so the real FAISS index picks up the new documents too.
I couldn't rebuild `data/index/faiss.index` myself through this connection, so this one click
on your machine is what actually activates them in the live system. The browser backup below
already has them baked in — no action needed there.

## Files in this folder

- **hcr_ide_viewer.html** + **hcr_code_bundle.json** — IDE-style code viewer: file tree, syntax-highlighted source for all 17 core files, plus an honest audit/review tab.
- **hcr_chat_backup.html** + **hcr_chunks.json** — Browser backup of the real Streamlit chat UI (`ui/app.py`). Same corpus (82 real chunks, 8 domain documents), same 0.45/0.35/0.20 scoring formula, computed live in the browser. Use this if Ollama/Streamlit can't run live in the review room.
- **review2_demo_playbook.html** — Run guide, demo script (which questions to ask, in what order, what each proves), fallback plan, and anticipated panel Q&A.

Open any `.html` file directly in a browser — no server needed.

## Live (hosted) versions

These same pages are also published on claude.ai, viewable from any device with a link, and stay linked to this session so they can be re-published with updates:

- Code viewer: https://claude.ai/artifact/GrGsyxyaQP7teT9qW4jdEU
- Chat UI backup: https://claude.ai/artifact/SXpKvEp3SuptNv2bMWBJ8x
- Demo playbook: https://claude.ai/artifact/AjZywwx4xL4ymd9dvphegk

## Recommended flow for the review

1. Run the **real** `ui/app.py` live (see playbook, Section 1) — this is the primary demo.
2. Open the **code viewer** if the panel asks to see the implementation.
3. Fall back to the **chat UI backup** only if Ollama/Streamlit can't run in the room.
