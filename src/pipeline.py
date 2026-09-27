"""
Pipeline — Full End-to-End Orchestration
==========================================
The main entry point that wires together all components:

    1. Ingestion  (load + chunk documents)
    2. Retrieval  (embed + FAISS index/search)
    3. Generation (local LLM via Ollama)
    4. XAI Layer  (cosine scores, attribution, provenance)
    5. Detection  (NLI + BERTScore + cosine aggregation)
    6. Controller (tri-state decision)

Usage:
    python src/pipeline.py              ← index documents in data/corpus/
    python src/pipeline.py query        ← interactive query mode
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.document_loader import load_corpus
from src.ingestion.chunker import chunk_pages, get_sentences
from src.retrieval.embedder import Embedder
from src.retrieval.faiss_store import FAISSStore
from src.generation.llm_generator import LLMGenerator
from src.explainability.xai_layer import generate_xai_report
from src.detection.nli_detector import NLIDetector
from src.detection.bertscore_detector import BERTScoreDetector
from src.detection.hallucination_engine import HallucinationEngine
from src.controller.decision_controller import DecisionController, ControlledResponse


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"[WARN] Config not found at {config_path}. Using defaults.")
        return {}
    with open(config_file, "r") as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Complete structured output from one pipeline invocation."""
    query:            str
    response:         Optional[ControlledResponse] = None
    xai_report:       Dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    raw_answer:       str = ""
    sentences:        List[str] = field(default_factory=list)
    verdicts:         list = field(default_factory=list)
    summary:          Dict[str, Any] = field(default_factory=dict)
    error:            Optional[str] = None

    def display(self) -> str:
        """Return a human-readable display string."""
        if self.error:
            return f"❌ ERROR: {self.error}"
        if self.response is None:
            return "No response generated."
        from src.controller.decision_controller import DecisionController
        ctrl = DecisionController()
        return ctrl.format_for_display(self.response)


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline class
# ─────────────────────────────────────────────────────────────────────────────

class RAGHallucinationPipeline:
    """
    Complete offline RAG pipeline with integrated hallucination detection.

    Lifecycle:
        pipeline = RAGHallucinationPipeline()
        pipeline.index_documents()           ← one-time indexing
        result = pipeline.query("...")       ← repeated use
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)

        # Extract config sections with defaults
        paths    = self.cfg.get("paths", {})
        ingest   = self.cfg.get("ingestion", {})
        ret      = self.cfg.get("retrieval", {})
        gen      = self.cfg.get("generation", {})
        nli      = self.cfg.get("nli", {})
        bs       = self.cfg.get("bertscore", {})
        det      = self.cfg.get("detection", {})
        ctrl     = self.cfg.get("controller", {})

        self.corpus_dir  = paths.get("corpus_dir", "data/corpus")
        self.index_dir   = paths.get("index_dir",  "data/index")
        self.chunk_size  = ingest.get("chunk_size", 512)
        self.chunk_overlap = ingest.get("chunk_overlap", 50)
        self.top_k       = ret.get("top_k", 5)
        self.emb_dim     = ret.get("embedding_dim", 384)
        self.emb_model   = ret.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.llm_url     = gen.get("ollama_base_url", "http://localhost:11434")
        self.llm_model   = gen.get("model_name", "mistral")
        self.llm_temp    = gen.get("temperature", 0.1)
        self.llm_tokens  = gen.get("max_tokens", 512)
        self.nli_model   = nli.get("model", "cross-encoder/nli-deberta-v3-small")
        self.bs_lang     = bs.get("lang", "en")
        self.bs_model    = bs.get("model_type", "distilbert-base-uncased")
        self.w_nli       = det.get("weight_nli", 0.45)
        self.w_bert      = det.get("weight_bertscore", 0.35)
        self.w_cos       = det.get("weight_cosine", 0.20)
        self.high_thr    = det.get("high_support_threshold", 0.75)
        self.medium_thr  = det.get("medium_support_threshold", 0.50)
        self.max_block   = ctrl.get("max_block_ratio", 0.50)
        self.cite_fmt    = ctrl.get("citation_format", "[{source}, p.{page}]")

        self._loaded = False

    def _load_models(self) -> None:
        """Lazy-load all models (called once on first use)."""
        if self._loaded:
            return

        print("\n" + "="*60)
        print("  Loading RAG Hallucination Detection Pipeline")
        print("="*60)

        self.embedder   = Embedder(self.emb_model)
        self.faiss_store = FAISSStore(self.index_dir, self.emb_dim)
        self.generator  = LLMGenerator(self.llm_url, self.llm_model,
                                        self.llm_temp, self.llm_tokens)
        self.nli_detector   = NLIDetector(self.nli_model)
        self.bert_detector  = BERTScoreDetector(self.bs_lang, self.bs_model)
        self.hall_engine    = HallucinationEngine(
            self.w_nli, self.w_bert, self.w_cos,
            self.high_thr, self.medium_thr
        )
        self.controller     = DecisionController(
            self.high_thr, self.medium_thr, self.max_block, self.cite_fmt
        )

        self._loaded = True
        print("="*60)
        print("  All models loaded. Pipeline ready.")
        print("="*60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # INDEXING
    # ─────────────────────────────────────────────────────────────────────────

    def index_documents(
        self,
        corpus_dir: Optional[str] = None,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """
        Load, chunk, embed, and index all documents in the corpus directory.

        Args:
            corpus_dir: Override the configured corpus directory.
            force_reindex: If True, rebuild even if an existing index exists.

        Returns:
            Dict with indexing statistics.
        """
        self._load_models()

        corpus_dir = corpus_dir or self.corpus_dir

        # Check if existing index can be reused
        if not force_reindex and self.faiss_store.load():
            print(f"[INFO] Reusing existing index ({self.faiss_store.num_chunks} chunks).")
            return {
                "status": "reused",
                "num_chunks": self.faiss_store.num_chunks,
                "corpus_dir": corpus_dir,
            }

        print(f"\n[INFO] Indexing corpus: {corpus_dir}")

        # Step 1: Load documents
        pages = load_corpus(corpus_dir)
        if not pages:
            return {"status": "error", "message": "No documents found in corpus."}

        # Step 2: Chunk pages
        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return {"status": "error", "message": "No chunks generated."}

        # Step 3: Embed chunks
        embeddings = self.embedder.embed_chunks(chunks, show_progress=True)

        # Step 4: Build FAISS index
        self.faiss_store.build(chunks, embeddings)

        stats = {
            "status": "indexed",
            "num_pages":  len(pages),
            "num_chunks": len(chunks),
            "corpus_dir": corpus_dir,
        }
        print(f"\n[INFO] Indexing complete: {stats}")
        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # QUERY
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        user_query: str,
        top_k: Optional[int] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline for a user query.

        Steps:
          1. Embed query → search FAISS → retrieve top-K chunks
          2. Generate answer using local LLM
          3. Split answer into sentences
          4. XAI layer: compute all explainability signals
          5. Detect hallucinations (NLI + BERTScore + cosine)
          6. Controller: apply tri-state decision logic

        Args:
            user_query: The user's question string.
            top_k: Override the configured top_k retrieval count.

        Returns:
            PipelineResult with full structured output.
        """
        self._load_models()

        if not self.faiss_store.is_loaded():
            if not self.faiss_store.load():
                return PipelineResult(
                    query=user_query,
                    error="Knowledge base is empty. Please index documents first."
                )

        top_k = top_k or self.top_k
        result = PipelineResult(query=user_query)

        try:
            # ── Step 1: Retrieve ──────────────────────────────────────────
            print(f"\n[PIPELINE] Query: {user_query[:80]}")
            print("[PIPELINE] Step 1: Embedding query and retrieving chunks...")

            query_vec = self.embedder.embed_query(user_query)
            retrieved = self.faiss_store.search(query_vec, k=top_k)
            result.retrieved_chunks = retrieved

            print(f"[PIPELINE] Retrieved {len(retrieved)} chunks.")
            for r in retrieved:
                print(f"  • [{r['rank']}] {r['source_file']} p.{r['page_num']} "
                      f"(sim={r['similarity_score']:.3f})")

            # ── Step 2: Generate ──────────────────────────────────────────
            print("[PIPELINE] Step 2: Generating answer via local LLM...")
            gen_result   = self.generator.generate(user_query, retrieved)
            raw_answer   = gen_result["answer"]
            result.raw_answer = raw_answer
            print(f"[PIPELINE] Raw answer: {raw_answer[:120]}...")

            # ── Step 3: Sentence splitting ────────────────────────────────
            print("[PIPELINE] Step 3: Splitting answer into claims...")
            sentences = get_sentences(raw_answer)
            sentences = [s for s in sentences if len(s.split()) >= 3]  # Filter fragments
            result.sentences = sentences
            print(f"[PIPELINE] {len(sentences)} claims to verify.")

            if not sentences:
                sentences = [raw_answer]

            # ── Step 4: XAI Layer ─────────────────────────────────────────
            print("[PIPELINE] Step 4: Computing XAI explainability signals...")
            chunk_texts = [c["text"] for c in retrieved]
            chunk_vecs  = self.embedder.embed(chunk_texts)

            xai_report = generate_xai_report(
                query=user_query,
                chunks=retrieved,
                query_vector=query_vec,
                chunk_vectors=chunk_vecs,
                sentences=sentences,
                embedder=self.embedder,
            )
            result.xai_report = xai_report

            print(f"[PIPELINE] Provenance score: "
                  f"{xai_report['provenance']['provenance_score']:.2%}")

            # ── Step 5: Hallucination Detection ───────────────────────────
            print("[PIPELINE] Step 5: Running hallucination detection...")
            attributions = xai_report["attributions"]

            nli_results  = self.nli_detector.score_sentences(
                sentences, retrieved, attributions
            )
            bert_results = self.bert_detector.score_sentences(sentences, retrieved)

            verdicts = self.hall_engine.analyze(
                sentences, nli_results, bert_results, attributions
            )
            summary  = self.hall_engine.summarize(verdicts)
            result.verdicts = verdicts
            result.summary  = summary

            print(f"[PIPELINE] Support score: {summary['overall_support_score']:.2%} | "
                  f"Hallucination ratio: {summary['hallucination_ratio']:.2%}")

            # ── Step 6: Decision Controller ───────────────────────────────
            print("[PIPELINE] Step 6: Applying tri-state decision controller...")
            prov_score = xai_report.get("provenance", {}).get("provenance_score", 0.0)
            response = self.controller.decide(
                verdicts,
                summary,
                raw_answer,
                retrieved_chunks=retrieved,
                provenance_score=prov_score,
            )
            result.response = response

            print(f"[PIPELINE] Decision state: {response.state}")

        except Exception as e:
            import traceback
            result.error = str(e)
            print(f"[ERROR] Pipeline failed: {e}")
            traceback.print_exc()

        return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = RAGHallucinationPipeline()

    # Index documents
    stats = pipeline.index_documents()
    print(f"\nIndex stats: {stats}")

    if len(sys.argv) > 1 and sys.argv[1] == "query":
        # Interactive mode
        print("\n🔍 Interactive Query Mode (type 'quit' to exit)\n")
        while True:
            try:
                q = input("Your question: ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if not q:
                    continue

                result = pipeline.query(q)
                print("\n" + "=" * 60)
                print(result.display())
                print("=" * 60 + "\n")

            except KeyboardInterrupt:
                print("\nExiting.")
                break
    else:
        # Demo query
        demo_query = (
            "What is hallucination in large language models and how does "
            "RAG help reduce it?"
        )
        print(f"\nRunning demo query: {demo_query}")
        result = pipeline.query(demo_query)
        print("\n" + "=" * 60)
        print(result.display())
        print("=" * 60)
