"""
Decision Controller — Controller Layer
=========================================
Takes the hallucination engine's verdict and decides what to output to the user.

Tri-State Output Logic:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  overall_support_score >= HIGH_THRESHOLD (0.75)                      │
  │  → STATE 1: Answer with inline citations                             │
  │    "According to [source.pdf, p.3]: ..."                             │
  │                                                                       │
  │  overall_support_score >= MEDIUM_THRESHOLD (0.50)                    │
  │  → STATE 2: Partial answer with flagged sentences                    │
  │    Supported sentences shown normally                                 │
  │    Weak/hallucinated sentences tagged with ⚠ WARNING                 │
  │                                                                       │
  │  overall_support_score < MEDIUM_THRESHOLD  OR                        │
  │  hallucination_ratio > MAX_BLOCK_RATIO (0.50)                        │
  │  → STATE 3: Block + clarification request                            │
  │    "I cannot find reliable evidence in the provided documents."      │
  └──────────────────────────────────────────────────────────────────────┘

Additional rule: If more than MAX_BLOCK_RATIO fraction of sentences are
HALLUCINATED, block the entire answer regardless of overall score.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.detection.hallucination_engine import SentenceVerdict


# ─────────────────────────────────────────────────────────────────────────────
# Output states
# ─────────────────────────────────────────────────────────────────────────────

STATE_ANSWER  = "ANSWER_WITH_CITATIONS"
STATE_PARTIAL = "PARTIAL_ANSWER"
STATE_BLOCK   = "BLOCKED"


@dataclass
class ControlledResponse:
    """The final structured response produced by the decision controller."""
    state:               str          # ANSWER_WITH_CITATIONS | PARTIAL_ANSWER | BLOCKED
    final_text:          str          # The text shown to the user
    overall_score:       float        # Aggregated support score
    verdicts:            List[SentenceVerdict] = field(default_factory=list)
    cited_sources:       List[str]    = field(default_factory=list)  # e.g. ["doc.pdf p.3"]
    num_suppressed:      int          = 0
    hallucination_ratio: float        = 0.0
    warning_message:     str          = ""
    retrieved_chunks:    List[Dict[str, Any]] = field(default_factory=list) # Chunks from retrieval
    provenance_score:    float        = 0.0 # From XAI layer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state":               self.state,
            "final_text":          self.final_text,
            "overall_score":       self.overall_score,
            "cited_sources":       self.cited_sources,
            "num_suppressed":      self.num_suppressed,
            "hallucination_ratio": self.hallucination_ratio,
            "warning_message":     self.warning_message,
            "provenance_score":    self.provenance_score,
            "verdicts":            [v.to_dict() for v in self.verdicts],
            "retrieved_chunks":    self.retrieved_chunks,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Decision Controller
# ─────────────────────────────────────────────────────────────────────────────

class DecisionController:
    """
    Applies tri-state decision logic to produce a controlled, safe final response.
    """

    def __init__(
        self,
        high_threshold:   float = 0.75,
        medium_threshold: float = 0.50,
        max_block_ratio:  float = 0.50,
        citation_format:  str   = "[{source}, p.{page}]",
    ):
        self.high_thr    = high_threshold
        self.medium_thr  = medium_threshold
        self.max_block   = max_block_ratio
        self.cite_fmt    = citation_format

    def decide(
        self,
        verdicts: List[SentenceVerdict],
        summary:  Dict[str, Any],
        raw_answer: str,
        retrieved_chunks: List[Dict[str, Any]] = None,
        provenance_score: float = 0.0,
    ) -> ControlledResponse:
        """
        Apply tri-state logic and return a ControlledResponse.

        Args:
            verdicts:   Per-sentence verdicts from HallucinationEngine.
            summary:    Summary dict from HallucinationEngine.summarize().
            raw_answer: Raw LLM generated answer text.
            retrieved_chunks: List of retrieved chunks from FAISS search.
            provenance_score: The global provenance score from XAI layer.

        Returns:
            ControlledResponse with state, final text, and metadata.
        """
        overall_score      = summary.get("overall_support_score", 0.0)
        hallucination_ratio = summary.get("hallucination_ratio", 0.0)
        retrieved_chunks   = retrieved_chunks or []

        # ── STATE 3: Block ────────────────────────────────────────────────
        should_block = (
            overall_score < self.medium_thr or
            hallucination_ratio > self.max_block
        )

        if not verdicts or not raw_answer.strip():
            return self._blocked(
                overall_score,
                hallucination_ratio,
                "No relevant documents found for this query.",
                [],
                retrieved_chunks,
                provenance_score,
            )

        if should_block:
            # Build fail-safe response with retrieved evidence & provenance
            evidence_parts = []
            for i, chunk in enumerate(retrieved_chunks):
                source = chunk.get("source_file", "unknown")
                page = chunk.get("page_num", "?")
                text = chunk.get("text", "").strip()
                evidence_parts.append(f"[Evidence {i+1}] Source: {source}, Page: {page}\n\"{text}\"")

            evidence_text = "\n\n".join(evidence_parts)
            blocked_reason = (
                "I cannot find reliable evidence in the provided documents to "
                "answer this question confidently. The response has been blocked due to high hallucination risk. "
                "Here is the raw retrieved source evidence instead:\n\n"
                f"{evidence_text}"
            )
            return self._blocked(
                overall_score,
                hallucination_ratio,
                blocked_reason,
                verdicts,
                retrieved_chunks,
                provenance_score,
            )

        # ── STATE 1: Full answer with citations ──────────────────────────
        # IEEE Paper tri-state alignment: only checks if overall_score >= high_support_threshold (0.75)
        if overall_score >= self.high_thr:
            return self._answer_with_citations(
                verdicts, overall_score, hallucination_ratio, retrieved_chunks, provenance_score
            )

        # ── STATE 2: Partial answer with warnings ────────────────────────
        return self._partial_answer(
            verdicts, overall_score, hallucination_ratio, retrieved_chunks, provenance_score
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _answer_with_citations(
        self,
        verdicts:            List[SentenceVerdict],
        overall_score:       float,
        hallucination_ratio: float,
        retrieved_chunks:    List[Dict[str, Any]],
        provenance_score:    float,
    ) -> ControlledResponse:
        """Build a fully cited answer — STATE 1."""
        parts = []
        cited_sources = []
        seen_sources = set()

        for v in verdicts:
            sentence = v.sentence.strip()
            if not sentence:
                continue

            # Build citation
            if v.source_file:
                citation = self.cite_fmt.format(
                    source=v.source_file,
                    page=v.source_page,
                )
                parts.append(f"{sentence} {citation}")

                src_key = f"{v.source_file} p.{v.source_page}"
                if src_key not in seen_sources:
                    cited_sources.append(src_key)
                    seen_sources.add(src_key)
            else:
                parts.append(sentence)

        final_text = " ".join(parts)

        return ControlledResponse(
            state=STATE_ANSWER,
            final_text=final_text,
            overall_score=overall_score,
            verdicts=verdicts,
            cited_sources=cited_sources,
            num_suppressed=0,
            hallucination_ratio=hallucination_ratio,
            warning_message="",
            retrieved_chunks=retrieved_chunks,
            provenance_score=provenance_score,
        )

    def _partial_answer(
        self,
        verdicts:            List[SentenceVerdict],
        overall_score:       float,
        hallucination_ratio: float,
        retrieved_chunks:    List[Dict[str, Any]],
        provenance_score:    float,
    ) -> ControlledResponse:
        """Build a partial answer with flagged sentences — STATE 2."""
        parts = []
        cited_sources = []
        seen_sources = set()
        num_suppressed = 0

        for v in verdicts:
            sentence = v.sentence.strip()
            if not sentence:
                continue

            if v.classification == "SUPPORTED":
                citation = ""
                if v.source_file:
                    citation = " " + self.cite_fmt.format(
                        source=v.source_file, page=v.source_page
                    )
                    src_key = f"{v.source_file} p.{v.source_page}"
                    if src_key not in seen_sources:
                        cited_sources.append(src_key)
                        seen_sources.add(src_key)
                parts.append(f"{sentence}{citation}")

            elif v.classification == "WEAKLY_SUPPORTED":
                parts.append(
                    f"⚠ [PARTIAL SUPPORT — score {v.support_score:.2f}]: {sentence}"
                )

            else:  # HALLUCINATED — suppress but count
                parts.append(
                    f"⛔ [LOW CONFIDENCE — suppressed claim, score {v.support_score:.2f}]"
                )
                num_suppressed += 1

        final_text = " ".join(parts)
        warning = (
            f"Note: {num_suppressed} claim(s) had insufficient document support "
            f"and were flagged. Overall grounding score: {overall_score:.2%}."
        )

        return ControlledResponse(
            state=STATE_PARTIAL,
            final_text=final_text,
            overall_score=overall_score,
            verdicts=verdicts,
            cited_sources=cited_sources,
            num_suppressed=num_suppressed,
            hallucination_ratio=hallucination_ratio,
            warning_message=warning,
            retrieved_chunks=retrieved_chunks,
            provenance_score=provenance_score,
        )

    def _blocked(
        self,
        overall_score:       float,
        hallucination_ratio: float,
        reason:              str,
        verdicts:            List[SentenceVerdict],
        retrieved_chunks:    List[Dict[str, Any]],
        provenance_score:    float,
    ) -> ControlledResponse:
        """Return a blocked response — STATE 3."""
        return ControlledResponse(
            state=STATE_BLOCK,
            final_text=reason,
            overall_score=overall_score,
            verdicts=verdicts,
            cited_sources=[],
            num_suppressed=len(verdicts),
            hallucination_ratio=hallucination_ratio,
            warning_message=(
                f"Response blocked. Support score: {overall_score:.2%}, "
                f"Hallucination ratio: {hallucination_ratio:.2%}"
            ),
            retrieved_chunks=retrieved_chunks,
            provenance_score=provenance_score,
        )

    def format_for_display(self, response: ControlledResponse) -> str:
        """Format the controlled response as a human-readable string."""
        lines = []

        if response.state == STATE_ANSWER:
            lines.append("✅ Answer (High Confidence — All claims cited)")
            lines.append("")
            lines.append(response.final_text)
            if response.cited_sources:
                lines.append("")
                lines.append("📚 Sources:")
                for src in response.cited_sources:
                    lines.append(f"   • {src}")

        elif response.state == STATE_PARTIAL:
            lines.append("⚠️  Partial Answer (Mixed Confidence)")
            lines.append("")
            lines.append(response.final_text)
            lines.append("")
            lines.append(f"ℹ️  {response.warning_message}")

        else:  # BLOCKED
            lines.append("🚫 Response Blocked (Insufficient Evidence)")
            lines.append("")
            lines.append(response.final_text)
            lines.append("")
            lines.append(f"⚠️  {response.warning_message}")
            if response.retrieved_chunks:
                lines.append("")
                lines.append("📚 Retrieved Evidence Chunks:")
                for i, chunk in enumerate(response.retrieved_chunks):
                    lines.append(f"   • Chunk {i+1} [{chunk.get('source_file')}, p.{chunk.get('page_num')}]:")
                    lines.append(f"     \"{chunk.get('text', '').strip()[:150]}...\"")

        lines.append("")
        lines.append(f"📊 Overall Grounding Score: {response.overall_score:.2%}")
        lines.append(f"🧬 Global Provenance Score: {response.provenance_score:.2%}")

        return "\n".join(lines)
