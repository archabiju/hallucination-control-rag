"""
XAI Layer — Explainability Layer (CORE NOVELTY)
=================================================
This is the heart of the project's novelty. Instead of treating retrieval
and generation as black boxes, this layer captures internal signals that
explain WHY the system produced each answer:

  1. Cosine similarity scores      — how close is each chunk to the query?
  2. Chunk attribution             — which chunk does each generated sentence come from?
  3. Provenance score              — what fraction of the answer is document-traceable?
  4. Similarity heatmap            — per-token/per-word visual saliency data for UI

These signals are NOT added after the fact — they are captured DURING retrieval
and used directly as hallucination detection features.
"""

import numpy as np
from typing import List, Dict, Any, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cosine Similarity Scores
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity_scores(
    query_vector: np.ndarray,
    chunk_vectors: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and each chunk vector.

    Because both are L2-normalized (from the Embedder), cosine similarity
    equals the dot product. This gives us interpretable scores in [0, 1].

    Args:
        query_vector: Shape (1, D) or (D,). L2-normalized.
        chunk_vectors: Shape (N, D). L2-normalized.

    Returns:
        np.ndarray of shape (N,) — cosine similarity per chunk.
    """
    if query_vector.ndim == 2:
        query_vector = query_vector.flatten()

    # Dot product of L2-normalized vectors = cosine similarity
    scores = chunk_vectors @ query_vector
    return np.clip(scores, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chunk Attribution
# ─────────────────────────────────────────────────────────────────────────────

def compute_chunk_attribution(
    sentences: List[str],
    chunks: List[Dict[str, Any]],
    embedder,
) -> List[Dict[str, Any]]:
    """
    For each generated sentence, find the best-matching source chunk using
    embedding similarity. This creates a sentence → source document mapping.

    Args:
        sentences: List of generated answer sentences (claims).
        chunks: List of retrieved chunk dicts (with text, source_file, page_num).
        embedder: Embedder instance (already loaded, no reload needed).

    Returns:
        List of attribution dicts, one per sentence:
          {
            "sentence": str,
            "best_chunk_text": str,
            "best_chunk_source": str,
            "best_chunk_page": int,
            "attribution_score": float,   # cosine similarity to best chunk
            "attributed": bool,           # True if attribution_score > 0.3
          }
    """
    if not sentences or not chunks:
        return []

    # Embed all sentences
    sentence_vecs = embedder.embed(sentences)  # (N_sents, D)

    # Embed all chunk texts
    chunk_texts = [c["text"] for c in chunks]
    chunk_vecs = embedder.embed(chunk_texts)   # (N_chunks, D)

    # Compute similarity matrix: (N_sents, N_chunks)
    sim_matrix = sentence_vecs @ chunk_vecs.T

    attributions = []
    for i, sentence in enumerate(sentences):
        sim_scores = sim_matrix[i]
        best_idx = int(np.argmax(sim_scores))
        best_score = float(sim_scores[best_idx])
        best_chunk = chunks[best_idx]

        attributions.append({
            "sentence": sentence,
            "sentence_idx": i,
            "best_chunk_text": best_chunk["text"],
            "best_chunk_source": best_chunk.get("source_file", "unknown"),
            "best_chunk_page": best_chunk.get("page_num", 0),
            "best_chunk_id": best_chunk.get("chunk_id", ""),
            "attribution_score": best_score,
            "all_scores": sim_scores.tolist(),
            "attributed": best_score >= 0.30,   # Minimum threshold to count as traceable
        })

    return attributions


# ─────────────────────────────────────────────────────────────────────────────
# 3. Provenance Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_provenance_score(
    attributions: List[Dict[str, Any]],
    attribution_threshold: float = 0.30,
) -> Dict[str, Any]:
    """
    Compute the overall provenance score — the fraction of generated sentences
    that can be traced back to a source document with sufficient confidence.

    A high provenance score means the answer is well-grounded in evidence.
    A low provenance score signals that many claims may be hallucinated.

    Args:
        attributions: Output from compute_chunk_attribution().
        attribution_threshold: Min attribution_score to count as "traceable".

    Returns:
        Dict with:
          - "provenance_score": float in [0, 1]
          - "traceable_count": int
          - "total_sentences": int
          - "untraceable_sentences": List[str]
    """
    if not attributions:
        return {
            "provenance_score": 0.0,
            "traceable_count": 0,
            "total_sentences": 0,
            "untraceable_sentences": [],
        }

    traceable = [a for a in attributions if a["attribution_score"] >= attribution_threshold]
    untraceable = [a["sentence"] for a in attributions if a["attribution_score"] < attribution_threshold]

    score = len(traceable) / len(attributions)

    return {
        "provenance_score": round(score, 4),
        "traceable_count": len(traceable),
        "total_sentences": len(attributions),
        "untraceable_sentences": untraceable,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Similarity Heatmap (for UI visualization)
# ─────────────────────────────────────────────────────────────────────────────

def build_similarity_heatmap(
    query: str,
    chunks: List[Dict[str, Any]],
    similarity_scores: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Build a structured heatmap dict for the Streamlit UI to display
    a bar chart of per-chunk cosine similarity.

    Args:
        query: User query string.
        chunks: Retrieved chunk dicts.
        similarity_scores: np.ndarray of per-chunk cosine scores.

    Returns:
        List of dicts, one per chunk:
          {
            "rank": int,
            "source": str,
            "page": int,
            "similarity": float,
            "text_preview": str,   # First 120 chars
            "bar_width": int,      # 0–100, for UI progress bar
          }
    """
    heatmap = []
    for i, (chunk, score) in enumerate(zip(chunks, similarity_scores)):
        heatmap.append({
            "rank": i + 1,
            "source": chunk.get("source_file", "unknown"),
            "page": chunk.get("page_num", 0),
            "similarity": round(float(score), 4),
            "text_preview": chunk["text"][:150].replace("\n", " ") + "…",
            "bar_width": int(float(score) * 100),
        })

    # Sort by similarity descending
    heatmap.sort(key=lambda x: x["similarity"], reverse=True)
    return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# 5. Word-level Saliency — Leave-One-Out Embedding (IEEE Paper Eq. 4)
# ─────────────────────────────────────────────────────────────────────────────

def compute_word_saliency(
    sentence: str,
    source_chunk: str,
    embedder=None,
) -> List[Dict[str, Any]]:
    """
    Compute a word-level saliency map using the leave-one-out embedding
    method described in the IEEE paper (Eq. 4):

        σ(w_k) = sim(s_j, c*(s_j)) − sim(s_j \\ {w_k}, c*(s_j))

    Words with high positive saliency anchor the sentence to its evidence;
    words with negligible or negative saliency are unsupported insertions
    that characterize hallucination.

    Args:
        sentence: Generated sentence s_j.
        source_chunk: Best-matching source chunk text c*(s_j).
        embedder: Embedder instance for computing cosine similarity.
                  If None, falls back to lexical overlap method.

    Returns:
        List of {word, saliency, in_source} dicts for UI heat-map rendering.
    """
    if embedder is None:
        return compute_word_saliency_lexical(sentence, source_chunk)

    words = sentence.split()
    if not words or not source_chunk.strip():
        return [{"word": w, "saliency": 0.0, "in_source": False} for w in words]

    try:
        # Embed the source chunk once
        chunk_vec = embedder.embed([source_chunk])[0]        # shape (D,)

        # Embed the full sentence → baseline similarity
        full_vec = embedder.embed([sentence])[0]             # shape (D,)
        baseline_sim = float(np.dot(full_vec, chunk_vec))    # cosine (L2-normed)

        source_words_lower = set(source_chunk.lower().split())
        result = []

        for i, word in enumerate(words):
            # Build sentence with word w_k removed
            reduced = " ".join(words[:i] + words[i + 1:])
            if not reduced.strip():
                # Single-word sentence: saliency = baseline
                result.append({
                    "word": word,
                    "saliency": round(baseline_sim, 4),
                    "in_source": word.lower().strip(".,!?;:\"'()") in source_words_lower,
                })
                continue

            reduced_vec = embedder.embed([reduced])[0]
            reduced_sim = float(np.dot(reduced_vec, chunk_vec))

            # σ(w_k) = sim(full) − sim(reduced)   [Eq. 4]
            saliency = baseline_sim - reduced_sim

            clean = word.lower().strip(".,!?;:\"'()")
            in_source = clean in source_words_lower and len(clean) > 2

            result.append({
                "word": word,
                "saliency": round(saliency, 4),
                "in_source": in_source,
            })

        return result

    except Exception as e:
        print(f"[WARN] Embedding-based saliency failed: {e}. Using lexical fallback.")
        return compute_word_saliency_lexical(sentence, source_chunk)


def compute_word_saliency_lexical(
    sentence: str,
    source_chunk: str,
) -> List[Dict[str, Any]]:
    """
    Lightweight fallback saliency using lexical overlap.
    Used when no embedder is available.

    Args:
        sentence: Generated sentence.
        source_chunk: Best-matching source chunk text.

    Returns:
        List of {word, saliency, in_source} dicts for UI highlighting.
    """
    source_words = set(source_chunk.lower().split())
    words = sentence.split()

    result = []
    for word in words:
        clean = word.lower().strip(".,!?;:\"'()")
        in_source = clean in source_words and len(clean) > 2
        result.append({
            "word": word,
            "in_source": in_source,
            "saliency": 0.8 if in_source else 0.1,
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Combined XAI Report
# ─────────────────────────────────────────────────────────────────────────────

def generate_xai_report(
    query: str,
    chunks: List[Dict[str, Any]],
    query_vector: np.ndarray,
    chunk_vectors: np.ndarray,
    sentences: List[str],
    embedder,
) -> Dict[str, Any]:
    """
    Full XAI report: similarity scores + attribution + provenance + heatmap
    + embedding-based word saliency (IEEE Paper Eq. 4).

    Args:
        query: User query string.
        chunks: Retrieved chunk dicts.
        query_vector: Query embedding (L2-normalized).
        chunk_vectors: Chunk embeddings (L2-normalized).
        sentences: Generated answer split into sentences/claims.
        embedder: Embedder instance (used for attribution AND saliency).

    Returns:
        Complete XAI report dict with all explainability signals.
    """
    # 1. Cosine similarity scores
    sim_scores = compute_similarity_scores(query_vector, chunk_vectors)

    # Attach scores to chunks
    scored_chunks = []
    for chunk, score in zip(chunks, sim_scores):
        c = dict(chunk)
        c["similarity_score"] = float(score)
        scored_chunks.append(c)

    # 2. Chunk attribution per sentence
    attributions = compute_chunk_attribution(sentences, scored_chunks, embedder)

    # 3. Provenance score
    provenance = compute_provenance_score(attributions)

    # 4. Similarity heatmap
    heatmap = build_similarity_heatmap(query, scored_chunks, sim_scores)

    # 5. Word saliency per sentence — leave-one-out embedding (Eq. 4)
    for attr in attributions:
        attr["word_saliency"] = compute_word_saliency(
            attr["sentence"], attr["best_chunk_text"], embedder=embedder,
        )

    return {
        "query": query,
        "similarity_scores": sim_scores.tolist(),
        "scored_chunks": scored_chunks,
        "attributions": attributions,
        "provenance": provenance,
        "heatmap": heatmap,
        "num_chunks": len(chunks),
        "num_sentences": len(sentences),
    }
