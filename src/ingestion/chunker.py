"""
Text Chunker — Ingestion Layer
================================
Converts loaded pages into fixed-size overlapping chunks using
a sliding-window approach.

Strategy:
  - Split text into sentences using NLTK.
  - Accumulate sentences until chunk_size tokens is reached.
  - Slide forward by (chunk_size - overlap) tokens.
  - Each chunk carries source metadata for attribution.
"""

import re
import uuid
from typing import List, Dict, Any


def simple_tokenize(text: str) -> List[str]:
    """Split text into word tokens (no external tokenizer required)."""
    return text.split()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex for offline use (no NLTK server)."""
    # Split on sentence-ending punctuation followed by whitespace or newline
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def chunk_text(
    text: str,
    source_file: str,
    page_num: int,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Chunk a single text block into overlapping token-based windows.

    Args:
        text: The raw text to chunk.
        source_file: Name of the source document.
        page_num: Page number in the source document.
        chunk_size: Max number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        List of chunk dicts: {chunk_id, text, source_file, page_num, token_count}
    """
    words = simple_tokenize(text)
    chunks = []

    if not words:
        return chunks

    step = max(1, chunk_size - chunk_overlap)
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": chunk_text_str,
            "source_file": source_file,
            "page_num": page_num,
            "token_count": len(chunk_words),
            "start_word": start,
            "end_word": end,
        })

        if end == len(words):
            break
        start += step

    return chunks


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Chunk a list of loaded page dicts into smaller overlapping chunks.

    Args:
        pages: Output from document_loader.load_corpus().
        chunk_size: Max tokens per chunk.
        chunk_overlap: Overlapping tokens between consecutive chunks.

    Returns:
        Flat list of chunk dicts across all pages.
    """
    all_chunks = []

    for page in pages:
        page_chunks = chunk_text(
            text=page["text"],
            source_file=page["source_file"],
            page_num=page["page_num"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(page_chunks)

    print(f"[INFO] Generated {len(all_chunks)} chunks from {len(pages)} pages.")
    return all_chunks


def get_sentences(text: str) -> List[str]:
    """
    Split generated answer text into individual sentences (claims).
    Used by the hallucination detector to process each claim separately.

    Args:
        text: Generated LLM answer text.

    Returns:
        List of individual sentence strings.
    """
    return split_into_sentences(text)


if __name__ == "__main__":
    # Quick test
    sample_text = (
        "Retrieval-Augmented Generation (RAG) frameworks combine retrieval systems "
        "with large language models. The retriever finds relevant documents and the "
        "generator produces grounded answers. Hallucinations occur when the model "
        "generates content not supported by retrieved evidence. This is a serious "
        "problem in high-stakes applications like medicine, law, and education."
    )
    sample_pages = [
        {"text": sample_text, "source_file": "sample.txt", "page_num": 1}
    ]
    chunks = chunk_pages(sample_pages, chunk_size=30, chunk_overlap=5)
    for c in chunks:
        print(f"\nChunk [{c['start_word']}:{c['end_word']}] ({c['token_count']} tokens)")
        print(c["text"])
