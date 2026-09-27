"""
Document Loader — Ingestion Layer
===================================
Loads PDF, DOCX, and TXT files from a directory.
Returns a list of page-level records with text and source metadata.

Supported formats:
  - .pdf   → PyPDF2
  - .docx  → python-docx
  - .txt / .md → plain read
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import PyPDF2
import docx as python_docx


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Load a PDF file and return a list of page-level dicts."""
    pages = []
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append({
                        "text": text,
                        "source_file": Path(file_path).name,
                        "page_num": page_num + 1,
                        "file_type": "pdf",
                    })
    except Exception as e:
        print(f"[WARN] Could not read PDF {file_path}: {e}")
    return pages


def load_docx(file_path: str) -> List[Dict[str, Any]]:
    """Load a DOCX file and return paragraph-level dicts grouped into pages."""
    paragraphs = []
    try:
        doc = python_docx.Document(file_path)
        all_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if all_text:
            paragraphs.append({
                "text": all_text,
                "source_file": Path(file_path).name,
                "page_num": 1,
                "file_type": "docx",
            })
    except Exception as e:
        print(f"[WARN] Could not read DOCX {file_path}: {e}")
    return paragraphs


def load_text(file_path: str) -> List[Dict[str, Any]]:
    """Load a plain text or markdown file."""
    pages = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            pages.append({
                "text": text,
                "source_file": Path(file_path).name,
                "page_num": 1,
                "file_type": "txt",
            })
    except Exception as e:
        print(f"[WARN] Could not read text file {file_path}: {e}")
    return pages


def load_document(file_path: str) -> List[Dict[str, Any]]:
    """Auto-detect file type and load accordingly."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext in (".txt", ".md"):
        return load_text(file_path)
    else:
        print(f"[SKIP] Unsupported file type: {file_path}")
        return []


def load_corpus(corpus_dir: str, extensions: List[str] = None) -> List[Dict[str, Any]]:
    """
    Walk a directory and load all supported documents.

    Args:
        corpus_dir: Path to the folder containing documents.
        extensions: List of file extensions to include (default: pdf, docx, txt, md).

    Returns:
        List of page-level dicts: {text, source_file, page_num, file_type}
    """
    if extensions is None:
        extensions = [".pdf", ".docx", ".txt", ".md"]

    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    all_pages = []
    files = [f for f in corpus_path.rglob("*") if f.suffix.lower() in extensions]

    if not files:
        print(f"[WARN] No supported files found in {corpus_dir}")
        return []

    print(f"[INFO] Found {len(files)} file(s) in corpus directory.")
    for file_path in files:
        print(f"  → Loading: {file_path.name}")
        pages = load_document(str(file_path))
        all_pages.extend(pages)
        print(f"     Loaded {len(pages)} page(s).")

    print(f"[INFO] Total pages loaded: {len(all_pages)}")
    return all_pages


if __name__ == "__main__":
    # Quick test
    import sys
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "data/corpus"
    pages = load_corpus(test_dir)
    for p in pages[:3]:
        print(f"\n--- {p['source_file']} p.{p['page_num']} ---")
        print(p["text"][:200])
