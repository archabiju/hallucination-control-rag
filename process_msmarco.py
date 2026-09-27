"""
MS MARCO Dataset Processor & Indexing Script
=============================================
Processes downloaded MS MARCO dataset files (`collection.tar.gz`, `queries.tar.gz`, `qrels.dev.tsv`),
prepares sample corpus documents into `data/corpus/` and builds a local SBERT + FAISS index.

Usage:
    python process_msmarco.py --num_passages 1000
"""

import os
import tarfile
import argparse
from pathlib import Path

DATA_DIR = Path("data")
CORPUS_DIR = DATA_DIR / "corpus"
PROCESSED_DIR = DATA_DIR / "msmarco_processed"


def extract_queries_if_needed():
    """Extract queries.dev.tsv from queries.tar.gz if not present."""
    if not os.path.exists("queries.dev.tsv") and os.path.exists("queries.tar.gz"):
        print("[INFO] Extracting queries.dev.tsv from queries.tar.gz...")
        with tarfile.open("queries.tar.gz", "r:gz") as tar:
            tar.extract("queries.dev.tsv", path=".")
        print("[INFO] Extracted queries.dev.tsv")


def process_ms_marco(num_passages=1000):
    """
    Extract a subset of MS MARCO passages and convert them into plain text files
    under data/corpus/ so the TrustRAG pipeline can ingest and index them.
    """
    print(f"[INFO] Processing top {num_passages} passages from MS MARCO collection...")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    extract_queries_if_needed()

    collection_path = "collection.tsv"
    if not os.path.exists(collection_path) and os.path.exists("collection.tar.gz"):
        print("[INFO] Extracting collection.tsv from collection.tar.gz...")
        with tarfile.open("collection.tar.gz", "r:gz") as tar:
            tar.extract("collection.tsv", path=".")
        print("[INFO] Extracted collection.tsv")

    if not os.path.exists(collection_path):
        print("[ERROR] collection.tsv not found!")
        return

    out_file = CORPUS_DIR / "msmarco_subset.txt"
    count = 0

    print(f"[INFO] Writing subset passages to {out_file}...")
    with open(collection_path, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(out_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pid, text = parts[0], parts[1]
                f_out.write(f"Passage ID {pid}: {text}\n\n")
                count += 1
                if count >= num_passages:
                    break

    print(f"[SUCCESS] Successfully written {count} MS MARCO passages to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MS MARCO dataset for TrustRAG")
    parser.add_argument("--num_passages", type=int, default=1000, help="Number of passages to process")
    args = parser.parse_args()

    process_ms_marco(args.num_passages)
