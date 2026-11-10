"""Print the environment and model/config versions used by the demo (cross-platform)."""
import platform
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import faiss
import sentence_transformers
import torch
import transformers

cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
d = cfg["detection"]
print("Python               :", sys.version.split()[0])
print("Platform             :", platform.platform())
print("torch                :", torch.__version__)
print("transformers         :", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("faiss                :", faiss.__version__)
print("Embedding model      :", cfg["retrieval"]["embedding_model"])
print("NLI model            :", cfg["nli"]["model"])
print("BERTScore model      :", cfg["bertscore"]["model_type"])
print("Fusion weights       :", d["weight_nli"], d["weight_bertscore"], d["weight_cosine"])
print("Thresholds           :", d["high_support_threshold"], d["medium_support_threshold"])
