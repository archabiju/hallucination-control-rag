"""Build a larger, independently annotated faithfulness benchmark.

RAGTruth is downloaded from its official repository and converted from
hallucinated character spans to sentence-level examples. This benchmark is
kept separate from the 37-question project benchmark:

* RAGTruth train can be used for calibration/model selection.
* RAGTruth test is held out for the final report.

The script never invents or relabels examples. Every positive label comes from
an annotated RAGTruth hallucination span.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "evaluation" / "benchmarks" / "ragtruth"
SOURCE_URL = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/"
    "main/dataset/source_info.jsonl"
)
RESPONSE_URL = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/"
    "main/dataset/response.jsonl"
)
SENTENCE_RE = re.compile(r"\S(?:.*?\S)?(?:[.!?]+(?=\s|$)|(?=\n)|$)", re.DOTALL)


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as response:
        path.write_bytes(response.read())


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def flatten_source(value: Any, prefix: str = "") -> str:
    """Convert RAGTruth's task-specific source_info value into evidence text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(
            text for item in value if (text := flatten_source(item, prefix))
        )
    if isinstance(value, dict):
        pieces = []
        for key, item in value.items():
            text = flatten_source(item, str(key))
            if text:
                pieces.append(f"{key}: {text}")
        return "\n".join(pieces)
    if value is None:
        return ""
    return str(value)


def sentence_spans(text: str) -> Iterable[tuple[int, int, str]]:
    for match in SENTENCE_RE.finditer(text):
        sentence = match.group(0).strip()
        if len(sentence.split()) >= 4:
            left_trim = len(match.group(0)) - len(match.group(0).lstrip())
            start = match.start() + left_trim
            yield start, start + len(sentence), sentence


def overlaps(start: int, end: int, labels: list[dict[str, Any]]) -> bool:
    return any(start < int(label["end"]) and int(label["start"]) < end for label in labels)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stratified_responses(
    response_path: Path,
    sources: dict[str, dict[str, Any]],
    limits: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Select an approximately equal number of QA, Summary, and Data2txt rows."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {} for split in limits
    }
    for response in read_jsonl(response_path):
        split = response.get("split")
        if split not in limits:
            continue
        if response.get("quality") not in (None, "", "good"):
            continue
        source = sources.get(response.get("source_id"))
        if not source:
            continue
        task = str(source.get("task_type"))
        grouped[split].setdefault(task, []).append(response)

    selected: dict[str, list[dict[str, Any]]] = {}
    for split, limit in limits.items():
        tasks = sorted(grouped[split])
        rows = []
        cursor = 0
        while len(rows) < limit and tasks:
            task = tasks[cursor % len(tasks)]
            bucket = grouped[split][task]
            if bucket:
                rows.append(bucket.pop(0))
            else:
                tasks.remove(task)
                if not tasks:
                    break
                cursor -= 1
            cursor += 1
        selected[split] = rows
    return selected


def build(out_dir: Path, max_train_responses: int, max_test_responses: int) -> None:
    raw = out_dir / "raw"
    source_path = raw / "source_info.jsonl"
    response_path = raw / "response.jsonl"
    download(SOURCE_URL, source_path)
    download(RESPONSE_URL, response_path)

    sources = {row["source_id"]: row for row in read_jsonl(source_path)}
    limits = {"train": max_train_responses, "test": max_test_responses}
    selected = stratified_responses(response_path, sources, limits)
    used = Counter()
    claims = Counter()
    hall = Counter()
    task_counts: dict[str, Counter] = {
        "train": Counter(),
        "test": Counter(),
    }
    handles = {
        split: (out_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in limits
    }
    try:
        for split, responses in selected.items():
            for response in responses:
                source = sources[response["source_id"]]
                evidence = flatten_source(source.get("source_info"))
                if not evidence:
                    continue
                labels = response.get("labels") or []
                examples = []
                for start, end, sentence in sentence_spans(response.get("response", "")):
                    is_hallucinated = overlaps(start, end, labels)
                    examples.append(
                        {
                            "claim": sentence,
                            "hallucinated": is_hallucinated,
                            "span_start": start,
                            "span_end": end,
                        }
                    )
                # Keep responses that contribute at least one usable sentence.
                # Do not balance by label: the test distribution remains natural.
                if not examples:
                    continue
                record = {
                    "response_id": response["id"],
                    "source_id": response["source_id"],
                    "split": split,
                    "task_type": source.get("task_type"),
                    "source_dataset": source.get("source"),
                    "model": response.get("model"),
                    "temperature": response.get("temperature"),
                    "prompt": source.get("prompt", ""),
                    "evidence": evidence,
                    "response": response.get("response", ""),
                    "claims": examples,
                }
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                used[split] += 1
                claims[split] += len(examples)
                hall[split] += sum(item["hallucinated"] for item in examples)
                task_counts[split][str(source.get("task_type"))] += 1
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "dataset": "RAGTruth",
        "source_repository": "https://github.com/ParticleMedia/RAGTruth",
        "license": "Apache-2.0 (see source repository)",
        "conversion": (
            "A sentence is hallucinated when its character range overlaps at "
            "least one RAGTruth hallucination annotation. No labels are generated."
        ),
        "train_is_for_calibration_only": True,
        "test_is_held_out": True,
        "counts": {
            split: {
                "responses": used[split],
                "claims": claims[split],
                "hallucinated_claims": hall[split],
                "supported_claims": claims[split] - hall[split],
                "task_types": dict(task_counts[split]),
            }
            for split in limits
        },
        "raw_sha256": {
            "source_info.jsonl": sha256(source_path),
            "response.jsonl": sha256(response_path),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-train-responses", type=int, default=800)
    parser.add_argument("--max-test-responses", type=int, default=400)
    args = parser.parse_args()
    build(args.out_dir, args.max_train_responses, args.max_test_responses)


if __name__ == "__main__":
    main()
