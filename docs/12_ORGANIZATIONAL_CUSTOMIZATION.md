# Organizational Customization

TrustRAG is a document-grounded system. Its deployable knowledge is the content
that an organization places in `data/corpus`; it is not a general-purpose
factual oracle.

## What an organization can add

The loader accepts PDF, DOCX, TXT, and Markdown files. Examples include:

- policies, standard operating procedures, and employee handbooks;
- technical manuals and internal knowledge-base exports;
- product documentation and incident runbooks;
- regulatory guidance and approved reference material.

Run the indexing step after documents are added or changed:

```bash
source .venv/bin/activate
python src/pipeline.py
```

The ingestion layer extracts text, splits it into overlapping chunks, embeds
the chunks with the configured sentence encoder, and rebuilds the FAISS index.
Queries are answered from retrieved chunks. The verification layer checks each
claim against those chunks before the controller accepts, warns, or blocks it.

## Boundary of the claim

The correct statement for a review or paper is:

> The system can be customized to an organization by rebuilding its local
> index from that organization's approved documents. Responses are constrained
> by the information retrieved from that index; absent, outdated, or
> contradictory documents can lead to refusal, weak support, or an incorrect
> result.

Do not state that adding documents guarantees correctness. Retrieval can miss
relevant evidence, documents can be wrong, and the verifier has false
positives and false negatives. Evaluation must therefore be repeated on a
held-out, organization-specific question set whenever the corpus changes.

## Evaluation layers

Use separate datasets for separate claims:

1. **Retrieval:** MS MARCO or organization-labelled query/document relevance.
2. **Faithfulness detection:** RAGTruth and the project claim benchmark.
3. **Organization deployment:** questions written and labelled by subject
   matter experts who did not tune the system.

The final test split must not be used to choose fusion weights, thresholds,
window size, or retrieval parameters. Report precision, recall, F1,
balanced accuracy, AUROC, and confidence intervals. A high raw accuracy alone
is not sufficient when most sentences are supported.

## Current demonstration data

`data/corpus/msmarco_subset.txt` currently contains 1,000 passages, not 2,000.
`process_msmarco.py --num_passages 2000` can create a 2,000-passage subset only
when the original MS MARCO `collection.tsv` is available. MS MARCO passages are
retrieval distractors in this project; they are not a substitute for a
sentence-level hallucination benchmark.

For a larger independently annotated faithfulness benchmark:

```bash
python evaluation/build_ragtruth_benchmark.py
python evaluation/run_ragtruth_eval.py --split train
python evaluation/run_ragtruth_eval.py --split test
python evaluation/calibrate_ragtruth_fusion.py
```

The builder creates an 800-response calibration split and a held-out
400-response test split from official RAGTruth annotations. The calibration
script enforces NLI-major non-negative weights and never selects parameters on
the held-out labels.
