# Step 6 — Hallucination Detection: NLI, BERTScore, and Cosine Similarity

**Module Location:** `src/detection/`  
**Files:** `nli_detector.py`, `bertscore_detector.py`, `hallucination_engine.py`

---

## The Core Problem This Step Solves

After the LLM generates an answer, we cannot simply trust it. Even with a well-written system prompt telling it to "only use the provided context," LLMs sometimes:
- Generate sentences that go slightly beyond what the retrieved chunks say
- Fill in gaps using memorized training knowledge
- State things confidently that the evidence only weakly supports

This step **verifies each sentence** of the generated response by checking it against the retrieved evidence using three independent methods. Each method looks at the problem from a different angle.

---

## Step 6A — NLI (Natural Language Inference): `nli_detector.py`

### What Is NLI?

Natural Language Inference is a classic NLP task where a model reads two pieces of text:
- **Premise:** An existing statement (in our case, the retrieved chunk)
- **Hypothesis:** A new statement to be verified (in our case, a generated sentence)

And classifies the relationship as:
- **ENTAILMENT:** The premise logically supports the hypothesis ✅
- **NEUTRAL:** The premise neither supports nor contradicts the hypothesis ～
- **CONTRADICTION:** The premise directly contradicts the hypothesis ❌

### Example

**Premise (Retrieved chunk):**  
*"FAISS performs vector similarity search and supports billion-scale datasets."*

**Hypothesis (Generated sentence):**  
*"FAISS is a library that enables fast similarity search over large vector collections."*

**NLI result:** ENTAILMENT — the hypothesis is logically supported by the premise ✅

---

**Hypothesis (Generated sentence):**  
*"FAISS requires a GPU to function."*

**NLI result:** CONTRADICTION — the premise says it supports billion-scale datasets (typically implying CPU capability); this claim is unsupported ❌

---

### Which Model Is Used?

**Model:** `cross-encoder/nli-deberta-v3-small`  
**Architecture:** DeBERTa-v3 (Decoding-Enhanced BERT with Disentangled Attention)  
**Library:** `transformers >= 4.35.0` (HuggingFace)

### Why DeBERTa-v3?

| Reason | Detail |
|:---|:---|
| **Cross-encoder architecture** | It processes the premise and hypothesis *together* in one forward pass — much more accurate than encoding them separately |
| **DeBERTa-v3** | Microsoft's improved BERT variant; achieves near state-of-the-art NLI performance at a manageable size |
| **`-small` variant** | We use the small version to keep inference fast on CPU; still achieves 84%+ accuracy on NLI benchmarks |
| **Local execution** | Cached by HuggingFace after first download — runs fully offline |
| **Three-class output** | Directly outputs ENTAILMENT / NEUTRAL / CONTRADICTION probabilities |

### What Score Is Produced?

The **entailment probability** (a number from 0.0 to 1.0) becomes the NLI signal:
- `S_NLI = 0.92` → The evidence strongly supports this sentence
- `S_NLI = 0.45` → Weak or unclear support
- `S_NLI = 0.08` → Evidence contradicts this sentence

---

## Step 6B — BERTScore: `bertscore_detector.py`

### What Is BERTScore?

BERTScore measures how **semantically similar** a generated sentence is to the reference evidence, at the token level.

It works by:
1. Encoding every word (token) in the generated sentence and in the retrieved evidence using a BERT-type model
2. For each token in the generated sentence, finding the most similar token in the evidence
3. Computing the average similarity across all tokens

### Why BERTScore Complements NLI

NLI checks **logical relationships** between complete sentences. But it can sometimes miss cases where the vocabulary is very different even though the meaning is the same (paraphrasing).

BERTScore catches **semantic overlap at the word level** — even when different words express the same idea.

**Example where BERTScore helps:**
- Evidence: *"LLMs hallucinate by generating text not grounded in facts."*
- Generated: *"Language models fabricate information unsupported by evidence."*
- NLI might give a moderate score because sentence structures differ
- BERTScore gives a high score because "hallucinate/fabricate" and "facts/evidence" are semantically close

### Model Used for BERTScore

**Model:** `distilbert-base-uncased`  
**Library:** `bert-score >= 0.3.13`

**Why DistilBERT?**
- It is a compressed version of BERT (40% smaller, 60% faster)
- Provides 97% of BERT's quality for this task
- Critical for offline speed — BERTScore needs to run for every sentence in every response

### What Score Is Produced?

`S_BERT` — a float from 0.0 to 1.0:
- Above 0.85 → Strong semantic overlap with the evidence
- 0.65 – 0.85 → Moderate overlap
- Below 0.65 → Weak semantic connection

---

## Step 6C — Cosine Similarity (from XAI Layer)

### What It Is

This is the same cosine similarity computed during retrieval (Step 3), reused here as a third verification signal.

Specifically, it computes the cosine similarity between:
- The **embedding of the generated sentence** (encoded with all-MiniLM-L6-v2)
- The **average embedding of all retrieved chunks**

```
e_R = (1/k) × sum(embedding(chunk_i))    # average over top-k chunks
S_COS = cosine(embedding(sentence), e_R)
```

### Why Include Cosine Alongside NLI and BERTScore?

| Signal | Strength | Weakness |
|:---|:---|:---|
| NLI | Captures logical entailment — excellent for factual claims | Can be confused by complex sentence structures |
| BERTScore | Captures paraphrase and synonymy at token level | May score high even if semantic meaning drifts slightly |
| Cosine Similarity | Fast, stable, based on the same embedding space used for retrieval | Less sensitive to logical nuance |

Using all three together gives the most robust and accurate signal. Each compensates for the others' weaknesses.

---

## Step 6D — Combining Signals: `hallucination_engine.py`

The three signals are combined by the `hallucination_engine.py` into a single **Trust Score** per sentence.

### Weights Used

From `config/config.yaml`:
```yaml
weight_nli:       0.45   # NLI gets the highest weight — most informative
weight_bertscore: 0.35   # BERTScore second — strong semantic signal
weight_cosine:    0.20   # Cosine third — supplementary embedding signal
```

### Trust Score Formula

```
Trust Score (T) = 0.45 × S_NLI + 0.35 × S_BERT + 0.20 × S_COS
```

Since all three scores are in [0, 1] and weights sum to 1.0, the Trust Score is also in [0, 1].

### Why NLI Gets the Highest Weight (0.45)?

In the ablation study from the paper:

| Detector Combination | F1-Score | AUC-ROC |
|:---|:---|:---|
| NLI only | 0.856 | 0.918 |
| BERTScore only | 0.792 | 0.874 |
| Cosine only | 0.729 | 0.821 |
| NLI + BERTScore | 0.878 | 0.939 |
| **All three (TrustRAG)** | **0.898** | **0.951** |

NLI alone gives the strongest individual signal, so it receives the highest weight.

---

## Current Status

- ✅ `nli_detector.py` — DeBERTa NLI cross-encoder implemented and tested
- ✅ `bertscore_detector.py` — BERTScore with DistilBERT implemented
- ✅ `hallucination_engine.py` — Trust Score aggregation (weighted combination) implemented
- ✅ All three signals feeding correctly into the decision controller
