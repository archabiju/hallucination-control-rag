# Step 5 — Local LLM Generation

**Module Location:** `src/generation/`  
**File:** `llm_generator.py`  
**Tool Used:** Ollama  
**Model Used:** Mistral-7B-Instruct

---

## What Happens at This Step?

Once the Top-K relevant chunks are retrieved and the XAI signals are captured, the system needs to actually **generate an answer** to the user's question.

This is done by a **Large Language Model (LLM)** — the same type of model used in ChatGPT, but running **completely locally on your own machine** with no internet connection.

---

## Why Not Use ChatGPT or Claude API?

| Reason | Detail |
|:---|:---|
| **Privacy** | Sending queries to OpenAI/Claude means your data leaves your machine and is processed on external servers |
| **Cost** | GPT-4 API costs money per query — not feasible for a research project running hundreds of test queries |
| **Offline requirement** | This project is specifically designed for environments without internet access |
| **Reproducibility** | With a local model, the same input always produces the same output — cloud APIs can change behavior with model updates |

---

## What Tool Is Used: Ollama

**Ollama** is a free, open-source tool that allows you to download and run large language models locally on your computer.

Think of Ollama like a local server: you start it on your machine, and it acts like a local API that your Python code can talk to — exactly like talking to OpenAI's API, but all on your own hardware.

**How it works:**
```
Python code (llm_generator.py)
    ↓ HTTP request to localhost:11434
Ollama Server (running locally)
    ↓ runs inference
Mistral-7B model (stored on your disk)
    ↓ returns generated text
Python code receives the response
```

**Ollama base URL used:** `http://localhost:11434` (set in `config/config.yaml`)

---

## Which LLM Model: Mistral-7B-Instruct

**Model:** `mistral` (Mistral-7B-Instruct v0.2)  
**Size:** ~4 GB (quantized to 4-bit for efficient local inference)  
**Parameters:** 7 billion

### Why Mistral-7B?

| Reason | Detail |
|:---|:---|
| **Strong instruction following** | The "Instruct" variant is specifically fine-tuned to follow precise instructions — critical for telling it "answer ONLY using the provided context" |
| **7B size is manageable** | 7 billion parameters can run on a consumer laptop with 16 GB RAM (especially with 4-bit quantization) |
| **High quality for its size** | Mistral-7B outperforms many larger models on standard benchmarks |
| **Fully open-source** | No restrictions on academic or research use |
| **Quantization support** | Ollama automatically uses GGUF quantized versions — reducing memory from ~14 GB (full float32) to ~4 GB (4-bit int) |

---

## How the LLM Is Instructed

The most important part of generation is the **system prompt** — the instruction given to the LLM before it sees the question.

**System prompt used (from `config/config.yaml`):**
```
You are a precise, factual assistant. Answer ONLY using the provided context.
If the context does not support the answer, say so explicitly.
Do NOT fabricate information.
```

This is critical. Without this instruction, the LLM might use its own training memory instead of the retrieved documents, causing hallucinations.

### The Full Prompt Structure

```
[SYSTEM PROMPT]
You are a precise, factual assistant...

[CONTEXT]
Context 1: "BERTScore measures token-level semantic similarity using BERT..." (survey.pdf, p.4)
Context 2: "NLI models classify text pairs as entail/neutral/contradict..." (guide.pdf, p.8)
...

[USER QUESTION]
How does BERTScore measure semantic similarity?

[LLM ANSWER]
BERTScore evaluates text by computing token-level similarity between generated 
and reference text using contextual BERT embeddings...
```

---

## Generation Settings

| Parameter | Value | Why |
|:---|:---|:---|
| `temperature` | `0.1` | Very low temperature means the model picks the most likely, factual answer rather than being creative or random |
| `max_tokens` | `512` | Limits response length — enough for a complete answer without rambling |

**Low temperature (0.1)** is used because we want *factual accuracy*, not creative writing. A temperature of 0 would always pick the single most probable token; 0.1 adds just a tiny amount of variation.

---

## Current Status

- ✅ `llm_generator.py` fully implemented
- ✅ Connects to Ollama server at `localhost:11434`
- ✅ System prompt configured for grounded, factual generation
- ✅ Mistral-7B model downloaded and running locally via Ollama
- ✅ Generated responses are passed directly to the hallucination detection engine
