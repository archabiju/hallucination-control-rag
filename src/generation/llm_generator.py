"""
LLM Generator — Generation Layer
====================================
Interfaces with a locally running Ollama server to generate answers
from retrieved context. No cloud API, no internet required.

Fallback mode: If Ollama is not running, returns a template message
so the rest of the pipeline (explainability, detection) can still be tested.

Setup:
    1. Install Ollama from https://ollama.ai (one-time download)
    2. Pull a model:  ollama pull mistral
    3. Start server:  ollama serve
    4. This module then calls http://localhost:11434/api/generate
"""

import json
import requests
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a strictly grounded, factual assistant. You ONLY answer questions using the provided context.

Rules:
1. If the provided context does NOT contain the specific information needed to answer the question, respond ONLY with:
   "I do not have any relevant documents related to that question in my knowledge base."
2. Do NOT mention, summarize, or describe unrelated topics from the context if they do not directly answer the user's question.
3. Do NOT guess, speculate, or fabricate information.
4. When the context DOES answer the question, be concise, direct, and factual in 2 to 3 complete sentences.
5. Finish every sentence completely."""


def build_prompt(query: str, context_chunks: list) -> str:
    """
    Build the full LLM prompt from the user query and retrieved chunks.

    Args:
        query: The user's question.
        context_chunks: List of retrieved chunk dicts (with 'text', 'source_file', 'page_num').

    Returns:
        Formatted prompt string.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = chunk.get("source_file", "unknown")
        page = chunk.get("page_num", "?")
        score = chunk.get("similarity_score", 0.0)
        context_parts.append(
            f"[Context {i+1}] Source: {source}, Page: {page} (similarity: {score:.3f})\n"
            f"{chunk['text']}"
        )

    context_text = "\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

=== CONTEXT ===
{context_text}

=== QUESTION ===
{query}

=== ANSWER ==="""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Generator class
# ─────────────────────────────────────────────────────────────────────────────

class LLMGenerator:
    """
    Wraps the Ollama local LLM API to generate grounded answers.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "mistral",
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._check_connection()

    def _check_connection(self) -> None:
        """Check if Ollama server is running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=15)
            if resp.status_code == 200:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                print(f"[INFO] Ollama running. Available models: {models}")
                if self.model_name not in " ".join(models):
                    print(f"[WARN] Model '{self.model_name}' may not be pulled. "
                          f"Run: ollama pull {self.model_name}")
            else:
                print(f"[WARN] Ollama responded with status {resp.status_code}")
        except Exception:
            print(f"[WARN] Ollama not reachable at {self.base_url}. "
                  "Fallback mode enabled (template responses only).")
            self._fallback_mode = True
            return
        self._fallback_mode = False

    def generate(
        self,
        query: str,
        context_chunks: list,
    ) -> dict:
        """
        Generate an answer for the given query using retrieved context chunks.

        Args:
            query: User question string.
            context_chunks: List of retrieved chunk dicts from FAISS search.

        Returns:
            dict with keys:
              - "answer": Generated text string.
              - "prompt": Full prompt sent to LLM.
              - "model": Model name used.
              - "fallback": True if Ollama was not available.
        """
        prompt = build_prompt(query, context_chunks)

        if getattr(self, "_fallback_mode", False):
            return self._fallback_response(query, context_chunks, prompt)

        try:
            ctx_size = 512 if "mistral" in self.model_name.lower() else 1024
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "num_ctx": ctx_size,
                    "stop": ["=== QUESTION ===", "=== CONTEXT ==="],
                },
            }

            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("response", "").strip()

            return {
                "answer": answer,
                "prompt": prompt,
                "model": self.model_name,
                "fallback": False,
            }

        except requests.exceptions.Timeout:
            print("[WARN] Ollama request timed out. Using fallback.")
            return self._fallback_response(query, context_chunks, prompt)
        except Exception as e:
            print(f"[WARN] LLM generation error: {e}. Using fallback.")
            return self._fallback_response(query, context_chunks, prompt)

    def _fallback_response(
        self,
        query: str,
        context_chunks: list,
        prompt: str,
    ) -> dict:
        """
        Produce a simple extractive fallback when Ollama is unavailable.
        Uses the top retrieved chunk's first two sentences as a proxy answer.
        """
        if context_chunks:
            top_chunk = context_chunks[0]["text"]
            # Take the first 200 chars as a basic answer
            answer = top_chunk[:300].strip() + "..."
            answer = (
                f"[FALLBACK — Ollama not running] Based on retrieved context:\n{answer}"
            )
        else:
            answer = (
                "[FALLBACK] No relevant documents found for this query. "
                "Please ensure documents are indexed and Ollama is running."
            )

        return {
            "answer": answer,
            "prompt": prompt,
            "model": "fallback",
            "fallback": True,
        }


if __name__ == "__main__":
    gen = LLMGenerator(model_name="tinyllama")
    fake_chunks = [
        {
            "text": "RAG combines retrieval with generation to produce grounded answers.",
            "source_file": "rag_paper.pdf",
            "page_num": 1,
            "similarity_score": 0.92,
        }
    ]
    result = gen.generate("What is RAG?", fake_chunks)
    print("\n=== Generated Answer ===")
    print(result["answer"])
    print(f"\nModel: {result['model']} | Fallback: {result['fallback']}")
