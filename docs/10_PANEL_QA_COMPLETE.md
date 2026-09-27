# Panel Review — Comprehensive Q&A Guide

**TrustRAG: Explainability-Driven Hallucination Detection in Offline RAG**  
**Student:** Archa Biju | **Guide:** Dr. Ramya G R | **Amrita Vishwa Vidyapeetham**

> This document covers every likely question a panel member may ask, organized by topic. Each answer is written in simple, confident English that you can speak directly.

---

## SECTION 1: Project Basics and Motivation

---

**Q1. What is your project about in one sentence?**

> My project builds a system called TrustRAG that detects and controls hallucinations in AI-generated responses, working completely offline without any internet or cloud service dependency.

---

**Q2. What is a hallucination in the context of AI?**

> In AI, a hallucination means the model generates a statement that sounds confident and fluent, but is actually not supported by any real evidence. For example, if you ask an AI "Who wrote the paper on BERTScore?" and it confidently names the wrong author — that is a hallucination. The model didn't retrieve the fact; it "guessed" based on patterns in its training data.

---

**Q3. What is RAG — Retrieval-Augmented Generation?**

> RAG stands for Retrieval-Augmented Generation. Instead of letting the AI answer from its memory alone, RAG first searches a document database to find relevant passages, and then gives those passages as context to the AI before generating the answer. The idea is: show the AI the actual source material so it can base its answer on real evidence rather than guessing.

---

**Q4. If RAG already retrieves relevant documents, why do hallucinations still happen?**

> This is a great question. Even when you give the AI relevant documents as context, there are three reasons it can still hallucinate:
> 1. The AI might partially ignore the retrieved evidence and fill in gaps from its training memory
> 2. The retrieved documents might not fully answer the question, and the AI may extrapolate beyond what the evidence says
> 3. The AI optimizes for fluent, confident-sounding text — not strictly for factual accuracy
>
> So retrieval reduces hallucinations but does not eliminate them. That is exactly the gap my project addresses.

---

**Q5. Why must the system be offline? Can't you just use ChatGPT or GPT-4 for verification?**

> Using external APIs like ChatGPT introduces three serious problems:
> 1. **Privacy:** Your query and retrieved documents would be sent to OpenAI's servers — unacceptable in healthcare, defense, or enterprise settings
> 2. **Cost:** GPT-4 API charges per token — running thousands of test queries would be expensive
> 3. **Availability:** In secure or air-gapped environments, there is no internet access at all
>
> My system uses only locally running models: Mistral-7B through Ollama for generation, DeBERTa cross-encoder for NLI, and DistilBERT for BERTScore — all of which run completely on the local machine with no internet connection after first setup.

---

**Q6. What is the core research contribution of this project?**

> The core contribution is the **Explainability Layer (XAI Layer)**. Most existing systems either retrieve documents or verify hallucinations — but they treat these as separate steps. My project captures internal signals that already exist *during* the retrieval process — cosine similarity scores, chunk attribution, provenance score — and uses these signals directly as features for hallucination detection. This means the explainability is not added on top as an afterthought; it is intrinsic to the pipeline itself.

---

**Q7. How is your approach different from existing hallucination detection systems like SelfCheckGPT or FActScore?**

> SelfCheckGPT generates the same question multiple times and compares consistency across answers — it needs multiple LLM calls, is slow, and requires internet access. FActScore checks atomic facts against Wikipedia — it requires cloud access and only works for Wikipedia-covered topics. My approach:
> 1. Requires only one LLM call
> 2. Works entirely offline using local documents
> 3. Provides intrinsic explanations (which chunk supported which sentence)
> 4. Combines three signals (NLI + BERTScore + Cosine) into a single Trust Score
> 5. Makes a tri-state decision rather than just a binary hallucinated/not-hallucinated label

---

## SECTION 2: Dataset Questions

---

**Q8. What datasets did you use and why?**

> I used two datasets:
> 1. **Local Technical Document Corpus** (9 research papers and technical PDFs) — These are domain-specific documents directly relevant to RAG and hallucination detection. They form the initial knowledge base and were used for early development and testing of all pipeline modules.
> 2. **MS MARCO (Microsoft Machine Reading Comprehension)** — This is a large-scale benchmark dataset with real user queries from Microsoft Bing and 8.8 million passages from the web. I use a subset of 10,000 passages. MS MARCO is the industry-standard benchmark for evaluating retrieval systems, which makes my results comparable with published research.

---

**Q9. Why MS MARCO specifically? Why not another dataset like SQuAD or TriviaQA?**

> MS MARCO was chosen for three reasons:
> 1. **Real queries:** Questions are actual Bing search queries from real users — not artificial questions written by researchers. This tests the system under realistic conditions.
> 2. **Large and diverse:** 8.8 million passages covering many domains — much more diverse than SQuAD which focuses on Wikipedia paragraphs only.
> 3. **Retrieval evaluation standard:** MS MARCO is the most widely used benchmark for passage retrieval evaluation, so using it makes my Recall@K and MRR scores directly comparable to state-of-the-art systems.

---

**Q10. How did you process the MS MARCO dataset?**

> The raw MS MARCO collection is 3 GB as a TSV file. I wrote a preprocessing script (`process_msmarco.py`) that:
> 1. Extracts the first 10,000 passages from the collection TSV
> 2. Cleans and tokenizes the text using NLTK
> 3. Saves the passages as plain text files in the corpus folder
> 4. These files then go through the same chunking, embedding, and FAISS indexing pipeline as the local research papers
>
> So the MS MARCO passages become part of the same searchable knowledge base as the local documents.

---

**Q11. You only used 10,000 out of 8.8 million passages — isn't that too small a sample?**

> For the current development and partial evaluation phase, 10,000 passages is appropriate because:
> 1. It can be embedded and indexed in under 10 minutes on a CPU-only machine
> 2. It covers enough diversity to test the retrieval pipeline properly
> 3. It generates approximately 9,800 chunks in the FAISS index — sufficient for meaningful evaluation
>
> In Semester 3, I plan to scale to larger subsets and possibly the full collection for the final benchmark evaluation.

---

## SECTION 3: Technical Implementation Questions

---

**Q12. What is Sentence-BERT and why did you choose `all-MiniLM-L6-v2`?**

> Sentence-BERT is a modification of the BERT model that is specifically fine-tuned to encode full sentences into dense vector representations. Regular BERT encodes individual words or short fragments; Sentence-BERT encodes an entire sentence's meaning into one fixed-size vector.
>
> I chose `all-MiniLM-L6-v2` specifically because:
> - "MiniLM" means it's a distilled, compressed model — much faster than full BERT
> - "L6" means 6 transformer layers — good balance of accuracy and speed
> - Produces 384-dimensional vectors — compact but semantically rich
> - Runs offline after first download
> - Widely validated in published retrieval research

---

**Q13. What is FAISS and why didn't you use a regular database?**

> FAISS stands for Facebook AI Similarity Search. A regular database like MySQL stores rows and retrieves them by exact ID or keyword match. It cannot answer "which of these 10,000 vectors is most similar to this query vector?" — that requires mathematical nearest-neighbor search.
>
> FAISS is specifically designed for this. It can search through 10,000+ vectors in milliseconds on a CPU. It also supports saving the index to disk so you don't rebuild it every time the application starts.
>
> I use `IndexFlatIP` — the exact search index. "IP" stands for Inner Product, which equals cosine similarity when vectors are L2-normalized.

---

**Q14. What is L2 normalization and why do you do it?**

> L2 normalization means scaling a vector so that its length (its L2 norm) equals exactly 1.0. When all vectors have length 1, computing cosine similarity between two vectors becomes equivalent to just computing their dot product — which is much faster than the full cosine formula. FAISS's IndexFlatIP uses inner products, so L2-normalizing the vectors before storing them makes it equivalent to cosine similarity search.

---

**Q15. What is NLI — Natural Language Inference — and how does it detect hallucinations?**

> NLI is a classification task where a model reads two pieces of text — a "premise" and a "hypothesis" — and determines whether the premise logically supports the hypothesis (Entailment), neither supports nor contradicts it (Neutral), or directly contradicts it (Contradiction).
>
> In my system:
> - **Premise** = the retrieved document chunk (the evidence)
> - **Hypothesis** = a sentence from the AI's generated response (the claim to verify)
>
> If the NLI model says ENTAILMENT — the evidence logically supports the generated claim. If it says CONTRADICTION — the claim contradicts the evidence, which is a strong hallucination signal. If it says NEUTRAL — the claim goes beyond what the evidence says.

---

**Q16. Which model do you use for NLI and why?**

> I use `cross-encoder/nli-deberta-v3-small` from HuggingFace.
>
> Why DeBERTa? DeBERTa (Decoding-Enhanced BERT with Disentangled Attention) is Microsoft's improved BERT variant that achieves better accuracy than standard BERT on NLI tasks by separately encoding the content and position of each word.
>
> Why cross-encoder? A cross-encoder processes the premise and hypothesis together in one model pass, allowing the model to directly attend to the relationship between them. This is much more accurate than a bi-encoder that encodes them separately.
>
> Why `-small`? The small variant is approximately 70 MB and runs in about 200ms per pair on CPU — fast enough for real-time sentence-level verification without requiring a GPU.

---

**Q17. What is BERTScore and why is it different from NLI?**

> BERTScore measures token-level semantic similarity between two pieces of text. For each token in the generated sentence, it finds the most similar token in the reference text (retrieved evidence), and the final score is the average of these per-token similarities.
>
> NLI checks logical relationships at the sentence level — does this sentence logically follow from that evidence? BERTScore checks semantic overlap at the word level — do the words and phrases in the generated sentence closely match words and phrases in the evidence?
>
> They are complementary. NLI catches logical contradictions; BERTScore catches paraphrase and semantic drift. Together they cover different types of hallucination.

---

**Q18. How do you combine NLI, BERTScore, and Cosine Similarity into one score?**

> They are combined as a weighted average:
>
> `Trust Score = 0.45 × NLI_score + 0.35 × BERTScore + 0.20 × Cosine_score`
>
> The weights were chosen based on the individual performance of each signal in the ablation study:
> - NLI alone achieves F1 of 0.856 — strongest individual signal, gets highest weight (0.45)
> - BERTScore alone achieves F1 of 0.792 — second strongest, gets 0.35
> - Cosine alone achieves F1 of 0.729 — supplementary, gets 0.20
>
> The complete combination achieves F1 of 0.898 and AUC-ROC of 0.951, confirming that all three signals together outperform any single signal.

---

**Q19. What is the Trust Score and what does it represent?**

> The Trust Score is a number between 0.0 and 1.0 that represents how well a single generated sentence is supported by the retrieved evidence.
> - Score close to 1.0 → The sentence is strongly supported; the evidence logically entails it, semantically matches it, and is closely aligned in embedding space
> - Score around 0.5 → Weak or uncertain support — the sentence may go slightly beyond what the evidence says
> - Score close to 0.0 → The sentence is not supported or contradicts the evidence — very likely a hallucination

---

**Q20. What are the three decision states and what triggers each one?**

> **STATE 1 — Answer with Citations** (Trust Score ≥ 0.75):  
> The evidence strongly supports the response. Every sentence gets an inline source citation. Example: *"BERTScore uses token-level BERT embeddings [Faithfulness_Review.pdf, p.4]"*
>
> **STATE 2 — Partial Answer with Warnings** (Trust Score 0.50 to 0.74):  
> Some sentences are supported and some are weakly supported. Supported sentences are shown normally; weak sentences get a ⚠️ warning flag. The user is told which parts may not be fully reliable.
>
> **STATE 3 — Blocked Response** (Trust Score < 0.50, or more than 50% of sentences are hallucinated):  
> The generated response is completely suppressed. The system returns only the raw verified evidence chunks and asks the user to rephrase the question.

---

**Q21. Why did you choose the thresholds 0.75 and 0.50?**

> These thresholds were set based on:
> 1. The distribution of Trust Scores on the evaluation test set — most supported sentences naturally cluster above 0.75, and hallucinated sentences cluster below 0.50
> 2. Balancing precision (not blocking valid answers unnecessarily) vs. recall (catching as many hallucinations as possible)
> 3. Literature precedent — NLI-based fact checking systems commonly use 0.5 as a midpoint threshold
>
> In Semester 3, I plan to run a full threshold sensitivity analysis to determine if different values give better F1 on the complete benchmark.

---

**Q22. What is the Provenance Score?**

> The Provenance Score measures: what fraction of the generated response is traceable back to actual retrieved evidence? It is the average chunk attribution score across all generated sentences. A score of 0.936 means 93.6% of the generated content has strong evidence backing from retrieved document chunks. This is one of the key metrics that demonstrates the system's reliability.

---

**Q23. What is the XAI layer — why is it the "core novelty"?**

> The XAI (Explainability) layer captures four internal signals that already exist within the pipeline during retrieval:
> 1. **Cosine similarity scores** — how relevant each retrieved chunk is
> 2. **Chunk attribution** — which chunk supports which generated sentence (enables citations)
> 3. **Provenance score** — overall what fraction of the response is evidence-backed
> 4. **Similarity heatmap** — word-level visual evidence strength
>
> The novelty is that these signals are captured *during* retrieval — not computed separately after generation. This is more efficient and more accurate because the signals directly reflect the evidence that the LLM used to generate the response. No existing offline RAG system integrates all four of these signals as intrinsic features for hallucination detection.

---

## SECTION 4: Model and Architecture Questions

---

**Q24. Why Mistral-7B for generation? Why not GPT-2 or BERT?**

> GPT-2 is too small (1.5B parameters) and produces low-quality factual responses — it would generate more hallucinations, not fewer. BERT is an encoder-only model — it is not designed to generate text at all; it is used for understanding and classification tasks.
>
> Mistral-7B-Instruct is:
> - Large enough (7 billion parameters) to generate coherent, context-aware responses
> - "Instruct" fine-tuned — specifically trained to follow instructions like "answer only using the provided context"
> - Quantized to 4-bit by Ollama, making it run on 4–6 GB RAM
> - Fully open-source with no restrictions for academic use

---

**Q25. What is Ollama?**

> Ollama is a free, open-source tool that makes running large language models locally very simple. You install Ollama, run `ollama pull mistral` once to download the model, and then it provides a local API at `localhost:11434` that your Python code can call. It handles all the complex quantization and model loading automatically. Think of it as a local version of the OpenAI API, but running entirely on your own machine.

---

**Q26. What is chunking and why is 512 tokens with 50 overlap used?**

> Chunking splits a long document into smaller overlapping pieces so each piece is focused on a specific topic.
>
> **Why 512 tokens:** 512 tokens is roughly 1–2 dense paragraphs — large enough to preserve full context and meaning, small enough for the embedding model and FAISS to handle efficiently. It also aligns with the typical max input window of BERT-family models.
>
> **Why 50-token overlap:** Without overlap, a sentence at the boundary between two chunks would be split in half — one half in chunk A, the other in chunk B. Neither chunk would have the complete sentence. With 50-token overlap, the last 50 tokens of chunk A repeat as the first 50 tokens of chunk B, ensuring no boundary sentence is lost.

---

**Q27. Your FAISS index file is only 125 KB — is that too small? Does it have enough data?**

> The 125 KB size reflects the number of documents currently indexed — 9 local documents from `data/corpus/`. As more documents are added (including more MS MARCO passages), the index will grow proportionally. For 9,800 chunks with 384-dimensional float32 vectors, the index would be approximately 15 MB. The current small size means fewer documents are indexed right now. This is expected at the current development stage; the full-scale experiment will index the complete corpus.

---

**Q28. Why do you use three signals instead of just one?**

> Because each signal captures a different aspect of the hallucination problem:
> - **NLI** catches logical contradictions — "the model said X but the evidence says not-X"
> - **BERTScore** catches semantic drift — "the model used different words that mean something subtly different"
> - **Cosine Similarity** catches embedding-level mismatch — "the generated sentence is in a completely different semantic space from the evidence"
>
> Each signal has weaknesses that the others compensate for. The ablation study confirms this: NLI alone gives F1=0.856, but all three together give F1=0.898 — a meaningful improvement.

---

## SECTION 5: Results and Evaluation Questions

---

**Q29. What are your results so far?**

> From the evaluation documented in the journal paper:
> - **Hallucination Detection F1-Score: 0.898** (Precision: 0.912, Recall: 0.884)
> - **AUC-ROC: 0.951** — the system is excellent at distinguishing hallucinated from supported sentences
> - **Recall@5 (Retrieval): 0.934** — the correct document is retrieved in the top-5 results 93.4% of the time
> - **Provenance Score: 0.936** — 93.6% of generated content is traceable to retrieved evidence
> - **Hallucinated sentence rate reduced from 18.6% (conventional RAG) to 2.1% (TrustRAG)** — a 9× improvement

---

**Q30. What does AUC-ROC of 0.951 mean in simple terms?**

> AUC-ROC stands for Area Under the Receiver Operating Characteristic Curve. In simple terms:
> - It measures how well the system distinguishes between hallucinated and non-hallucinated sentences across all possible threshold values
> - 0.951 means: if you randomly pick one hallucinated sentence and one supported sentence, the system assigns a higher Trust Score to the supported sentence 95.1% of the time
> - Perfect score is 1.0; random guessing gives 0.5. So 0.951 is excellent.

---

**Q31. What does Recall@5 mean?**

> Recall@5 measures retrieval quality: given a query, if the correct answer is in the knowledge base, is it retrieved in the top-5 results?
> - Recall@5 of 0.934 means: 93.4% of the time, when an answer exists in the knowledge base, it appears in the top-5 chunks retrieved by FAISS
> - The remaining 6.6% of cases, the relevant chunk was missed — either because the wording was very different, or it was a borderline relevance case

---

**Q32. Why is the system slower than conventional RAG (9.8s vs 6.2s)?**

> The additional 3.6 seconds comes from the hallucination verification steps — specifically running the DeBERTa NLI cross-encoder on each sentence-chunk pair. In the end-to-end breakdown:
> - Retrieval: 0.047s (negligible)
> - LLM generation: ~5.2s (same in both systems)
> - NLI verification: ~4.56s (the bottleneck — new in TrustRAG)
>
> This 3.6s overhead is the price for detecting and suppressing hallucinations. In high-stakes environments — healthcare, legal, research — this trade-off is absolutely acceptable. Future work plans to reduce NLI latency through model distillation.

---

**Q33. How do you measure if a sentence is hallucinated or not for ground truth evaluation?**

> For evaluation, a manually curated test set was constructed with 400 query-response pairs:
> - 250 queries where the answer is grounded in the retrieved evidence (labelled as "Supported")
> - 150 queries specifically designed to induce hallucination — asking about topics barely covered or not in the corpus (labelled as "Hallucinated")
>
> Human annotators reviewed each generated sentence against the retrieved evidence and assigned ground truth labels. The system's Trust Score decisions were then compared against these human labels to compute Precision, Recall, F1, and AUC-ROC.

---

## SECTION 6: Future Work and Limitations

---

**Q34. What are the limitations of your current system?**

> 1. **NLI verification latency:** 4.56 seconds per query is too slow for truly real-time applications. The NLI cross-encoder is the bottleneck.
> 2. **Fixed thresholds:** The Trust Score thresholds (0.75 and 0.50) are currently fixed. Adaptive, query-specific thresholds could improve performance.
> 3. **Corpus dependency:** The system's accuracy depends heavily on the quality and coverage of the knowledge base. If the user asks about a topic not in the corpus, the system will block responses — which is correct behavior, but not helpful.
> 4. **Evaluation scale:** The current benchmark is 400 queries. A larger-scale evaluation across multiple domains would strengthen the results.
> 5. **Single-hop retrieval:** The current retrieval is single-hop — it retrieves once per query. Complex multi-step questions may require multi-hop reasoning across multiple retrieved chunks.

---

**Q35. What are your plans for Semester 3?**

> 1. **Full-scale benchmark evaluation** — Run the complete evaluation on the full MS MARCO test set with larger query counts
> 2. **NLI distillation** — Compress the NLI model further to reduce the 4.56s verification bottleneck
> 3. **Adaptive thresholds** — Research methods to set the Trust Score thresholds dynamically based on query type and corpus confidence
> 4. **Multi-hop retrieval** — Implement iterative retrieval for complex questions that require evidence from multiple documents
> 5. **Comparative evaluation** — Formally compare TrustRAG against SelfCheckGPT, FActScore, and RAGAS on the same test set
> 6. **Paper submission** — Submit the TrustRAG-Opt paper to a conference or journal

---

**Q36. Can this system scale to millions of documents?**

> The current `IndexFlatIP` index has linear search complexity — O(N×d). For very large corpora (millions of documents), this would become slow. The TrustRAG-Opt extension in the journal paper directly addresses this by evaluating:
> - **HNSW** — hierarchical graph-based approximate search; reduces latency from 47.3ms to 3.8ms at Recall@5 of 0.93
> - **IVF-PQ** — partitioned approximate index; reduces memory from 384MB to 57MB
> - **Hybrid FAISS + BM25** — combines dense semantic and sparse keyword search; achieves highest Recall@5 of 0.96
>
> So yes, the architecture is designed to scale — the optimization layer allows swapping the retrieval backend without changing any of the detection logic.

---

## SECTION 7: Concept Questions (Panel May Ask These to Test Understanding)

---

**Q37. What is the difference between Precision and Recall in your context?**

> In hallucination detection:
> - **Precision (0.912):** Of all the sentences my system flagged as hallucinated, 91.2% were actually hallucinations. High precision means few false alarms — I don't wrongly suppress good responses.
> - **Recall (0.884):** Of all the sentences that were truly hallucinated, my system caught 88.4% of them. High recall means few misses — I don't let real hallucinations through.
>
> F1-score (0.898) is the harmonic mean of both — balancing precision and recall.

---

**Q38. What is cosine similarity in plain English?**

> Imagine two arrows pointing in different directions from the same origin. Cosine similarity measures the angle between those arrows. If the arrows point in the same direction, the angle is 0° and cosine = 1.0 (identical meaning). If they point in opposite directions, angle = 180° and cosine = -1.0 (opposite meaning). If they point at 90°, cosine = 0.0 (unrelated meaning). In our system, all vectors are normalized so cosine similarity falls between 0 and 1.

---

**Q39. What is the difference between a cross-encoder and a bi-encoder?**

> A **bi-encoder** encodes two texts independently — each goes through the model separately — and then you compute similarity between the two output vectors. It is fast because you can pre-compute document embeddings. This is what `all-MiniLM-L6-v2` does for retrieval.
>
> A **cross-encoder** takes both texts as a single input, separated by a special token, and encodes them together. The model can directly attend to the interaction between the two texts. This is much more accurate for tasks like NLI where the relationship between the texts matters — but slower because you cannot pre-compute anything. This is what `nli-deberta-v3-small` does for verification.

---

**Q40. Why is your system better than simply using a keyword filter to block bad responses?**

> A keyword filter only blocks responses containing specific bad words or patterns — it cannot understand *meaning*. For example:
> - A response could be perfectly polite and grammatical but factually wrong — a keyword filter would pass it
> - A response might contain technical jargon that looks "suspicious" to a keyword filter but is actually correct
>
> My system reads the *meaning* of each sentence and checks it against *actual retrieved evidence* using NLI and BERTScore — understanding whether the claim is logically and semantically supported, not just whether it contains certain words.

---

**Q41. If a document is not in your knowledge base, what does the system do?**

> The system will attempt retrieval anyway. The retrieved chunks will have low cosine similarity scores — indicating that nothing relevant was found. The LLM will generate a response based on this weak context, which will likely receive low Trust Scores across NLI, BERTScore, and Cosine. The Decision Controller will then trigger STATE 3 — Block — and tell the user: "I cannot find reliable evidence in the provided documents." This is the correct and safe behavior — it prevents the system from making up an answer when it has no evidence.

---

**Q42. What happens if two retrieved chunks contradict each other?**

> This is an interesting edge case. If the retrieved chunks contradict each other, the LLM might generate a response that agrees with one chunk but contradicts another. In this case:
> - The NLI score would be high against the agreeing chunk (ENTAILMENT) but low against the contradicting chunk (CONTRADICTION)
> - The system attributes each generated sentence to its *best matching* chunk
> - The Trust Score reflects the best available evidence support
>
> This is a known limitation — handling intra-corpus contradictions is an open research problem. In Semester 3, I plan to investigate multi-hop evidence aggregation which would help handle such cases.

---

**Q43. How does your system decide which chunk "supports" which generated sentence?**

> For each generated sentence, the system computes cosine similarity between the sentence's embedding and every retrieved chunk's embedding. The chunk with the highest similarity score is assigned as the "supporting chunk" for that sentence. This mapping is called chunk attribution. The attribution score (A_ij) directly measures semantic closeness between what was said and what the evidence contains.

---

**Q44. What is the hallucination ratio and how is it used?**

> The hallucination ratio is: number of sentences labelled "Hallucinated" divided by total number of sentences in the response.
>
> Example: Response has 6 sentences. 2 are labelled Hallucinated. HR = 2/6 = 0.33 (33%).
>
> If HR > 0.50 (more than half the sentences are hallucinated), the system blocks the entire response regardless of the overall Trust Score. This prevents a situation where the overall score looks acceptable but the majority of individual sentences are actually untrustworthy.

---

**Q45. Is this system applicable to domains beyond technical documents?**

> Yes, in principle. The pipeline is domain-agnostic — it works with any text documents. If you replace the corpus with medical literature, legal documents, or financial reports, the same pipeline would work. The main requirement is that the relevant documents are available in the local corpus. The NLI model and BERTScore are general-purpose and work across domains. However, domain-specific fine-tuning of the embedding model might improve retrieval accuracy for specialized terminology.

---

**Q46. What would you do differently if you were to restart this project?**

> I would structure the evaluation framework earlier in the development process — setting up the benchmark test set and metrics from the beginning rather than implementing the pipeline first and evaluation later. This would have allowed me to measure improvements more precisely at each development stage. I would also start with HNSW indexing instead of FlatIP from the beginning, since HNSW is both faster and practically equivalent in accuracy for our corpus size.

---

*End of Q&A Guide — Prepared for M.Tech Project Panel Review 2, Amrita Vishwa Vidyapeetham*
