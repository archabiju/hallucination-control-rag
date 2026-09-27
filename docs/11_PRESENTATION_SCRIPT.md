# Slide-by-Slide Presentation Script — Phase 1 Review
**TrustRAG | CB.SC.P2CSE25001 | Archa Biju | Amrita Vishwa Vidyapeetham**

> ⚠️ **Phase-1 Guideline:** This script is calibrated for a Phase-1 review.
> - Present only what is **actually completed**
> - Clearly say what is **in progress**
> - Clearly say what is **planned for Semester 3**
> - Do NOT present results, ablation studies, or comparisons — those belong to Phase 2

---

## SLIDE 1 — Title Slide

**📢 What to say:**
> "Good morning / Good afternoon to the respected panel members and my guide, Dr. Ramya G R.
> I am Archa Biju, and this is my Phase 1 review of my M.Tech project titled:
> **TrustRAG — A Trustworthy Offline Retrieval-Augmented Generation Framework with Explainable Multi-Signal Hallucination Detection.**
> I will present my motivation, the problem I am addressing, my proposed methodology, the current implementation progress, and the plan for Semester 3."

**⏱️ Time:** 30 seconds

---

## SLIDE 2 — Guide Approval Mail

**📢 What to say:**
> "This slide confirms the guide approval from Dr. Ramya G R for the project topic and scope, received on 27th August 2026."

**⏱️ Time:** 15 seconds — move quickly

---

## SLIDE 3 — Introduction and Motivation

**📢 What to say:**
> "Let me begin with the context.
>
> Retrieval-Augmented Generation — RAG — is a method where an AI system first searches a document database for relevant passages, and then generates an answer based on those passages. It was introduced to reduce the problem of AI models making up facts from their training memory.
>
> However, the key limitation is that retrieval alone does not guarantee that every generated statement is actually supported by the retrieved evidence. The model can still go beyond or ignore the evidence.
>
> This motivated me to design a system that does not just retrieve evidence, but also:
> - Explains which documents were selected and why
> - Verifies each sentence of the generated response against the evidence
> - Detects and controls unsupported claims
> - And operates completely offline — no cloud APIs, no external services
>
> The goal is a knowledge system that is trustworthy and transparent, not just a response generator."

**👀 Point at:** Each bullet point as you say it.
**⏱️ Time:** 1.5 minutes

---

## SLIDE 4 — Problem Statement

**📢 What to say:**
> "The core problem is this:
>
> Even when an AI system is given retrieved documents as context, it can still generate statements that are unsupported or only weakly supported by those documents. This is the hallucination problem in RAG systems.
>
> Existing approaches to detect and fix this have three drawbacks:
>
> **One** — they use external fact-checking models that need internet access.
> **Two** — they rely on heavyweight LLMs or cloud APIs for verification — which introduces high computational cost and data privacy risks.
> **Three** — they lack transparency — they often give a binary hallucinated/not-hallucinated label with no explanation of which part of the evidence was used.
>
> These limitations make existing systems unsuitable for offline, high-security, or resource-constrained environments.
>
> My project addresses all three: a lightweight, fully local, explainable system that works without any external service."

**👀 Point at:** Each of the three bullet points.
**⏱️ Time:** 1.5 minutes

---

## SLIDE 5 & 6 — Literature Review

**📢 What to say:**
> "I reviewed the core literature in the areas of RAG, hallucination detection, and explainability.
>
> The key papers I studied are:
>
> **Lewis et al. (2020)** — the original RAG paper that introduced retrieval-augmented generation for knowledge-intensive tasks. This is the foundation of my retrieval pipeline.
>
> **Reimers and Gurevych (2019)** — Sentence-BERT, which I use to encode text into semantic vectors for retrieval.
>
> **Johnson et al. (2021)** — FAISS, the vector search library I use for offline similarity retrieval.
>
> **Zhang et al. (2020)** — BERTScore, which I plan to use as one of my three verification signals for measuring semantic similarity between generated text and evidence.
>
> **Manakul et al. (2023)** — SelfCheckGPT, which detects hallucinations through consistency checking across multiple LLM outputs — but requires multiple generation passes and internet access, which is a limitation my work addresses.
>
> **Es et al. (2024)** — RAGAS, an evaluation framework for RAG systems.
>
> These papers together define the landscape of the problem and guided my design choices."

**👀 Point at:** Paper titles/authors in the table.
**⏱️ Time:** 1.5 minutes

---

## SLIDE 7 — Research Gap

**📢 What to say:**
> "Based on the literature, I identified four specific gaps:
>
> **Gap 1:** Existing RAG systems do not make use of the internal signals that are already generated during retrieval — like cosine similarity scores and chunk relevance rankings. These signals could directly indicate evidence quality, but they are discarded after retrieval.
>
> **Gap 2:** There is no claim-level evidence support assessment within the RAG pipeline. Existing systems evaluate the response as a whole — not sentence by sentence.
>
> **Gap 3:** Most detection methods are independent post-hoc modules attached after generation, not intrinsically integrated with the retrieval process.
>
> **Gap 4:** There is no unified, lightweight, offline-capable framework that connects retrieval explainability with generation reliability in a single end-to-end pipeline.
>
> My proposed system, TrustRAG, is designed to fill all four of these gaps."

**👀 Point at:** Each bullet.
**⏱️ Time:** 1.5 minutes

---

## SLIDE 8 — Architecture Diagram

**📢 What to say:**
> "This is the proposed system architecture of TrustRAG. Let me walk through the flow step by step.
>
> **Step 1 — Query Embedding:** The user's question is converted into a 384-dimensional dense vector using Sentence-BERT, specifically the `all-MiniLM-L6-v2` model. This runs locally.
>
> **Step 2 — Semantic Retrieval:** The query vector is compared against all stored document chunk vectors in the FAISS index using cosine similarity. The Top-5 most relevant chunks are retrieved.
>
> **Step 3 — Explainability Layer (XAI):** The XAI layer captures internal signals from the retrieval — how similar each chunk is to the query, and which chunk best supports each generated sentence. These signals are used later for verification.
>
> **Step 4 — Local Generation:** The retrieved evidence is passed to Mistral-7B, running locally through Ollama, to generate a grounded response.
>
> **Step 5 — Claim Segmentation:** The generated response is split into individual sentences — each sentence is treated as a separate claim to be verified.
>
> **Step 6 — Multi-Signal Detection:** Each sentence is verified against the evidence using three signals: NLI entailment, BERTScore, and cosine similarity.
>
> **Step 7 — Trust Score:** The three signals are combined into a single Trust Score per sentence.
>
> **Step 8 — Tri-State Decision:** Based on the Trust Score, the decision controller outputs one of three responses: answer with citations, partial answer with warnings, or blocked response.
>
> The entire flow — from query to final verified output — runs without any internet or cloud dependency."

**👀 Point at:** Each component in the architecture diagram as you name it. Trace the arrows.
**⏱️ Time:** 2.5 minutes

---

## SLIDE 9 — Architecture & Workflow

**📢 What to say:**
> "This slide shows the end-to-end workflow in a more detailed layout.
>
> The left side shows the data preparation: documents are loaded, chunked into 512-token windows with 50-token overlap, embedded using Sentence-BERT, and indexed into FAISS.
>
> The right side shows the query-time flow: the query is embedded, Top-K chunks are retrieved, the XAI layer captures attribution signals, the LLM generates a response, the response is segmented into claims, and each claim is verified.
>
> All components in this workflow are defined and the core retrieval path — from ingestion to FAISS retrieval — is implemented and working."

**👀 Point at:** Left half (data prep) then right half (query flow).
**⏱️ Time:** 1 minute

---

## SLIDE 10 — Technical Method Part 1 (Retrieval + Attribution)

**📢 What to say:**
> "Now the mathematical formulation of the methodology.
>
> The user query q is embedded using Sentence-BERT to get vector e_q.
>
> Cosine similarity between the query and each document chunk d_i is: S(q, d_i) = e_q dot e_i, since all vectors are L2-normalized. The Top-K chunks with the highest similarity scores are selected: R(q) = TopK(S(q, d_i)).
>
> The Evidence Attribution step: for each generated sentence g_j, its embedding is compared with each retrieved chunk embedding to compute A_ij. The chunk with the highest A_ij score is identified as the supporting source for that sentence. This is what enables inline citations.
>
> The retrieved context is then passed to Mistral-7B via Ollama to generate the response.
>
> Finally, the response is split into individual sentences G = {g1, g2, ..., gm} for claim-level verification."

**👀 Point at:** Each equation as you say it.
**⏱️ Time:** 2 minutes

---

## SLIDE 11 — Technical Method Part 2 (Detection + Trust Score + Decision)

**📢 What to say:**
> "For each generated sentence g_j, three verification signals are computed:
>
> **NLI Entailment (S_NLI,j):** A DeBERTa cross-encoder takes the retrieved chunk as the premise and the generated sentence as the hypothesis, and classifies the logical relationship as Entailment, Neutral, or Contradiction. The entailment probability is used as S_NLI.
>
> **BERTScore (S_BERT,j):** Token-level semantic similarity between the generated sentence and the evidence is computed using BERTScore.
>
> **Cosine Similarity (S_COS,j):** The generated sentence's embedding is compared with the mean of the retrieved chunk embeddings, giving an embedding-level alignment score.
>
> These three signals are combined into the Trust Score: T_j = w1 times S_NLI plus w2 times S_BERT plus w3 times S_COS, where w1 + w2 + w3 = 1.
>
> The Tri-State Decision Controller then classifies each sentence:
> - If T_j is 0.75 or above → Supported — the answer is shown with a citation
> - If T_j is between 0.50 and 0.75 → Weakly Supported — shown with a warning
> - If T_j is below 0.50 → Hallucinated — the sentence is suppressed
>
> And at the response level, if more than 50% of sentences are hallucinated, the entire response is blocked and only the verified evidence is returned to the user."

**👀 Point at:** Each formula. Point clearly at the Trust Score box and the tri-state decision.
**⏱️ Time:** 2.5 minutes

---

## SLIDE 12 — Datasets

**📢 What to say:**
> "I am using two datasets in this project.
>
> **Dataset 1 — Local Technical Knowledge Corpus:** This consists of 9 domain-specific research papers and technical documents on RAG and hallucination detection, totalling around 6.7 megabytes. These documents form the initial knowledge base. They were used for early development — testing the retrieval, chunking, embedding, and the XAI layer.
>
> **Dataset 2 — MS MARCO:** MS MARCO stands for Microsoft Machine Reading Comprehension. It is a publicly available large-scale dataset with real user queries from Bing and web passages. I selected a subset of 10,000 passages for local offline indexing. This is used as a benchmark dataset because it allows retrieval results to be compared with published research.
>
> Both datasets go through the same pipeline — chunking, Sentence-BERT embedding, and FAISS indexing. They are stored together in one FAISS vector index."

**👀 Point at:** Dataset 1 section, then Dataset 2 section, then the pipeline figure.
**⏱️ Time:** 1.5 minutes

---

## SLIDE 13 — Implementation Progress, Partial Results & Discussion

> ⚠️ **Critical slide — be precise about what is done vs. in progress. Do not overclaim.**

**📢 What to say:**
> "Let me now present where the implementation stands at Phase 1.
>
> **What is completed:**
> - Document ingestion: loading PDF, DOCX, and TXT files from the corpus — done.
> - Text chunking: 512-token sliding window with 50-token overlap — done.
> - Sentence-BERT embedding: all chunks embedded using `all-MiniLM-L6-v2` — done.
> - FAISS vector indexing: the local knowledge base is indexed and saved to disk — done.
> - MS MARCO preprocessing: 10,000 passages cleaned and added to the index — done.
> - Top-K semantic retrieval: the system can retrieve the 5 most relevant chunks for any query — done.
> - The retrieval is fully offline — no cloud API is used at any point.
>
> **What is currently in progress:**
> - The Explainability Layer — chunk attribution and provenance score computation — is under development.
> - The NLI detector, BERTScore detector, and Trust Score computation have been designed and implemented at the module level, but full pipeline integration is in progress.
> - The tri-state decision controller is implemented but is being tested with the full pipeline.
>
> **Preliminary observation:**
> - The FAISS retrieval correctly returns relevant document chunks for test queries.
> - Early testing shows that high retrieval similarity does not always guarantee that every generated sentence is fully supported. This confirms the necessity of the claim-level verification layer.
>
> **What is planned for Semester 3:**
> - Full pipeline integration and end-to-end testing
> - Formal evaluation using F1, Recall@K, and Provenance Score
> - Comparative analysis against baseline approaches"

**👀 Point at:** Completed items ✅, In Progress items 🔄, Planned items — as you say each group.
**⏱️ Time:** 2.5 minutes

---

## SLIDE 14 — SDG Mapping

**📢 What to say:**
> "My project aligns with two Sustainable Development Goals.
>
> **SDG 9 — Industry, Innovation and Infrastructure:** TrustRAG contributes to building reliable AI infrastructure by providing a trustworthy, verifiable, and explainable knowledge retrieval system.
>
> **SDG 16 — Peace, Justice and Strong Institutions:** The system is designed for high-security, privacy-sensitive environments — like healthcare or government institutions — where data cannot leave the local system. This supports accountable and transparent use of AI in institutional settings."

**👀 Point at:** The SDG icons and justification text.
**⏱️ Time:** 45 seconds

---

## SLIDE 15 — Gantt Chart / Project Plan for Semester 3

**📢 What to say:**
> "This is the plan for Semester 3.
>
> In the first phase, I will complete the full pipeline integration — connecting the XAI layer, NLI detector, BERTScore detector, Trust Score computation, and tri-state controller into one working end-to-end system.
>
> In the second phase, I will run formal evaluation — computing metrics like F1-score, Recall@K, and Provenance Score on a test set.
>
> In the third phase, I plan to investigate retrieval optimizations using HNSW indexing and explore lighter NLI models to reduce verification latency.
>
> The final phase is thesis writing and paper preparation."

**👀 Point at:** Each phase on the Gantt chart.
**⏱️ Time:** 1 minute

---

## SLIDE 16 — Domain Expertise Course

**📢 What to say:**
> "For the Domain Expertise requirement, I enrolled in the course **RAG with Python: Build Chatbots That Talk to Your Data** on Udemy, with a duration of 10 hours and 34 minutes.
>
> The course is directly relevant to my project — it covers document processing, chunking, vector embeddings, FAISS retrieval, and evidence-grounded generation. The knowledge from this course was applied in the implementation of the data ingestion and retrieval pipeline."

**⏱️ Time:** 45 seconds

---

## SLIDE 17 — Technical Enrichment Activities

**📢 What to say:**
> "For Technical Enrichment, I studied and worked with the core libraries used in the project — Sentence-Transformers, FAISS, HuggingFace Transformers, and BERTScore — and reviewed the five primary reference papers to understand the theoretical basis of each component before implementing it."

**⏱️ Time:** 30 seconds

---

## SLIDE 18 — Conclusion

**📢 What to say:**
> "To summarize what I have presented today:
>
> **The Problem:** RAG systems can still hallucinate even with retrieved evidence. Existing detection methods are cloud-dependent, heavyweight, and lack explainability.
>
> **My Proposed Solution — TrustRAG:** A fully offline framework that verifies each generated sentence against retrieved evidence using three signals — NLI entailment, BERTScore, and cosine similarity — combined into a Trust Score. The tri-state decision controller then outputs a verified, warned, or blocked response.
>
> **Phase 1 Status:** The retrieval pipeline — document ingestion, chunking, embedding, and FAISS indexing — is complete. The detection modules are implemented at the module level and are being integrated.
>
> **Next Phase:** Full integration, formal evaluation, and optimization of the verification pipeline.
>
> The goal is a system that is trustworthy, explainable, and deployable in offline environments without any external API dependency."

**👀 Point at:** Each conclusion bullet.
**⏱️ Time:** 1.5 minutes

---

## SLIDE 19 — References

**📢 What to say:**
> "These are the ten key references that informed the design of TrustRAG — spanning RAG foundations, semantic retrieval, hallucination detection, and explainability. I am happy to discuss any specific paper if the panel has questions."

**⏱️ Time:** 15 seconds — do not read them out individually

---

## 🎯 Closing Statement

**📢 What to say:**
> "Thank you for your time and attention. I am happy to answer any questions from the panel."

---

## ⏱️ Time Summary

| Section | Slides | Estimated Time |
|:---|:---|:---|
| Title + Approval | 1–2 | 45 sec |
| Introduction + Problem | 3–4 | 3 min |
| Literature + Research Gap | 5–7 | 3 min |
| Architecture + Workflow | 8–9 | 3.5 min |
| Technical Method | 10–11 | 4.5 min |
| Datasets | 12 | 1.5 min |
| Implementation Status | 13 | 2.5 min |
| SDG + Gantt + Domain + Enrichment | 14–17 | 2.5 min |
| Conclusion + References | 18–19 | 2 min |
| **Total** | **19 slides** | **~23 minutes** |

---

## ✅ Phase-1 Status Reference Card

Use this when answering panel questions about what is done:

| Component | Status |
|:---|:---|
| Document loading (PDF, DOCX, TXT) | ✅ Complete |
| Text chunking (512 tokens, 50 overlap) | ✅ Complete |
| Sentence-BERT embedding (all-MiniLM-L6-v2) | ✅ Complete |
| FAISS IndexFlatIP vector database | ✅ Complete |
| MS MARCO preprocessing (10,000 passages) | ✅ Complete |
| Top-K semantic retrieval | ✅ Complete |
| Offline operation (no cloud APIs) | ✅ Complete |
| XAI Layer — chunk attribution + provenance | 🔄 In Progress |
| NLI detector (DeBERTa cross-encoder) | 🔄 Module done, pipeline integration in progress |
| BERTScore detector | 🔄 Module done, pipeline integration in progress |
| Hallucination engine (Trust Score) | 🔄 Module done, pipeline integration in progress |
| Tri-state Decision Controller | 🔄 Module done, pipeline integration in progress |
| Local LLM generation (Mistral-7B, Ollama) | 🔄 In Progress |
| Streamlit UI dashboard | 🔄 In Progress |
| Full end-to-end pipeline integration | 📋 Planned — Semester 3 |
| Formal evaluation (F1, Recall@K, AUC-ROC) | 📋 Planned — Semester 3 |
| Comparative analysis vs. baselines | 📋 Planned — Semester 3 |
