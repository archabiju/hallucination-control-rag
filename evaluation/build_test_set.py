"""
Builds evaluation/test_set.json — a hand-labelled claim-level benchmark grounded in data/corpus.

Labelling rule (faithfulness to the corpus, the standard RAG-evaluation convention):
  hallucinated = False : the claim is stated by, or directly entailed by, the corpus.
  hallucinated = True  : the claim contradicts the corpus (intrinsic) or adds a specific
                         detail the corpus never states (extrinsic).
`source` lists the corpus file(s) that contain the evidence, used for retrieval metrics.
"""
import json
from pathlib import Path

S, H = False, True
T = []
def q(question, source, *claims):
    T.append({"question": question, "relevant_sources": source,
              "claims": [{"text": t, "hallucinated": h} for t, h in claims]})

q("What is Retrieval-Augmented Generation?", ["retrieval_augmented_generation.md"],
  ("RAG combines a retriever that searches an external knowledge source with a generator that produces text.", S),
  ("A RAG system looks up relevant passages at query time and gives them to the language model as context.", S),
  ("RAG completely eliminates hallucination in language models.", H),
  ("RAG requires retraining the language model every time the knowledge base is updated.", H))
q("Why is RAG not a guarantee against hallucination?", ["retrieval_augmented_generation.md"],
  ("A language model can ignore or misread the retrieved context and still produce unsupported claims.", S),
  ("Hallucination-detection layers check after generation whether each sentence is backed by the retrieved evidence.", S),
  ("A model that receives retrieved context can never generate a claim unsupported by any chunk.", H))
q("Can RAG run offline?", ["retrieval_augmented_generation.md", "local_llm_inference_ollama.md"],
  ("RAG can run entirely offline when the embedding model, vector index and generator are all hosted locally.", S),
  ("Offline RAG typically uses smaller quantized open-weight models served through a runtime such as Ollama.", S),
  ("Offline RAG requires a cloud API for the generator model.", H))
q("What does the typical RAG pipeline consist of?", ["retrieval_augmented_generation.md"],
  ("Documents are chunked, embedded and stored in a vector index, then the query is embedded and similar chunks are retrieved.", S),
  ("The retrieved chunks are inserted into a prompt together with the question for the language model.", S),
  ("The typical RAG pipeline has seven stages, starting with manual annotation of every document.", H))
q("What is a sentence embedding?", ["sentence_embeddings.md"],
  ("A sentence embedding is a fixed-length numeric vector that represents the meaning of a piece of text.", S),
  ("Texts with similar meaning end up close together in the embedding vector space.", S),
  ("Sentence embeddings only capture which words appear in a text, like bag-of-words.", H))
q("What is all-MiniLM-L6-v2?", ["sentence_embeddings.md"],
  ("all-MiniLM-L6-v2 produces 384-dimensional vectors.", S),
  ("all-MiniLM-L6-v2 is about 80 megabytes and runs comfortably on a CPU.", S),
  ("all-MiniLM-L6-v2 produces 768-dimensional vectors and needs a GPU.", H))
q("Why are embeddings normalized before indexing?", ["sentence_embeddings.md", "faiss_vector_databases.md"],
  ("When vectors are normalized to unit length, a dot product is equivalent to cosine similarity.", S),
  ("Normalizing lets a fast dot-product search stand in for cosine similarity at no extra query-time cost.", S),
  ("Normalized embeddings make the dot product equal to the Euclidean distance.", H))
q("How is semantic search better than keyword search?", ["sentence_embeddings.md"],
  ("Semantic search can retrieve a relevant passage even when the query and passage share almost no exact words.", S),
  ("Keyword search is more accurate than semantic search when the query uses different vocabulary from the passage.", H))
q("What is FAISS?", ["faiss_vector_databases.md"],
  ("FAISS is an open-source similarity-search library developed by Meta AI.", S),
  ("FAISS runs entirely in memory on one machine with no external server or network call.", S),
  ("FAISS requires a separate database server and credentials to operate.", H),
  ("FAISS was developed by Google Research.", H))
q("What index types does FAISS provide?", ["faiss_vector_databases.md"],
  ("IndexFlatIP and IndexFlatL2 perform exact brute-force search over every stored vector.", S),
  ("IVF and HNSW are approximate nearest-neighbor structures that trade a little accuracy for speed.", S),
  ("IndexFlatIP is an approximate index that skips most stored vectors.", H))
q("Does a FAISS index update automatically when documents change?", ["faiss_vector_databases.md"],
  ("An application must explicitly re-embed and re-add or rebuild the index when the document store changes.", S),
  ("FAISS automatically detects deleted documents and removes their vectors in the background.", H))
q("What is hallucination in large language models?", ["hallucination_in_llms.md"],
  ("Hallucination is generated content that is fluent and confident-sounding but factually incorrect or unsupported.", S),
  ("Language models are trained to predict plausible next tokens, not to verify truth.", S),
  ("Hallucinations occur only when a model is malfunctioning and produces visibly broken text.", H))
q("What is the difference between intrinsic and extrinsic hallucination?", ["hallucination_in_llms.md"],
  ("Intrinsic hallucinations directly contradict the source material the model was given.", S),
  ("Extrinsic hallucinations introduce information that cannot be verified from the source.", S),
  ("Intrinsic hallucinations are claims that are invented from the model's training data.", H))
q("What causes hallucination?", ["hallucination_in_llms.md"],
  ("Training data contains errors and outdated information that models can absorb as established fact.", S),
  ("Models can show sycophancy, generating an answer that matches what the question seems to expect.", S),
  ("Hallucination is caused exclusively by too small a context window.", H))
q("How can hallucination be mitigated?", ["hallucination_in_llms.md"],
  ("Grounding the model in retrieved evidence and prompting it to say it does not know are pre-generation mitigations.", S),
  ("Post-generation mitigation verifies each claim with entailment classifiers or similarity scores.", S),
  ("Post-generation verification is considered a weaker safeguard than prompting alone.", H))
q("What is Natural Language Inference?", ["natural_language_inference.md"],
  ("NLI determines the logical relationship between a premise and a hypothesis.", S),
  ("NLI classifies a hypothesis as entailment, contradiction or neutral given the premise.", S),
  ("NLI classifies pairs into five graded levels of similarity.", H))
q("How is NLI used for hallucination detection?", ["natural_language_inference.md"],
  ("The generated sentence is treated as the hypothesis and the retrieved chunk as the premise.", S),
  ("A contradiction score means the generated sentence conflicts with the evidence.", S),
  ("A neutral score means the sentence is fully supported by the evidence.", H))
q("What is a cross-encoder NLI model?", ["natural_language_inference.md"],
  ("A cross-encoder processes the premise and hypothesis together in a single forward pass.", S),
  ("Cross-encoders are more accurate than comparing separate sentence embeddings, but slower.", S),
  ("Cross-encoders reuse precomputed sentence vectors so they are faster than embedding comparison.", H))
q("What is BERTScore?", ["bertscore_similarity_metrics.md"],
  ("BERTScore measures how similar two texts are in meaning rather than in exact wording.", S),
  ("BERTScore uses contextual token embeddings and cosine similarity between best-matching tokens.", S),
  ("BERTScore counts shared n-grams between candidate and reference text.", H))
q("How is BERTScore F1 computed?", ["bertscore_similarity_metrics.md"],
  ("Precision matches each candidate token to its most similar reference token and recall does the reverse.", S),
  ("F1 is the harmonic mean of the two directional scores.", S),
  ("F1 is the arithmetic mean of precision and recall computed from n-gram counts.", H))
q("Does BERTScore need training data or internet access?", ["bertscore_similarity_metrics.md"],
  ("BERTScore needs only a pretrained language model and can run offline once that model is downloaded.", S),
  ("BERTScore must be fine-tuned on a labelled hallucination dataset before use.", H))
q("What is XAI in a retrieval system?", ["explainable_ai_xai.md"],
  ("In retrieval systems XAI explains why a chunk was retrieved and why the answer took its shape.", S),
  ("Chunk attribution labels which source document and page a piece of retrieved text came from.", S),
  ("XAI in retrieval systems is only possible by training an additional black-box classifier.", H))
q("What is word-level saliency?", ["explainable_ai_xai.md"],
  ("Word saliency removes a word and recomputes similarity to see how much the word contributes.", S),
  ("A sharp drop in similarity after removing a word means the word anchors the claim to the source.", S),
  ("A word whose removal sharply lowers similarity is likely filler or an unsupported addition.", H))
q("What is provenance scoring?", ["explainable_ai_xai.md"],
  ("Provenance scoring measures what fraction of the generated response is traceable to retrieved evidence.", S),
  ("Provenance scoring measures how fast the retriever returns results.", H))
q("What is the three-state decision scheme?", ["tri_state_decision_control.md"],
  ("A score at or above the high threshold is classified as Supported and shown with its citation.", S),
  ("A score between the medium and high thresholds is Weakly Supported and shown with a warning.", S),
  ("A score below the medium threshold is classified as Weakly Supported.", H))
q("When is an entire response blocked?", ["tri_state_decision_control.md"],
  ("A response is blocked when more than a set fraction of its sentences, commonly fifty percent, are Hallucinated.", S),
  ("When a response is blocked the system surfaces the raw retrieved evidence instead.", S),
  ("A response is blocked as soon as any single sentence is classified as Hallucinated.", H))
q("Are the decision thresholds fixed?", ["tri_state_decision_control.md"],
  ("Both thresholds are tunable parameters depending on tolerance for false blocks versus false passes.", S),
  ("Raising the high threshold makes the system more conservative.", S),
  ("The thresholds are fixed constants that must not be changed.", H))
q("What is chunking and why does it matter?", ["chunking_strategies.md"],
  ("Chunking splits documents into pieces because embedding models have a maximum input length.", S),
  ("Chunks that are too small lose context needed to interpret a sentence.", S),
  ("Chunk size has no measurable effect on retrieval quality.", H))
q("What is a common chunking configuration?", ["chunking_strategies.md"],
  ("A common configuration is 512 tokens per chunk with a 50-token overlap.", S),
  ("Overlap ensures content near a cut point appears in full in at least one chunk.", S),
  ("The common configuration uses 128 tokens per chunk with no overlap.", H))
q("What alternatives to fixed-size chunking exist?", ["chunking_strategies.md"],
  ("Semantic chunking uses embedding similarity between adjacent sentences to find topic boundaries.", S),
  ("Recursive chunking splits by sections, then paragraphs, then sentences before falling back to fixed size.", S),
  ("Sentence-based chunking produces chunks of exactly equal size.", H))
q("What do precision and recall measure for hallucination detection?", ["evaluation_metrics.md"],
  ("Precision is the fraction of flagged claims that were actually hallucinated.", S),
  ("Recall is the fraction of actual hallucinations that the system caught.", S),
  ("Precision is the fraction of all hallucinated claims that the system caught.", H))
q("What does AUROC measure?", ["evaluation_metrics.md"],
  ("AUROC evaluates a system across every possible decision threshold at once.", S),
  ("An AUROC of 0.5 means the score is no better than a random guess.", S),
  ("An AUROC of 0.5 means the two classes are perfectly separable.", H))
q("Which error is more costly in hallucination detection?", ["evaluation_metrics.md"],
  ("A missed hallucination is generally more costly than an unnecessarily flagged correct claim.", S),
  ("Reporting only accuracy on an imbalanced test set can be misleading.", S),
  ("False positives are generally considered more costly than false negatives.", H))
q("What is Ollama?", ["local_llm_inference_ollama.md"],
  ("Ollama exposes a local HTTP API, by default at http://localhost:11434.", S),
  ("Ollama handles quantization so open-weight models use less memory.", S),
  ("Ollama sends prompts to a hosted cloud service for inference.", H))
q("What are the trade-offs of local inference?", ["local_llm_inference_ollama.md"],
  ("No data leaves the machine, so local inference suits privacy-sensitive environments.", S),
  ("Local open-weight models are typically smaller and slower than the largest proprietary models.", S),
  ("Local inference incurs a per-token cost and an external rate limit.", H))
q("Why is prompt engineering not enough to prevent hallucination?", ["prompt_engineering.md", "local_llm_inference_ollama.md"],
  ("A language model has no mechanism that enforces a grounding instruction the way software enforces a rule.", S),
  ("Prompt engineering is a first line of defense while independent verification catches what the prompt failed to prevent.", S),
  ("A well-written grounding prompt guarantees the model never uses unretrieved facts.", H))
q("How can prompts reduce hallucination?", ["prompt_engineering.md"],
  ("Labelling each retrieved passage with its source and page helps the model cite correctly.", S),
  ("Lowering the generation temperature reduces the tendency to embellish.", S),
  ("Raising the temperature makes the model stick more closely to the provided context.", H))

Path("evaluation/test_set.json").write_text(json.dumps(T, indent=2, ensure_ascii=False))
n = sum(len(t["claims"]) for t in T); h = sum(c["hallucinated"] for t in T for c in t["claims"])
print(f"{len(T)} questions, {n} claims ({h} hallucinated, {n-h} supported)")
