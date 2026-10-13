"""Build the camera-ready clarification Word document from the measured run."""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

OUT = Path(__file__).resolve().parent.parent / "TrustRAG-Opt_Reviewer_Clarification.docx"
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
RULE = "1F3A5F"


def shade(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def set_run_font(run, name="Times New Roman", size=11, bold=False, italic=False, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def add_p(doc, text, size=11, bold=False, italic=False, space_after=8, space_before=0, center=False, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing = 1.15
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold, italic=italic, color=color)
    return p


def add_rich(doc, parts, space_after=8, space_before=0):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing = 1.15
    for text, kwargs in parts:
        r = p.add_run(text)
        set_run_font(r, **kwargs)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_run_font(r, size=14 if level == 1 else 12, bold=True, color=NAVY)
    return p


def bullet(doc, text, bold_lead=None):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.first_line_indent = Cm(-0.35)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.1
    r = p.add_run("• ")
    set_run_font(r, size=11)
    if bold_lead:
        r2 = p.add_run(bold_lead)
        set_run_font(r2, size=11, bold=True)
        r3 = p.add_run(text)
        set_run_font(r3, size=11)
    else:
        r2 = p.add_run(text)
        set_run_font(r2, size=11)
    return p


def comment(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.4)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.space_before = Pt(2)
    r = p.add_run("Reviewer comment. ")
    set_run_font(r, size=11, bold=True, italic=True)
    r2 = p.add_run(text)
    set_run_font(r2, size=11, italic=True)
    return p


def set_cell(cell, text, bold=False, size=9, center=True, fill=None, color=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold, color=color or (RGBColor(255, 255, 255) if fill == RULE else None))
    if fill:
        shade(cell, fill)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True
    for i, h in enumerate(headers):
        set_cell(table.rows[0].cells[i], h, bold=True, size=9, fill=RULE, color=RGBColor(255, 255, 255))
    for r_i, row in enumerate(rows):
        fill = "F4F7FB" if r_i % 2 == 0 else "FFFFFF"
        for c_i, val in enumerate(row):
            set_cell(table.rows[r_i + 1].cells[c_i], val, size=9, fill=fill, center=(c_i != 0))
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table


def footer(section):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("TrustRAG-Opt camera-ready clarification  ·  measured 13 October 2026  ·  page ")
    set_run_font(r, size=9, color=RGBColor(0x55, 0x55, 0x55))
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run = p.add_run()
    set_run_font(run, size=9, color=RGBColor(0x55, 0x55, 0x55))
    run._r.append(fld_begin)
    run2 = p.add_run()
    set_run_font(run2, size=9, color=RGBColor(0x55, 0x55, 0x55))
    run2._r.append(instr)
    run3 = p.add_run()
    set_run_font(run3, size=9, color=RGBColor(0x55, 0x55, 0x55))
    run3._r.append(fld_end)


def build():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(1.9)
    section.right_margin = Cm(1.9)
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    footer(section)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)

    add_p(doc, "TrustRAG-Opt", size=20, bold=True, center=True, space_after=2, color=NAVY)
    add_p(doc, "Clarifications for the camera-ready version", size=14, center=True, space_after=2)
    add_p(doc, "Response to the ICMLDE reviewer comments, from the GitHub repository and the repeated-run measurements", size=12, italic=True, center=True, space_after=2)
    add_p(doc, "Archa Biju and Ramya G R", size=12, center=True, space_after=2)
    add_p(doc, "Department of Computer Science and Engineering, Amrita School of Computing, Coimbatore", size=11, italic=True, center=True, space_after=2)
    add_p(doc, "13 October 2026", size=11, center=True, space_after=12)

    add_p(
        doc,
        "This note answers every reviewer comment on the accepted paper “TrustRAG-Opt: Efficient Retrieval Optimization for Explainable Offline Retrieval-Augmented Generation.” Section 1 states what the GitHub repository itself contains. The tables in Section 6 come from a repeated run of that code on 13 October 2026. Variance was computed on that run. It was not attached to the submitted point estimates, because that run does not recover the 9,800-chunk, 400-query experiment described in the submission.",
    )

    heading(doc, "1. Answers from the GitHub repository")
    add_p(
        doc,
        "These answers are what https://github.com/archabiju/hallucination-control-rag shows on branch master, commit a7dd4d1. It is the only repository on the author’s GitHub account for this project. The five-run latency tables later in this note were produced locally from that same code. They are not yet committed to the repository.",
    )

    heading(doc, "1.1  Base system and what this paper adds", level=2)
    add_p(doc, "Reviewer 1 asked for a citation of the earlier TrustRAG framework, because the paper calls this work an extension.", italic=True)
    add_p(
        doc,
        "The base system is this repository. README.md describes offline retrieval with all-MiniLM-L6-v2, FAISS, NLI, BERTScore, cosine similarity, and the tri-state controller. The repository does not contain a published bibliographic entry for an earlier TrustRAG paper. The four-index comparison is written in ieee_paper/ and my_manuscripts/. The running retrieval code, src/retrieval/faiss_store.py, implements only IndexFlatIP.",
    )

    heading(doc, "1.2  Corpus, embedding model, and the 9,800-chunk claim", level=2)
    add_p(doc, "Reviewer 1 asked whether results on one technical collection and one embedding model can generalize.", italic=True)
    add_p(
        doc,
        "The saved evaluation in evaluation/results/review2_report.json uses 213 chunks and the single embedding model named in config/config.yaml, sentence-transformers/all-MiniLM-L6-v2, at 384 dimensions. The 9,800-chunk and 400-query figures appear in the manuscript. ieee_paper/README.txt states that those Section V numbers are expected values for a target study of 120 documents and 400 queries, and are not measurements from the current codebase.",
    )

    heading(doc, "1.3  Repeated runs and variance", level=2)
    add_p(doc, "Reviewer 1 asked for standard deviations or confidence intervals on Tables 2 and 3, especially for 95th-percentile latency.", italic=True)
    add_p(
        doc,
        "The repository contains one saved detection run, evaluation/results/review2_report.json, dated 5 October 2026. That file has no standard deviation and no confidence interval. It does not contain a repeated latency log for Tables 2 and 3.",
    )

    heading(doc, "1.4  Conventional RAG baseline", level=2)
    add_p(doc, "Reviewer 1 asked what conventional RAG means in the reported drop from 18.6% to 2.1% hallucinated sentences.", italic=True)
    add_p(
        doc,
        "That comparison is not in the evaluation logs. The implemented controller, src/controller/decision_controller.py, accepts, warns, or blocks from the Trust Score. A conventional condition that returns the generated answer without NLI, BERTScore, cosine verification, or tri-state control was not saved.",
    )

    heading(doc, "1.5  Multi-objective selection", level=2)
    add_p(doc, "Reviewer 1 asked for a quantitative score or Pareto analysis instead of qualitative deployment advice.", italic=True)
    add_p(
        doc,
        "The manuscripts discuss the latency, recall, and memory trade-off in prose. The repository has no Pareto script and no result file for that analysis. The non-dominated set reported later in this note comes from the local five-run measurement, not from a file already on GitHub.",
    )

    heading(doc, "1.6  End-to-end latency and the title", level=2)
    add_p(doc, "Reviewer 2 noted that generation and NLI dominate total latency, and asked for that to be stated as a core limitation.", italic=True)
    add_p(
        doc,
        "REVIEW2_RUNBOOK.md already records that the saved pipeline’s measured cost is verification, and that the manuscript’s F1 of 0.898 and AUC of 0.951 on 9,800 chunks and 400 queries have not been reproduced by this repository. The 9.76–9.81 second end-to-end table is in the manuscript, not in evaluation/results/. README.md marks Ollama generation as optional.",
    )

    heading(doc, "1.7  Memory of 384 MB, 412 MB, and 421 MB", level=2)
    add_p(doc, "Reviewer 2 asked why HNSW and hybrid memory sit at or above FlatIP if FlatIP is described as the largest index.", italic=True)
    add_p(
        doc,
        "docs/10_PANEL_QA_COMPLETE.md already states the size formula: 9,800 chunks of 384-dimensional float32 vectors are about 15 MB, not 384 MB. The saved exact index is data/index/faiss.index. The submitted 384 MB, 57 MB, 412 MB, and 421 MB values are not files in the repository. requirements.txt installs faiss-cpu, and the retrieval code contains no GPU or multi-thread comparison, which is the hardware point raised by Reviewer 2.",
    )

    heading(doc, "1.8  Retrieval agreement versus labelled recall", level=2)
    add_p(doc, "Reviewer 2 asked that Equation 7 be named retrieval agreement, because it measures overlap with FlatIP rather than labelled relevance.", italic=True)
    add_p(
        doc,
        "In the manuscript, Equation 7 is overlap with the FlatIP top-k. The retrieval number saved in evaluation/results/review2_tables.md is a different metric: document-level Recall@5 of 0.8514, MRR of 0.8108, and NDCG@5 of 0.8053, scored against the source files listed in evaluation/test_set.json.",
    )

    heading(doc, "1.9  Who curated the queries, and were the hyperparameters tuned?", level=2)
    add_p(doc, "Reviewer 3 asked who curated the 400 queries, whether a second annotator was used, and whether nlist, M, and alpha were tuned on those queries.", italic=True)
    add_p(
        doc,
        "There is no 400-query file. evaluation/build_test_set.py writes evaluation/test_set.json: 37 questions and 109 claims, labelled by the author from the local corpus. No second annotator is recorded. config/config.yaml fixes the Trust Score weights at 0.45, 0.35, and 0.20 and the thresholds at 0.75 and 0.50. There is no tuning script, and the running code has no HNSW M, IVF nlist, or hybrid alpha setting.",
    )

    heading(doc, "1.10  Verification ablation across indexes", level=2)
    add_p(doc, "Reviewer 3 asked whether the gain from NLI alone to the full verifier still holds for lower-agreement indexes such as IVF-PQ.", italic=True)
    add_p(
        doc,
        "IVF-PQ is not implemented in src/, so that comparison is not in the repository. The one saved ablation is the FlatIP evidence-window run in evaluation/results/review2_report.json. At the 0.50 threshold, F1 is 0.340 for NLI only, 0.000 for BERTScore only, 0.454 for cosine only, 0.275 for NLI plus BERTScore, and 0.314 for all three signals. AUC-ROC is 0.571, 0.797, 0.573, 0.738, and 0.701. This does not reproduce the submitted 0.856, 0.878, and 0.898 figures. ieee_paper/README.txt also says not to cite evaluation/results/_old_mock_do_not_cite/evaluation_report.json, because those perfect scores are mock values.",
    )

    heading(doc, "2. What was measured")
    add_p(doc, "The saved project index and the saved question file were used without modification. The earlier Review 2 detection report was also left unchanged.")
    bullet(doc, "Five independent runs. Approximate indexes were rebuilt at the start of each run.", bold_lead="Runs. ")
    bullet(doc, "37 questions from evaluation/test_set.json, reused unchanged in every run and every configuration. The file contains 109 hand-labelled claims (70 supported, 39 hallucinated).", bold_lead="Queries. ")
    bullet(doc, "The saved FAISS FlatIP index contains 213 vectors of dimension 384, L2-normalized. It was opened read-only.", bold_lead="Corpus. ")
    bullet(doc, "1,000 discarded searches per configuration per run, then 500 timed searches per query. Query embeddings were computed once and excluded from the timed search. Timing used a per-search clock so that a 95th percentile is defined. FAISS was restricted to one thread.", bold_lead="Warm-up and repetitions. ")
    bullet(doc, "Apple M5 Pro, 48 GB memory, macOS, Python 3.14.6, FAISS 1.15.1. NLI used cross-encoder/nli-deberta-v3-small on the Apple GPU (MPS). Ollama was not running, so generation latency was not measured.", bold_lead="Machine. ")
    bullet(doc, "The submitted IVF-PQ setting (128 lists, 8 subspaces, 8 bits) did not train: each product-quantization codebook has 256 centroids, and the index has 213 vectors. The measured IVF-PQ uses 7-bit codes. The list count, subspace count, and nprobe = 16 are unchanged. FAISS warned that 128 lists expect at least 4,992 training points.", bold_lead="IVF-PQ. ")
    bullet(doc, "In-process Okapi BM25 (k1 = 1.5, b = 0.75) fused with FlatIP at α = 0.7. Elasticsearch was not used.", bold_lead="Hybrid. ")
    bullet(doc, "evaluation/results/trustrag_opt_variability_20261013/. Each run has per-query rows and the raw latency samples.", bold_lead="Logs. ")

    heading(doc, "3. Reviewer 1")

    heading(doc, "3.1  Citation for the base TrustRAG framework", level=2)
    comment(
        doc,
        "The paper calls this an extension of a previously developed TrustRAG framework, but that base system is not cited, so it is unclear what is new.",
    )
    add_p(doc, "Camera-ready change. Name the base system as the authors’ own prior implementation, and state what this paper adds. Add a bibliographic citation only when that earlier manuscript has one. The project record does not contain a published reference for the base pipeline.")
    add_p(doc, "Replacement text for the introduction.", bold=True, space_after=4)
    add_p(
        doc,
        "TrustRAG-Opt extends the authors’ previously implemented offline TrustRAG pipeline. That pipeline embeds chunks with all-MiniLM-L6-v2, retrieves them with FAISS FlatIP, attributes each generated sentence to retrieved evidence, and combines natural-language inference, BERTScore, and cosine similarity into a Trust Score. A tri-state controller then labels each sentence Supported, Weakly Supported, or Hallucinated. The present paper keeps that verification pipeline fixed and compares four retrieval configurations in front of it: exact FlatIP, IVF-PQ, HNSW, and a dense–sparse hybrid. The new component is this controlled index comparison, including repeated latency measurements, retrieval agreement with FlatIP, and index-footprint measurement. The base pipeline itself is not claimed as a contribution of this paper.",
    )

    heading(doc, "3.2  Single corpus and a single embedding model", level=2)
    comment(
        doc,
        "Results on one knowledge base of about 9,800 Git, Docker, and Python chunks, and one embedding model, may not generalize, because IVF-PQ recall loss depends on corpus size and dimension.",
    )
    add_p(doc, "Camera-ready change. State the corpus that was actually indexed, and state the generalization limit in the experimental setup and again in the limitations, rather than only in the discussion.")
    add_p(
        doc,
        "The repeated run uses the saved index of 213 chunks from the project corpus (local technical notes, a small MS MARCO passage subset, and a small set of project PDFs), embedded with all-MiniLM-L6-v2 at 384 dimensions. One embedding model and one collection were used. IVF-PQ at the submitted 8-bit width could not be trained at this size, which is itself evidence that the quantization setting depends on the number of vectors. The latency ordering measured here, where several approximate indexes are slower than exact search, is expected for a few hundred vectors and should be presented as a small-collection result. A claim about 9,800 chunks, or about larger and more mixed collections, needs a separate experiment.",
    )

    heading(doc, "3.3  Variance for latency, recall, and memory", level=2)
    comment(
        doc,
        "Tables 2 and 3 look like single-run point estimates. Please report confidence intervals or standard deviations, especially for 95th-percentile latency.",
    )
    add_p(doc, "Camera-ready change. Replace Tables 2 and 3 with the five-run tables in Section 5. The ± value is the sample standard deviation across runs. The interval is a 95% Student-t confidence interval for the mean, with 4 degrees of freedom.")
    add_p(
        doc,
        "Retrieval agreement did not change across the five rebuilds: the top-5 identifier sets were the same in every run, so the standard deviation of agreement is 0 at the recorded precision. Latency did vary. The 95th percentile is computed inside each run on all 18,500 timed searches (37 queries × 500 repetitions). The five run-level percentiles are then summarized. Index footprint is a file size and is constant across runs; it is reported once, together with the memory procedure.",
    )

    heading(doc, "3.4  Conventional RAG baseline", level=2)
    comment(
        doc,
        "Section 5.4 reports a drop in hallucination rate from 18.6% to 2.1% against conventional RAG, but that baseline is never defined.",
    )
    add_p(doc, "Camera-ready change. Remove the 18.6% to 2.1% comparison, and the accompanying provenance and BERTScore deltas (0.741 to 0.936, and 0.868 to 0.917), until a defined baseline is run and logged.")
    add_p(
        doc,
        "A conventional RAG baseline, for this paper, would use the same chunks, the same embedding model, the same top-5 retriever, and the same generator, and would return the generated text with retrieval and without NLI, BERTScore, cosine verification, or tri-state blocking. That condition is not in the saved logs. Ollama was not running during the repeated measurement, so generation was not re-executed. The camera-ready should describe this intended baseline in the methodology and should report the comparison only after both conditions are measured on the same questions.",
    )

    heading(doc, "3.5  Quantitative multi-objective selection", level=2)
    comment(
        doc,
        "Sections 3.8 and 6 recommend indexes qualitatively. A quantitative score or a Pareto analysis would make the recommendations actionable.",
    )
    add_p(doc, "Camera-ready change. Keep Equation 6, define each objective on the measured quantities, and report the non-dominated set. The qualitative sentences that assume a 12.4× HNSW speedup should be replaced by the operating points below.")
    add_p(doc, "On this collection the three objectives in Equation 6 are operationalized as follows.", space_after=4)
    bullet(doc, "Retrieval agreement with the FlatIP top-5, as defined in Section 4.5 of this note. Higher is better.", bold_lead="Evidence quality. ")
    bullet(doc, "Mean search time after warm-up, in milliseconds. Lower is better.", bold_lead="Latency. ")
    bullet(doc, "Serialized index size in megabytes, measured with the same write path for every configuration. Lower is better.", bold_lead="Memory. ")
    add_p(
        doc,
        "A configuration is dominated when another configuration is at least as good on every objective and strictly better on one. Under that rule, FlatIP dominates IVF-PQ and HNSW at efSearch of 32, 64, and 128: those settings have equal or lower agreement, higher latency, and a larger footprint. The non-dominated set is FlatIP and HNSW with efSearch = 16. FlatIP has agreement 1.000, mean latency 0.00541 ms, and a 0.312 MB footprint. HNSW at efSearch = 16 has agreement 0.968, mean latency 0.0047 ms, and a 0.367 MB footprint. It is the latency extreme. FlatIP is the agreement and memory extreme.",
    )
    add_p(
        doc,
        "Hybrid retrieval is dominated on those three objectives because its agreement with FlatIP is 0.514, its search is slower, and its footprint is larger. A second quality objective changes that conclusion. Against the author-labelled source documents, hybrid Recall@5 is 0.986, while every dense configuration is 0.851. If the quality objective is labelled-source recall rather than agreement with FlatIP, hybrid is a non-dominated accuracy extreme, paid for with latency and memory. The camera-ready should present both objectives separately so that “highest recall” is not used for both.",
    )

    heading(doc, "4. Reviewer 2")

    heading(doc, "4.1  Controlled comparison", level=2)
    comment(
        doc,
        "Comparing four retrieval strategies under one fixed verification pipeline cleanly isolates the retrieval contribution.",
    )
    add_p(
        doc,
        "This design is retained. Generation and verification settings are held fixed in the method. In the repeated run, generation could not be executed, and NLI was timed on the claims already paired with each configuration’s retrieved chunks. The retrieval comparison itself uses the same queries, the same vectors, and the same K = 5.",
    )

    heading(doc, "4.2  End-to-end latency undercuts the title’s emphasis", level=2)
    comment(
        doc,
        "Table 3 shows total latency of 9.76–9.81 s against retrieval under 0.05 s. This should be stated as a core limitation, not only in the discussion. The title’s emphasis on efficient retrieval optimization is weakened by that result.",
    )
    add_p(doc, "Camera-ready change. State this in the abstract, the introduction, and the conclusion, in addition to the discussion. The measured split makes the same point more strongly than the submitted table, and with a different magnitude.")
    add_p(
        doc,
        "On the repeated run, mean search time is between 0.0047 ms and 0.076 ms. Mean NLI time for a question’s labelled claims is about 605–629 ms. Search is under 0.02% of that retrieval-plus-verification total. Generation was not measured. The practical limitation is that changing the index does not materially change response time while verification, and generation when it is enabled, dominate. Suggested abstract sentence: “Across five runs, index choice changes search time only within a fraction of a millisecond, while NLI verification accounts for essentially all of the measured retrieval-plus-verification latency; retrieval-index selection is therefore a quality and memory decision on this collection, rather than an end-to-end latency optimization.” The accepted title can stay. The abstract and the first page should carry this limitation so that a reader does not expect a large end-to-end speedup.",
    )

    heading(doc, "4.3  Memory: HNSW and hybrid above FlatIP", level=2)
    comment(
        doc,
        "HNSW and hybrid are reported at 412 MB and 421 MB, at or above FlatIP’s 384 MB, although FlatIP is described as the highest-memory configuration. The graph and metadata overhead should be explained.",
    )
    add_p(doc, "Camera-ready change. Replace the memory column and correct the prose. The submitted 384 / 57 / 412 / 421 MB figures are inconsistent with the float32 index-size formula, so they should not be explained as graph overhead.")
    add_p(
        doc,
        "A FlatIP index stores one float32 per dimension per vector: N × d × 4 bytes. For the saved index, 213 × 384 × 4 = 327,168 bytes (0.312 MB), which matches the serialized file. For the 9,800 chunks named in the submission, the same formula gives 9,800 × 384 × 4 = 15.05 MB, which is the figure already implied by the paper’s own complexity discussion. An HNSW graph adds a bounded number of neighbor links per vector; at M = 32 that overhead is on the order of a few megabytes for 9,800 vectors, which can make HNSW larger than FlatIP, but it does not produce a 384 MB or 412 MB index. The value 384 in the submitted table coincides with the embedding dimension. The repeated run therefore reports serialized bytes, not that column.",
    )
    add_p(doc, "Measured footprints, same write procedure for every configuration:", space_after=4)
    bullet(doc, "0.312 MB for the vectors. A fresh Python process that loads only this file uses about 68.3 MB in total; 0.25 MB of that appears when the index is loaded. The 68 MB is the interpreter and the FAISS library, and it is almost the same for every configuration, so it is reported separately from the index footprint.", bold_lead="FlatIP. ")
    bullet(doc, "0.379 MB. Incremental process memory after load is 0.44 MB. On this collection the 7-bit index is larger than FlatIP, because the coarse quantizer and codebooks are not amortized over enough vectors.", bold_lead="IVF-PQ. ")
    bullet(doc, "0.367 MB for one graph shared by every efSearch value. efSearch is a query-time parameter and does not change the file size. The graph makes this index slightly larger than FlatIP, which is the overhead the reviewer asked to have explained, at a scale of tens of kilobytes here.", bold_lead="HNSW. ")
    bullet(doc, "0.868 MB, equal to the FlatIP file plus the in-process BM25 postings. The extra memory is the sparse index, which is why this configuration sits above FlatIP.", bold_lead="Hybrid. ")
    add_p(doc, "Runtime memory procedure. For each configuration, a new Python process imports FAISS, records resident set size with ps, reads the index file, records resident set size again, runs 100 searches, and records it a third time. Index footprint and process resident set size are both kept, because they answer different questions.")

    heading(doc, "4.4  Single-threaded CPU versus production hardware", level=2)
    comment(
        doc,
        "The single-CPU, no-GPU, single-thread setup is clear. Please discuss how the latency trade-offs might change with GPU FAISS or multi-threading, which are common in production.",
    )
    add_p(
        doc,
        "The repeated search measurements were single-threaded, matching the submitted protocol, so that index algorithms are compared under the same thread count. NLI ran on the Apple GPU. No GPU FAISS build and no multi-thread sweep were run, so no speedup factor for those deployments is stated.",
    )
    add_p(
        doc,
        "What can be said from this run is structural. At 213 vectors the exact scan is already a few microseconds, and HNSW with a wider beam is slower than FlatIP because graph traversal adds overhead that a tiny scan does not. Multi-threading or a GPU would shorten the exact scan further and would not create the 12× gap reported in the submission. The relative gap between indexes can grow once N is large enough that an O(Nd) scan exceeds graph-search overhead. That crossover was not measured. On end-to-end time, a faster index would still leave NLI, measured here at about 0.6 s per question, as the larger term. GPU acceleration of the cross-encoder would change end-to-end latency more than GPU acceleration of retrieval.",
    )

    heading(doc, "4.5  Equation 7 is agreement with FlatIP, not labelled recall", level=2)
    comment(
        doc,
        "Equation 7 measures overlap with FlatIP’s retrieved set rather than human-labelled relevance. Please call it retrieval agreement, so it is not confused with standard recall.",
    )
    add_p(doc, "Camera-ready change. Rename the Table 2 metric and Equation 7. Keep labelled-source recall as a separate column.")
    add_p(doc, "Replacement definition.", bold=True, space_after=4)
    add_p(
        doc,
        "RetrievalAgreement@K(q) = |R_config(q) ∩ R_FlatIP(q)| / K. This is agreement with exact inner-product search. It is 1 for FlatIP by construction. It is not Recall@K against a human relevance judgment. Labelled-source Recall@5 is computed separately as the fraction of the question’s author-marked source documents that appear among the top-5 retrieved chunks.",
    )
    add_p(
        doc,
        "This distinction changes the hybrid result. Hybrid agreement with FlatIP is 0.514, so the submitted statement that hybrid has the highest Recall@5 against the exact reference cannot be kept: agreement with the exact set cannot exceed FlatIP’s value of 1. Hybrid’s labelled-source Recall@5 is 0.986, which is higher than the dense configurations’ 0.851. Hybrid changes the ranking by adding lexical scores; it does not reproduce FlatIP, and on the labelled sources it retrieves the marked document more often.",
    )

    heading(doc, "5. Reviewer 3")

    heading(doc, "5.1  The experimental question", level=2)
    comment(
        doc,
        "The paper identifies a real gap: index choice can affect evidence attribution and hallucination verification, and a fixed verification pipeline is the right way to isolate that question.",
    )
    add_p(
        doc,
        "The question is unchanged. The camera-ready should answer it with the measured agreement and labelled-source recall, and should say that a full verification-quality comparison across indexes, with generation enabled, remains to be run. NLI latency was measured for every index on that index’s retrieved chunks. Detection F1 was not recomputed for every index.",
    )

    heading(doc, "5.2  Who curated the 400 queries, and were hyperparameters tuned on them?", level=2)
    comment(
        doc,
        "The 400-query set is only described as manually curated. Please state who curated it, whether a second annotator was used, and whether the queries were held out from tuning of nlist, M, and α. The hyperparameters look fixed; please say so.",
    )
    add_p(doc, "Camera-ready change. Replace the 400-query description with the query file that exists, and state the tuning status explicitly.")
    bullet(doc, "There is no 400-query file, and no 250/150 answerable-versus-adversarial split, in the repository. A reuse of 400 queries across runs cannot be confirmed.", bold_lead="Query count. ")
    bullet(doc, "The 37 questions and 109 claims were written by the author against the local corpus and stored in evaluation/build_test_set.py. A claim is labelled supported when the corpus states it or directly entails it, and hallucinated when it contradicts the corpus or adds a detail the corpus does not state. No second annotator and no inter-annotator statistic are recorded.", bold_lead="Curation. ")
    bullet(doc, "nlist = 128, nprobe = 16, m = 8, M = 32, efConstruction = 200, efSearch ∈ {16, 32, 64, 128}, and α = 0.7 were fixed before the repeated run from the submitted configuration. They were not selected by a search on these 37 questions, and there was no held-out tuning split. The only change forced by the data was IVF-PQ code width, from 8 bits to 7 bits, because 8-bit training is impossible at N = 213.", bold_lead="Hyperparameters. ")

    heading(doc, "5.3  Verification ablation only on FlatIP", level=2)
    comment(
        doc,
        "The ablation NLI alone F1 0.856, NLI+BERTScore 0.878, and full model 0.898 is reported only for FlatIP. Please check that the combination still helps under lower-agreement retrieval such as IVF-PQ.",
    )
    add_p(doc, "Camera-ready change. Remove the 0.856 / 0.878 / 0.898 ablation from the camera-ready. It was not reproduced, and it was not repeated on IVF-PQ or the other indexes.")
    add_p(
        doc,
        "The saved detection run (evaluation/results/review2_report.json, 5 October 2026) scores the same 109 claims once, with the shipped evidence-window verifier, on the FlatIP pipeline. Positive class is the hallucinated claim. At the fixed 0.50 support threshold the F1 scores are: NLI only 0.340, BERTScore only 0.000, cosine only 0.454, NLI+BERTScore 0.275, and all three signals 0.314. AUC-ROC is 0.571, 0.797, 0.573, 0.738, and 0.701 respectively. That single run does not show a gain from 0.856 to 0.898 as signals are added, and it provides no estimate for IVF-PQ. Confirming the reviewer’s request requires a new labelled run of the three signal combinations on the chunks returned by each index.",
    )

    heading(doc, "6. Tables to use in the camera-ready")
    add_p(doc, "Table A replaces submitted Table 2. Agreement and labelled-source recall are separate. Latency is in milliseconds. The ± value is the standard deviation across five runs.")
    add_table(
        doc,
        ["Configuration", "Agreement@5", "Labelled Recall@5", "Mean latency (ms)", "95th pct. (ms)", "Footprint (MB)"],
        [
            ["FlatIP", "1.000 ± 0.000", "0.851 ± 0.000", "0.00541 ± 0.00002", "0.0062 ± 0.0002", "0.312"],
            ["IVF-PQ (7-bit)", "0.892 ± 0.000", "0.851 ± 0.000", "0.0085 ± 0.0003", "0.0106 ± 0.0014", "0.379"],
            ["HNSW ef=16", "0.968 ± 0.000", "0.851 ± 0.000", "0.0047 ± 0.0002", "0.0057 ± 0.0004", "0.367"],
            ["HNSW ef=32", "1.000 ± 0.000", "0.851 ± 0.000", "0.0060 ± 0.0001", "0.0071 ± 0.0003", "0.367"],
            ["HNSW ef=64", "1.000 ± 0.000", "0.851 ± 0.000", "0.0085 ± 0.0002", "0.0101 ± 0.0005", "0.367"],
            ["HNSW ef=128", "1.000 ± 0.000", "0.851 ± 0.000", "0.0158 ± 0.0004", "0.0192 ± 0.0009", "0.367"],
            ["Hybrid FlatIP+BM25", "0.514 ± 0.000", "0.986 ± 0.000", "0.0764 ± 0.0010", "0.1225 ± 0.0014", "0.868"],
        ],
    )
    add_p(doc, "95% confidence intervals for mean search latency (ms): FlatIP 0.00539–0.00543; IVF-PQ 0.0081–0.0089; HNSW ef=16 0.0045–0.0050; ef=32 0.0059–0.0061; ef=64 0.0083–0.0087; ef=128 0.0153–0.0164; hybrid 0.0752–0.0777.", size=10, italic=True)
    add_p(doc, "Speedup against FlatIP on this collection is 1.15× for HNSW ef=16. Every other configuration is slower than FlatIP. The submitted 22.5×, 12.4×, and 6.7× speedups, and the 47.3 ms FlatIP latency, do not describe this index.")

    add_p(doc, "Table B replaces submitted Table 3. End-to-end here is mean search time plus NLI on the question’s labelled claims. Generation is omitted because it was not run. Search and NLI were timed in separate processes: loading FAISS and PyTorch together aborts with an OpenMP runtime error, so the total is their sum.", space_before=6)
    add_table(
        doc,
        ["Configuration", "Search (ms)", "NLI (ms)", "Search + NLI (ms)", "95% CI of the mean"],
        [
            ["FlatIP", "0.0054", "605 ± 22", "605 ± 22", "578–633"],
            ["IVF-PQ (7-bit)", "0.0085", "610 ± 47", "610 ± 47", "552–668"],
            ["HNSW ef=16", "0.0047", "610 ± 43", "610 ± 43", "556–663"],
            ["HNSW ef=32", "0.0060", "613 ± 38", "613 ± 38", "566–660"],
            ["HNSW ef=64", "0.0085", "620 ± 41", "620 ± 41", "568–671"],
            ["HNSW ef=128", "0.0158", "623 ± 38", "623 ± 38", "576–671"],
            ["Hybrid", "0.076", "629 ± 36", "630 ± 36", "585–674"],
        ],
    )
    add_p(doc, "The end-to-end 95th percentile, taken across the 37 questions inside each run and then across runs, is 725 ± 47 ms for FlatIP and between 715 and 760 ms for the other configurations. With only 37 questions, that percentile is coarser than the retrieval percentile, which uses 18,500 searches.", size=10, italic=True)

    heading(doc, "7. Replacement paragraphs")
    add_p(doc, "Abstract, results sentence.", bold=True, space_after=4)
    add_p(
        doc,
        "On the saved 213-chunk index and 37 evaluation questions, five runs show that HNSW with efSearch = 16 is the only configuration faster than exact FlatIP (0.0047 ms versus 0.00541 ms) and that its agreement with FlatIP is 0.968. Wider HNSW beams match FlatIP’s top-5 exactly and are slower. IVF-PQ at the widest trainable code width agrees with FlatIP on 0.892 of the top-5 and does not reduce the index footprint at this scale. Hybrid retrieval agrees with FlatIP on 0.514 of the top-5 and raises labelled-source Recall@5 from 0.851 to 0.986. NLI verification, at about 0.61 s per question, dominates search time by more than three orders of magnitude.",
    )
    add_p(doc, "Results, operating points.", bold=True, space_after=4)
    add_p(
        doc,
        "For this collection, exact FlatIP is the audit configuration: agreement 1, the smallest footprint, and search time statistically indistinguishable from a few microseconds. HNSW with efSearch = 16 is the low-latency configuration, with a small agreement drop. HNSW at efSearch of 32 or above reproduces FlatIP’s top-5 and costs additional search time, so it is not selected once exact search is already this cheap. IVF-PQ is not the memory-minimizing choice at N = 213. Hybrid retrieval is the configuration to select when labelled-source coverage matters more than reproducing the exact dense ranking.",
    )
    add_p(doc, "Limitations, paragraph to place in Section 6.", bold=True, space_after=4)
    add_p(
        doc,
        "The evidence is a single 213-chunk collection and a single 384-dimensional embedding model, with author-written questions and no second annotator. Hyperparameters were fixed in advance and were not tuned. IVF-PQ could not be trained at 8 bits. Generation was not measured. Detection F1 was not recomputed under each index. Resident memory of the Python process is dominated by the runtime library, so index comparisons use serialized footprint plus the incremental resident set after load. Results at a few hundred vectors do not establish the latency or memory ordering of a multi-thousand-chunk or multi-million-chunk index, nor the effect of multi-threaded or GPU search. Those settings remain future measurements.",
    )

    heading(doc, "8. Submitted statements to withdraw")
    add_p(doc, "The following submitted statements should leave the camera-ready unless a new logged experiment produces them. They were not produced by the 13 October 2026 run, and the saved project files do not contain the corpus or query set they presuppose.")
    rows = [
        ("About 9,800 chunks from Git, Docker, and Python manuals", "Saved index has 213 vectors."),
        ("400 queries, 250 answerable and 150 hallucination-inducing", "Saved test file has 37 questions and 109 claims."),
        ("FlatIP latency 47.3 ms, 95th percentile 52.1 ms, memory 384 MB", "Measured 0.00541 ms, 95th percentile 0.0062 ms, footprint 0.312 MB."),
        ("HNSW ef=64 at 3.8 ms, Recall@5 0.93, 12.4× speedup", "Measured 0.0085 ms, agreement 1.000, slower than FlatIP."),
        ("IVF-PQ at 7.1 ms, Recall@5 0.89, 57 MB, 6.7× compression", "8-bit IVF-PQ did not train. 7-bit IVF-PQ is 0.0085 ms, agreement 0.892, footprint 0.379 MB."),
        ("Hybrid Recall@5 0.96 at 8.2 ms and 421 MB", "Agreement with FlatIP is 0.514. Labelled-source Recall@5 is 0.986. Latency 0.076 ms. Footprint 0.868 MB."),
        ("End-to-end 9.81 s = 0.047 + 5.20 generation + 4.56 NLI", "Search + NLI is about 0.61 s. Generation was not measured."),
        ("Detection F1 0.898, AUC 0.951, and the 0.856 to 0.898 ablation", "Not reproduced. The saved 109-claim run has full-model F1 0.314 and AUC 0.701."),
        ("Hallucination rate reduced from 18.6% to 2.1% versus conventional RAG", "The conventional condition is undefined and was not run."),
        ("Hardware: Intel i7-11700, 16 GB, Ubuntu, FAISS 1.7.4, Python 3.9", "This run: Apple M5 Pro, 48 GB, macOS, FAISS 1.15.1, Python 3.14."),
    ]
    add_table(doc, ["Submitted statement", "What the available experiment shows"], rows)

    heading(doc, "9. Short reply the authors can send with the revision")
    add_p(
        doc,
        "We thank the reviewers. The camera-ready cites the base system as our prior offline TrustRAG pipeline and states that this paper’s contribution is the controlled index comparison. Equation 7 is renamed Retrieval Agreement@K and is reported separately from labelled-source recall. Tables 2 and 3 are replaced by five-run measurements on the saved 213-chunk index and the saved 37 questions, with standard deviations and 95% confidence intervals. The 95th percentile is summarized across runs. Index footprint is the serialized file size; process memory is reported separately and is dominated by the runtime. The non-dominated set on agreement, latency, and footprint is FlatIP and HNSW with efSearch = 16. Hybrid retrieval is reported as a labelled-source recall gain, not as higher agreement with exact search. The conventional-RAG hallucination comparison and the 0.898 detection ablation are removed because they are not supported by the logged runs. We state explicitly that the queries were author-written, that no second annotator was used, and that the index hyperparameters were fixed in advance. We also state, in the abstract and the conclusion, that NLI dominates end-to-end time and that these small-collection orderings are not claimed to generalize to larger corpora or to GPU and multi-thread deployment.",
    )

    add_p(doc, "Logs", bold=True, space_before=8, space_after=4)
    add_p(doc, "Repeated retrieval and NLI run: evaluation/results/trustrag_opt_variability_20261013/ (summary.md, summary.json, run_01 through run_05, per-query JSONL, and latency samples).")
    add_p(doc, "Saved detection run, not overwritten: evaluation/results/review2_report.json.")

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
