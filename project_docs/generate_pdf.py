import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.pdfgen import canvas

# Define custom Canvas for running header and footer with dynamic total page count
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # We suppress running headers and footers on page 1 (Cover Page)
        if self._pageNumber > 1:
            # Header Text
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#1B365D"))
            self.drawString(54, 750, "M.TECH MAIN PROJECT ROADMAP")
            
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#4A5568"))
            self.drawRightString(558, 750, "Offline RAG Hallucination Detection Framework")
            
            # Header Line
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.75)
            self.line(54, 742, 558, 742)
            
            # Footer Line
            self.line(54, 55, 558, 55)
            
            # Footer Text
            self.drawString(54, 42, "Department of Computer Science & Engineering - Confidential")
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(558, 42, page_text)
            
        else:
            # Subtle decorative element on Cover Page footer
            self.setStrokeColor(colors.HexColor("#1B365D"))
            self.setLineWidth(2)
            self.line(54, 55, 558, 55)
            
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#718096"))
            self.drawString(54, 42, "Design & Development Phase Technical Documentation")
            self.drawRightString(558, 42, "June 2026")
            
        self.restoreState()

def create_divider(color=colors.HexColor('#E2E8F0'), thickness=1, space_after=12):
    t = Table([['']], colWidths=[504], rowHeights=[thickness])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), color),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    return KeepTogether([t, Spacer(1, space_after)])

def make_callout(text, styles, title="IMPORTANT", mode="info"):
    content = f"<b>{title}:</b> {text}"
    p = Paragraph(content, styles['ProjectBody'])
    
    if mode == "info":
        bg = colors.HexColor('#F0FDF4')
        border_col = colors.HexColor('#16A34A')
        line_col = colors.HexColor('#DCFCE7')
    elif mode == "warning":
        bg = colors.HexColor('#FFFBEB')
        border_col = colors.HexColor('#D97706')
        line_col = colors.HexColor('#FEF3C7')
    else:
        bg = colors.HexColor('#F8FAFC')
        border_col = colors.HexColor('#475569')
        line_col = colors.HexColor('#E2E8F0')
        
    t = Table([[p]], colWidths=[504])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg),
        ('BOX', (0, 0), (-1, -1), 0.5, line_col),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LINELEFT', (0, 0), (0, -1), 3.0, border_col),
    ]))
    return KeepTogether([t, Spacer(1, 10)])

def make_table(data, col_widths, styles):
    table_data = []
    for row_idx, row in enumerate(data):
        row_data = []
        for col_idx, text in enumerate(row):
            if row_idx == 0:
                p = Paragraph(text, styles['ProjectTableHeader'])
            else:
                p = Paragraph(text, styles['ProjectTableCell'])
            row_data.append(p)
        table_data.append(row_data)
    
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
    ]))
    return t

def build_pdf(filename="Project_Plan.pdf"):
    # Page setup: letter is 612x792. Margins 54pt (0.75in) left/right, 72pt (1.0in) top/bottom.
    # Printable width: 504pt. Printable height: 648pt (from Y=72 to Y=720).
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette Styling
    primary_color = colors.HexColor("#1B365D")   # Deep Navy
    secondary_color = colors.HexColor("#2E5B82") # Slate Blue
    accent_color = colors.HexColor("#00A896")    # Teal
    text_color = colors.HexColor("#2D3748")      # Dark Grey/Charcoal
    
    # Text styles
    styles.add(ParagraphStyle(
        name='ProjectCoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        spaceAfter=15
    ))
    
    styles.add(ParagraphStyle(
        name='ProjectCoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceAfter=30
    ))
    
    styles.add(ParagraphStyle(
        name='ProjectSectionHeading',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=10,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='ProjectSubsectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='ProjectSubSubheading',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='ProjectBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=text_color,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name='ProjectBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=text_color,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    ))

    styles.add(ParagraphStyle(
        name='ProjectTableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    ))

    styles.add(ParagraphStyle(
        name='ProjectTableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=text_color
    ))

    styles.add(ParagraphStyle(
        name='ProjectCodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1A202C"),
        backColor=colors.HexColor("#F7FAFC"),
        borderPadding=6,
        spaceAfter=6
    ))

    story = []

    # ================= PAGE 1: COVER & EXECUTIVE SUMMARY =================
    story.append(Spacer(1, 40))
    story.append(Paragraph("PROJECT IMPLEMENTATION ROADMAP", styles['ProjectCoverSubtitle']))
    story.append(Paragraph("Design and Development of an Explainability-Driven Hallucination Detection Framework for Offline Retrieval-Augmented Generation", styles['ProjectCoverTitle']))
    
    story.append(create_divider(accent_color, thickness=3, space_after=20))
    
    # Metadata Block
    meta_data = [
        ["Course / Project:", "S2 M.Tech Main Project"],
        ["Domain Area:", "Natural Language Processing (NLP) & Explainable AI (XAI)"],
        ["Implementation:", "Fully Offline Deployment (No API Dependencies)"],
        ["Key Focus:", "Intrinsic Hallucination Detection & Citation-Grounding Logic"],
    ]
    t_meta = Table(meta_data, colWidths=[130, 374])
    t_meta.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('TEXTCOLOR', (0,0), (0,-1), primary_color),
        ('TEXTCOLOR', (1,0), (1,-1), text_color),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 30))
    
    story.append(Paragraph("Executive Summary", styles['ProjectSectionHeading']))
    story.append(Paragraph("Large Language Models (LLMs) deployed within Retrieval-Augmented Generation (RAG) pipelines are highly susceptible to <i>hallucinations</i>—generating claims that are unsupported or contradicted by retrieved context. In offline, high-security, or resource-constrained environments, standard cloud-based fact-checking models and commercial APIs are completely unusable. This project establishes an <b>entirely offline, explainability-driven framework</b> that leverages internal retrieval signals (similarity heatmaps, document-chunk attribution, provenance, and confidence metrics) as primary features for real-time hallucination detection and response filtering.", styles['ProjectBody']))
    story.append(Spacer(1, 10))
    
    summary_bullet_1 = "<b>Intrinsically Grounded:</b> Leverages internal metrics generated directly during retrieval and inference rather than relying on heavy post-hoc external validation models."
    summary_bullet_2 = "<b>Robust Fallbacks:</b> Employs a decision controller to either output verified claims with citations, return a partial response warning the user of weak grounding, or block the output entirely to prevent misinformation."
    summary_bullet_3 = "<b>Privacy & Speed:</b> Designed to run locally on consumer-grade workstations using optimized local models (sentence-transformers, quantized LLMs, and lightweight NLI cross-encoders)."
    
    story.append(Paragraph(f"&bull; {summary_bullet_1}", styles['ProjectBullet']))
    story.append(Paragraph(f"&bull; {summary_bullet_2}", styles['ProjectBullet']))
    story.append(Paragraph(f"&bull; {summary_bullet_3}", styles['ProjectBullet']))
    
    story.append(PageBreak())

    # ================= PAGE 2: PROBLEM STATEMENT & CORE NOVELTY =================
    story.append(Paragraph("Problem Formulation & Core Objectives", styles['ProjectSectionHeading']))
    story.append(create_divider())
    
    story.append(Paragraph("The Problem Statement", styles['ProjectSubsectionHeading']))
    problem_text = "Despite RAG providing grounding context, generative LLMs frequently output claims that are weakly supported or contradicted by retrieved evidence. Traditional hallucination mitigation relies on heavy, black-box external services or cloud APIs (e.g. OpenAI GPT-4 API) for validation. These methods introduce high latency, compromise data privacy, and are completely infeasible for offline installations. Meanwhile, internal explainability signals—such as cosine similarity scores, chunk attribution vectors, and base token probabilities—remain vastly underutilized as features for real-time hallucination control."
    story.append(Paragraph(problem_text, styles['ProjectBody']))
    
    story.append(make_callout("The core research question of this M.Tech project is: <i>How can internal explainability signals generated within an offline RAG framework be systematically combined with lightweight local NLP models to detect, isolate, and suppress hallucinated outputs, while ensuring transparency and evidence grounding?</i>", styles, "CORE RESEARCH QUESTION", "warning"))
    
    story.append(Paragraph("Project Objectives", styles['ProjectSubsectionHeading']))
    story.append(Paragraph("&bull; <b>Design Local Ingestion & Retrieval Core:</b> Build a local knowledge base using document parsers, token-level chunking, and FAISS indexing for fast vector lookup.", styles['ProjectBullet']))
    story.append(Paragraph("&bull; <b>Formulate an Explainability Layer:</b> Capture cosine similarity, chunk-source mappings, and model-level confidence outputs to trace each generated statement to its exact source chunk.", styles['ProjectBullet']))
    story.append(Paragraph("&bull; <b>Develop Multi-Technique Hallucination Detection:</b> Combine lightweight local Natural Language Inference (NLI) entailment, token-level BERTScore, and vector similarity thresholds to detect logical and semantic hallucination types.", styles['ProjectBullet']))
    story.append(Paragraph("&bull; <b>Implement a Tri-State Decision Controller:</b> Execute rule-based response filtering to output answers (with precise inline citations), deliver partial/warned answers, or completely block untrusted responses.", styles['ProjectBullet']))
    story.append(Paragraph("&bull; <b>Deploy an Interactive Web Demo:</b> Construct a local Streamlit interface that visualizes retrieved evidence, citation linkages, NLI classification outputs, and attention maps.", styles['ProjectBullet']))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("Core Novelty: Intrinsic Explainability Co-integration", styles['ProjectSubsectionHeading']))
    novelty_text = "Most RAG guardrails perform <i>post-hoc checking</i>—treating the system as a black box and passing the final text to a secondary LLM for verification. This is computationally expensive and slow. <b>This project's novelty lies in co-integrating explainability and detection:</b> using the scores calculated during document retrieval and generation as primary inputs to the detector, creating an intrinsic, lightweight feedback loop that operates natively within the pipeline."
    story.append(Paragraph(novelty_text, styles['ProjectBody']))
    
    story.append(PageBreak())

    # ================= PAGE 3: SYSTEM ARCHITECTURE & WORKFLOW =================
    story.append(Paragraph("Proposed System Architecture", styles['ProjectSectionHeading']))
    story.append(create_divider())
    
    arch_intro = "The system is structured as an end-to-end local pipeline where explainability signals and hallucination detection layers are deeply interwoven. Rather than executing as isolated blocks, the retrieval, generation, and validation phases exchange metrics to control the final output."
    story.append(Paragraph(arch_intro, styles['ProjectBody']))
    story.append(Spacer(1, 10))
    
    # Textual Flow Description
    story.append(Paragraph("Pipeline Flow Sequence", styles['ProjectSubsectionHeading']))
    flow_steps = [
        "<b>1. Query Ingestion & Embedding:</b> The user query is tokenized and converted to a dense vector using the local <i>all-MiniLM-L6-v2</i> embedding model.",
        "<b>2. Context Retrieval:</b> FAISS performs a fast cosine similarity search against the document store and retrieves the top-<i>K</i> chunks. Similarity scores are saved.",
        "<b>3. Local LLM Generation:</b> The query and retrieved context are formatted into a system prompt and passed to the local LLM (e.g. <i>Mistral-7B-Instruct</i> via Ollama). The model generates the raw response.",
        "<b>4. Claim Segmentation:</b> The generated response is split into individual sentences (claims) for granular inspection.",
        "<b>5. Triple-Verification Detection:</b> Each claim is verified against retrieved chunks using: (a) NLI Entailment scoring (Cross-Encoder), (b) Token-level semantic overlap (BERTScore), and (c) Vector similarity.",
        "<b>6. Metric Aggregation:</b> The controller calculates a <i>Provenance/Support Score</i> for each claim, incorporating retrieval cosine scores and NLI probabilities.",
        "<b>7. Tri-State Control Action:</b> The controller selects the final output pathway: <i>Answer with Citations</i>, <i>Partial Answer with Warned Claims</i>, or <i>Block & Clarify</i>."
    ]
    for step in flow_steps:
        story.append(Paragraph(f"&bull; {step}", styles['ProjectBullet']))
        
    story.append(Spacer(1, 15))
    story.append(Paragraph("System Architecture Visual Flow Map", styles['ProjectSubsectionHeading']))
    
    # ASCII Architecture Map inside a Code Block for visual appeal
    ascii_map = """
 [ User Query ] ------> [ all-MiniLM-L6-v2 Embedder ] -------> [ FAISS Vector DB ]
                                                                     |
                                                           (Top-K Chunks + Sim Scores)
                                                                     |
 [ Output Response ] <--- [ Decision Controller ] <--- [ Explainability Layer ]
                               ^                             |
                         (Tri-State)            (Attribution, Sim, NLI, BERTScore)
                               |                             v
                     [ Block / Partial / Cite ] <---- [ Hallucination Detector ]
    """
    story.append(Paragraph(ascii_map, styles['ProjectCodeBlock']))
    
    story.append(PageBreak())

    # ================= PAGE 4: TECHNOLOGY STACK SELECTION =================
    story.append(Paragraph("Detailed Technology Stack Selection", styles['ProjectSectionHeading']))
    story.append(create_divider())
    
    tech_intro = "All selected technologies are open-source, lightweight, and capable of running fully offline on consumer-grade desktop hardware. This ensures data privacy and zero operating costs."
    story.append(Paragraph(tech_intro, styles['ProjectBody']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Layer-by-Layer Components", styles['ProjectSubsectionHeading']))
    
    tech_data = [
        ["Pipeline Layer", "Selected Tool / Model", "Technical Justification"],
        ["Document Embeddings", "all-MiniLM-L6-v2", "Extremely compact (80MB), fast CPU inference, 384-dimensional dense vectors optimized for semantic text similarity."],
        ["Vector database", "FAISS (Facebook AI Search)", "Runs entirely in-memory, serverless, compiled C++ backend, enabling sub-millisecond retrieval without network overhead."],
        ["Local LLM Generator", "Mistral-7B-Instruct / LLaMA-3-8B (Q4_K_M GGUF)", "High-performance open-source models, quantized to 4-bit to run efficiently within 8GB VRAM using Ollama/llama.cpp."],
        ["NLI Entailment Classifier", "cross-encoder/nli-deberta-v3-large", "Fine-tuned DeBERTa model that excels at sentence-level implication (Entails/Neutral/Contradicts) with higher precision than raw LLMs."],
        ["Semantic Overlap", "BERTScore (Local)", "Token-level similarity using contextual embeddings; avoids lexical match limits without calling cloud services."],
        ["Orchestration & UI", "Python, LangChain, Streamlit", "Streamlit provides rapid dashboard prototyping, while LangChain manages local model inputs/outputs and chunk parsing."]
    ]
    story.append(make_table(tech_data, [100, 140, 264], styles))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Alternative Technologies Evaluated & Rejected", styles['ProjectSubsectionHeading']))
    
    rejections = [
        ["Component", "Rejected Alternative", "Reason for Rejection"],
        ["Embeddings", "OpenAI text-embedding-ada-002", "Requires cloud connection, violates 'offline' mandate, incurs recurring API usage costs."],
        ["Vector DB", "Pinecone / Milvus Server", "Pinecone is cloud-only. Milvus requires dedicated docker services, causing high overhead for desktop deployment."],
        ["Generator LLM", "GPT-4 / ChatGPT API", "Black-box nature blocks access to internal model probabilities, attention weights, and requires active internet."],
        ["Detector", "Heavyweight LLM Prompting", "Running a 70B parameter model just to evaluate the output of a 7B model is computationally redundant and introduces extreme latency."]
    ]
    story.append(make_table(rejections, [90, 140, 274], styles))
    
    story.append(PageBreak())

    # ================= PAGE 5: DATASET & EVALUATION STRATEGY =================
    story.append(Paragraph("Dataset and Evaluation Strategy", styles['ProjectSectionHeading']))
    story.append(create_divider())
    
    dataset_intro = "To rigorously validate the hallucination detection framework, the evaluation methodology employs three distinct data streams: a primary domain-specific knowledge corpus, established public evaluation benchmarks, and a custom-labeled gold standard dataset."
    story.append(Paragraph(dataset_intro, styles['ProjectBody']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("1. Primary Domain Knowledge Corpus", styles['ProjectSubsectionHeading']))
    story.append(Paragraph("The system will be tested on a specialized technical domain to demonstrate adaptation. The chosen domain is <b>Technical Documentation / Software Engineering Manuals</b> (specifically, Git Version Control Documentation, Docker reference manuals, and Python developer guides). This domain provides structured, fact-dense text where hallucinated arguments, command flags, or syntax would be instantly fatal.", styles['ProjectBody']))
    
    story.append(Paragraph("2. Hallucination & Truthfulness Benchmarks", styles['ProjectSubsectionHeading']))
    story.append(Paragraph("To evaluate the detector's classification accuracy independent of the specific domain corpus, the system will run against samples from the following public datasets:", styles['ProjectBody']))
    story.append(Paragraph("&bull; <b>RAGTruth:</b> A fine-grained hallucination corpus consisting of 17,790 LLM responses across QA, summarization, and data-to-text tasks, complete with word-level and sentence-level hallucination labels.", styles['ProjectBullet']))
    story.append(Paragraph("&bull; <b>HaluEval:</b> A large benchmark containing over 30,000 generated responses with paired factual and hallucinated texts, specifically designed for testing detection models.", styles['ProjectBullet']))
    
    story.append(Paragraph("3. Custom Golden Evaluation Dataset Setup", styles['ProjectSubsectionHeading']))
    story.append(Paragraph("To evaluate the end-to-end system under typical M.Tech project conditions, a custom test set of <b>150 QA pairs</b> will be curated and labeled:", styles['ProjectBody']))
    story.append(Paragraph("&bull; <b>Dataset split:</b> 50 questions with complete evidence support in the corpus, 50 questions with partial evidence support, and 50 questions designed to trigger out-of-corpus hallucinations (queries about topics not contained in the documents).", styles['ProjectBullet']))
    story.append(Paragraph("&bull; <b>Ground Truth Annotation:</b> Each generated claim will be manually annotated as: (1) <i>Fully Supported</i>, (2) <i>Partially Supported</i>, or (3) <i>Hallucinated / Unsupported</i>.", styles['ProjectBullet']))
    
    story.append(make_callout("This custom evaluation set will serve as the validation ground for establishing the optimal threshold values for the NLI, BERTScore, and Cosine similarity metrics in the decision controller.", styles, "METRIC THRESHOLD TUNING", "info"))
    
    story.append(Spacer(1, 5))
    story.append(Paragraph("Evaluation Framework Overview Table", styles['ProjectSubsectionHeading']))
    eval_framework_data = [
        ["Dataset Type", "Source / Model", "Purpose", "Target Metric"],
        ["Knowledge Corpus", "Local Tech Docs (PDFs/MDs)", "Supplies context chunks to LLM", "Retrieval Recall@K, MRR"],
        ["Public Benchmark", "RAGTruth / HaluEval", "Tests detector's generalizability", "AUC-ROC, F1-Score"],
        ["Custom Golden Set", "150 Hand-curated QA Pairs", "Tunes decision thresholds", "Precision / Recall trade-off"]
    ]
    story.append(make_table(eval_framework_data, [100, 140, 164, 100], styles))
    
    story.append(PageBreak())

    # ================= PAGE 6: STEP-BY-STEP ROADMAP =================
    story.append(Paragraph("Project Roadmap & Implementation Phases", styles['ProjectSectionHeading']))
    story.append(create_divider())
    
    roadmap_intro = "The implementation plan spans 16 weeks, covering ingestion core creation, explainability tracking, metric formulation, threshold optimization, web demo development, and thesis compilation."
    story.append(Paragraph(roadmap_intro, styles['ProjectBody']))
    story.append(Spacer(1, 10))
    
    # Timeline Table
    timeline_data = [
        ["Phase & Weeks", "Core Tasks & Implementation Details", "Key Deliverables / Milestone"],
        ["Phase 1\n(W1 - W2)", "Set up local virtual environment. Implement PDF/Markdown text extraction using python-docx and PyPDF2. Write token-based sliding-window chunker.", "Data ingestion script that processes folders and outputs text chunks."],
        ["Phase 2\n(W3 - W4)", "Integrate all-MiniLM-L6-v2. Create FAISS index script. Optimize retrieval parameters (chunk size, overlap, K retrieved chunks).", "Retrieval module yielding top-K chunks with similarity scores."],
        ["Phase 3\n(W5 - W6)", "Configure local Mistral 7B LLM via Ollama. Build generator prompt wrapper. Write chunk-to-sentence attribution tracer.", "Local generator script returning raw answers mapped to source chunks."],
        ["Phase 4\n(W7 - W9)", "Implement NLI classifier (cross-encoder). Implement local BERTScore. Formulate cosine threshold filters.", "Detection engine returning NLI, BERTScore, and cosine stats per sentence."],
        ["Phase 5\n(W10 - W11)", "Design aggregation logic for the Provenance Score. Implement Tri-State controller (citations, warnings, blocking).", "Decision engine that filters answers based on support levels."],
        ["Phase 6\n(W12 - W14)", "Develop local Streamlit web application. Run benchmarks on RAGTruth/HaluEval. Plot ROC/Precision-Recall curves.", "Streamlit UI demo and comprehensive performance evaluation charts."],
        ["Phase 7\n(W15 - W16)", "Perform error analysis. Compile thesis chapters. Conduct review defense preparation.", "Final M.Tech Project Report and slide deck ready for defense."]
    ]
    
    story.append(make_table(timeline_data, [85, 279, 140], styles))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("Critical Path & Milestones Graph", styles['ProjectSubsectionHeading']))
    
    milestone_ascii = """
 W1-2: Chunks  ===>  W3-4: FAISS  ===>  W5-6: Local LLM  ===>  W7-9: Detector
                                                                   ||
 W15-16: Thesis <=== W12-14: Demo UI <=== W10-11: Controller <====//
    """
    story.append(Paragraph(milestone_ascii, styles['ProjectCodeBlock']))
    
    story.append(PageBreak())

    # ================= PAGE 7: EVALUATION METRICS & DEMO PLAN =================
    story.append(Paragraph("System Evaluation & Demo Visualization Plan", styles['ProjectSectionHeading']))
    story.append(create_divider())
    
    story.append(Paragraph("1. Evaluation Metrics", styles['ProjectSubsectionHeading']))
    
    metric_data = [
        ["Metric Category", "Specific Metrics Used", "Formula / Concept", "Target Value"],
        ["Retrieval", "Recall@K, MRR (Mean Reciprocal Rank)", "Percentage of relevant chunks in top-K; position of first relevant chunk.", "Recall@3 >= 0.85\nMRR >= 0.80"],
        ["Detection Accuracy", "F1-Score, AUC-ROC, Precision", "Harmonic mean of precision and recall. Trade-off curve for hallucination vs truth.", "F1 >= 0.82\nAUC-ROC >= 0.88"],
        ["Faithfulness", "Provenance/Support Score, Citation Coverage", "Average NLI entailment probability across all sentences. Citation density.", "Support Score >= 0.85\nCoverage = 100%"],
        ["System Performance", "Inference Latency, Memory Footprint", "Time in seconds to generate and verify answer. GPU/RAM consumption.", "Latency <= 5.0s\nMemory <= 12GB RAM"]
    ]
    story.append(make_table(metric_data, [105, 130, 179, 90], styles))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("2. Web Demo Visualization Layout (Streamlit Dashboard)", styles['ProjectSubsectionHeading']))
    story.append(Paragraph("The Streamlit dashboard serves as the user-facing demonstration and visual proof of the system's explainability features. The layout is divided into three primary segments:", styles['ProjectBody']))
    
    ui_desc_1 = "<b>Query & Generation Box:</b> Input text field for user query. Real-time answer generation box showing words as they stream, and color-coding sentences based on support (Green = Entailed, Orange = Weak Support, Red = Contradicted/Unsupported)."
    ui_desc_2 = "<b>Explainability Panel:</b> Toggleable view displaying attention heatmaps over retrieved chunks, cosine similarity bar charts for retrieved documents, and direct mapping indicating which sentence came from which source chunk."
    ui_desc_3 = "<b>Safety Gauges & Logs:</b> Speedometer showing the aggregated Provenance Score (0 to 100%). Log console printing model decisions: <i>'ACTION: Cite source'</i> or <i>'ACTION: Suppressed claim #3 due to NLI Contradiction'</i>."
    
    story.append(Paragraph(f"&bull; {ui_desc_1}", styles['ProjectBullet']))
    story.append(Paragraph(f"&bull; {ui_desc_2}", styles['ProjectBullet']))
    story.append(Paragraph(f"&bull; {ui_desc_3}", styles['ProjectBullet']))
    
    story.append(make_callout("The Streamlit UI runs entirely on localhost:8501, loading the models locally via Hugging Face pipelines and Ollama endpoints, preserving the strictly offline nature of the framework.", styles, "LOCAL DEPLOYMENT CONFIG", "info"))
    
    story.append(PageBreak())

    # ================= PAGE 8: THESIS STRUCTURE & ACADEMIC OUTLINE =================
    story.append(Paragraph("M.Tech Thesis Structure & Academic Outline", styles['ProjectSectionHeading']))
    story.append(create_divider())
    
    thesis_intro = "The project report/thesis will be organized into six academic chapters in accordance with standard graduate engineering guidelines, focusing on mathematical formulation, experimental rigor, and system validation."
    story.append(Paragraph(thesis_intro, styles['ProjectBody']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Chapter 1: Introduction", styles['ProjectSubsectionHeading']))
    story.append(Paragraph("Background on Generative AI and RAG; The threat of factual hallucinations; Contextualizing the problem in offline environments; Research questions, scope, and objectives of the work; Document structure overview.", styles['ProjectBody']))
    
    story.append(Paragraph("Chapter 2: Literature Review", styles['ProjectSubsectionHeading']))
    story.append(Paragraph("Historical overview of RAG systems; Survey of hallucination types (intrinsic vs. extrinsic); Critical review of post-hoc fact checking vs. internal explainability; Entailment, BERTScore, and semantic metrics; Identification of current research gaps.", styles['ProjectBody']))
    
    story.append(Paragraph("Chapter 3: System Design and Architecture", styles['ProjectSubsectionHeading']))
    story.append(Paragraph("Detailed pipeline architecture; Mathematical modeling of explainability indicators; Formula for combining NLI entailment probabilities, BERTScore weights, and cosine similarity; State transitions of the Decision Controller.", styles['ProjectBody']))
    
    story.append(Paragraph("Chapter 4: Implementation Methodology", styles['ProjectSubsectionHeading']))
    story.append(Paragraph("Setting up the offline pipeline; Local models configuration (all-MiniLM-L6-v2, Cross-Encoder, quantized Mistral-7B); Tokenization and vector store construction with FAISS; Custom paragraph and claim parser development.", styles['ProjectBody']))
    
    story.append(Paragraph("Chapter 5: Experimental Results and Analysis", styles['ProjectSubsectionHeading']))
    story.append(Paragraph("Testing corpus details; Verification of retrieval accuracy (MRR, Recall@K); Precision-Recall performance of the NLI detector; ROC curves and optimal decision threshold calibration; Latency vs. accuracy trade-off curves.", styles['ProjectBody']))
    
    story.append(Paragraph("Chapter 6: Conclusion and Future Work", styles['ProjectSubsectionHeading']))
    story.append(Paragraph("Summary of contributions; Answers to primary research questions; Project limitations (hardware constraints, vocabulary overlap); Suggestions for future extensions (attention map extraction, multi-lingual adaptation).", styles['ProjectBody']))
    
    story.append(Spacer(1, 10))
    story.append(make_callout("All bibliography entries will follow the standard IEEE formatting guidelines, ensuring high-quality citations of major conferences (ACL, EMNLP, NeurIPS) and journals (IEEE Access, IEEE TKDE).", styles, "REPORT FORMATTING RULES", "info"))
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    build_pdf()
