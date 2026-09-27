import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfgen import canvas

class SinglePageCanvas(canvas.Canvas):
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
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        # Top Header Accent Bar
        self.setFillColor(colors.HexColor("#1B365D")) # Deep Navy
        self.rect(36, 758, 540, 4, fill=True, stroke=False)
        
        # Header text
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1B365D"))
        self.drawString(36, 746, "M.TECH MAIN PROJECT — DOMAIN EXPERTISE & COURSE ALIGNMENT SUMMARY")
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))
        self.drawRightString(576, 746, "TrustRAG Framework")
        
        # Header separator line
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.75)
        self.line(36, 740, 576, 740)
        
        # Bottom Footer separator line
        self.line(36, 36, 576, 36)
        
        # Footer text
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(36, 25, "Domain Expertise Certification & Project Justification | Udemy RAG with Python")
        self.drawRightString(576, 25, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf():
    pdf_filename = os.path.join("project_docs", "Domain_Expertise_Summary.pdf")
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1B365D'),
        spaceAfter=2
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=5
    )

    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#1B365D'),
        spaceBefore=4,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor('#1E293B')
    )

    story = []

    # Title Banner
    story.append(Paragraph("Domain Expertise & Course Alignment Summary", title_style))
    story.append(Paragraph("<b>Project Title:</b> TrustRAG: Hallucination Control and Evidence Attribution in Offline RAG", subtitle_style))

    # Metadata Card Table
    meta_data = [
        [
            Paragraph("<b>Course Name:</b> RAG with Python: Build Chatbots That Talk to Your Data", table_cell_style),
            Paragraph("<b>Platform:</b> Udemy", table_cell_style),
            Paragraph("<b>Duration:</b> 10 Hours 34 Minutes", table_cell_style)
        ],
        [
            Paragraph("<b>Instructor:</b> Python & AI Academy / Udemy", table_cell_style),
            Paragraph("<b>Requirement Met:</b> 10–20 Hours Mandate", table_cell_style),
            Paragraph("<b>Certificate Retained:</b> Yes (Verified)", table_cell_style)
        ],
        [
            Paragraph("<b>Course URL:</b> <font color='#2563EB'><u>https://www.udemy.com/course/rag-with-python-build-chatbots-that-talk-to-your-data/</u></font>", table_cell_style),
            "", ""
        ]
    ]
    meta_table = Table(meta_data, colWidths=[240, 150, 150])
    meta_table.setStyle(TableStyle([
        ('SPAN', (0, 2), (2, 2)),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 4))

    # Executive Overview
    story.append(Paragraph("1. Course Overview & Domain Relevance", section_style))
    overview_text = (
        "This 10-hour 34-minute course addresses core engineering and theoretical prerequisites "
        "for developing end-to-end Retrieval-Augmented Generation (RAG) systems in Python. It covers document chunking, "
        "vector embeddings, dense vector databases, conversational state management, and RAG evaluation frameworks. "
        "Completing this course satisfies the mandatory 10–20 hour domain expertise requirement "
        "for the M.Tech Main Project and provides immediate practical alignment with the <b>TrustRAG</b> framework."
    )
    story.append(Paragraph(overview_text, body_style))

    # Syllabus & Technical Alignment Table
    story.append(Paragraph("2. Course Syllabus & Project Alignment Matrix", section_style))
    
    syllabus_data = [
        [
            Paragraph("<b>Udemy Course Module / Topic</b>", table_header_style),
            Paragraph("<b>Key Skills & Concepts Covered</b>", table_header_style),
            Paragraph("<b>Direct Application in TrustRAG Project</b>", table_header_style)
        ],
        [
            Paragraph("<b>1. RAG Fundamentals & Architecture</b>", table_cell_style),
            Paragraph("RAG vs fine-tuning, retrieval pipelines, grounding LLM outputs with facts.", table_cell_style),
            Paragraph("Formulates the offline TrustRAG pipeline baseline architecture.", table_cell_style)
        ],
        [
            Paragraph("<b>2. Document Processing & Chunking</b>", table_cell_style),
            Paragraph("PyPDF, Recursive Text Splitters, metadata tracking, chunk size optimization.", table_cell_style),
            Paragraph("Processing MS MARCO passages and local PDF/text doc ingestion.", table_cell_style)
        ],
        [
            Paragraph("<b>3. Embeddings & Semantic Search</b>", table_cell_style),
            Paragraph("Sentence-Transformers, vector spaces, cosine similarity, dense retrieval.", table_cell_style),
            Paragraph("Sentence-BERT embedding generation and cosine similarity scoring.", table_cell_style)
        ],
        [
            Paragraph("<b>4. Vector Databases & Indexing</b>", table_cell_style),
            Paragraph("Vector stores (FAISS / ChromaDB), persistent indexing, top-k retrieval.", table_cell_style),
            Paragraph("Local FAISS vector store indexing for offline query processing.", table_cell_style)
        ],
        [
            Paragraph("<b>5. Conversational RAG & Pipelines</b>", table_cell_style),
            Paragraph("LangChain / LlamaIndex orchestration, memory buffer, prompt engineering.", table_cell_style),
            Paragraph("Orchestrating local Ollama LLM queries with retrieved context.", table_cell_style)
        ],
        [
            Paragraph("<b>6. RAG Evaluation & Hallucination</b>", table_cell_style),
            Paragraph("Ragas framework, faithfulness metrics, answer relevance, context precision.", table_cell_style),
            Paragraph("Multi-signal XAI verification (DeBERTa NLI, BERTScore, Tri-State).", table_cell_style)
        ]
    ]

    syl_table = Table(syllabus_data, colWidths=[130, 185, 225])
    syl_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(syl_table)
    story.append(Spacer(1, 4))

    # How the Course Supports TrustRAG
    story.append(Paragraph("3. Technical Justification: How the Course Supports TrustRAG", section_style))
    justification_points = [
        "<b>• Offline Vector Retrieval & Indexing:</b> Hands-on experience in building local FAISS indexes with Sentence-BERT embeddings. Forms the foundation for TrustRAG's offline vector retrieval on MS MARCO without external APIs.",
        "<b>• Explainable Hallucination Verification:</b> Semantic alignment and faithfulness modules informed TrustRAG's 3-signal XAI engine combining DeBERTa-v3 NLI entailment, BERTScore, and SBERT cosine similarity.",
        "<b>• Response Suppression Controller:</b> Studying RAG failure modes guided TrustRAG's Tri-State sentence classification (Supported, Weakly Supported, Hallucinated) and automated response suppression when thresholds are exceeded.",
        "<b>• End-to-End Local Execution:</b> Focus on privacy-preserving, local AI application design aligns directly with TrustRAG's zero-cloud, fully offline deployment requirements for sensitive data environments."
    ]
    for pt in justification_points:
        story.append(Paragraph(pt, body_style))

    # Summary Callout Box
    story.append(Spacer(1, 3))
    callout_content = Paragraph(
        "<b>Compliance & Retained Artifacts:</b> The course completion certificate and syllabus details have been retained "
        "in project documentation. The technical competencies acquired directly enable the design and execution of the M.Tech Main Project.",
        ParagraphStyle('CalloutText', parent=body_style, fontSize=7.5, leading=9.5, textColor=colors.HexColor('#0F5132'))
    )
    callout_table = Table([[callout_content]], colWidths=[540])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#D1E7DD')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#A3CFBB')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(callout_table)

    # Build the document
    doc.build(story, canvasmaker=SinglePageCanvas)
    print(f"Successfully generated {pdf_filename}")

if __name__ == "__main__":
    build_pdf()
