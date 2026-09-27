"""
Streamlit UI — Conversational Interface (ChatGPT Style)
=======================================================
A modern, conversational interface for the Explainability-Driven Hallucination
Detection Framework. 

Features:
  • ChatGPT-like chat history and input.
  • Color-coded sentences (Green=Supported, Yellow=Weak, Red=Hallucinated)
    rendered directly in the chat bubble.
  • Expandable "Explainability & Detection Logs" panel inside each AI response
    to show charts, provenance scores, and retrieved chunks.
  • Sidebar for indexing and configuration.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Page config & Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="XAI-RAG Chat",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Premium Dark Theme */
    .stApp {
        background: #0E1117;
        color: #FAFAFA;
    }

    /* Top Header */
    .header-container {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 2rem;
        background: linear-gradient(180deg, rgba(30,30,47,0.8) 0%, rgba(14,17,23,0) 100%);
    }
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* Sentence color coding within chat */
    .sent-supported {
        background-color: rgba(34,197,94,0.1);
        border-bottom: 2px solid #22c55e;
        padding: 2px 4px;
        border-radius: 4px;
        margin: 2px 0;
        display: inline;
    }
    .sent-weak {
        background-color: rgba(245,158,11,0.1);
        border-bottom: 2px solid #f59e0b;
        padding: 2px 4px;
        border-radius: 4px;
        margin: 2px 0;
        display: inline;
    }
    .sent-hallucinated {
        background-color: rgba(239,68,68,0.1);
        border-bottom: 2px solid #ef4444;
        padding: 2px 4px;
        border-radius: 4px;
        margin: 2px 0;
        display: inline;
        color: #fca5a5;
        text-decoration: line-through;
    }

    /* Status Badges */
    .status-badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 12px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-cite { background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid rgba(34,197,94,0.5); }
    .badge-warn { background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid rgba(245,158,11,0.5); }
    .badge-block { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.5); }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "index_stats" not in st.session_state:
    st.session_state.index_stats = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your offline, explainability-driven RAG assistant. Build the document index from the sidebar, then ask me anything. I will cite my sources and flag any hallucinations.", "result": None}
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Helper to render XAI charts inside the chat
# ─────────────────────────────────────────────────────────────────────────────

def render_explainability_panel(result):
    """Renders the detailed charts and tables for a given pipeline result."""
    
    # 1. Top Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        score = result.summary.get("overall_support_score", 0)
        st.metric("Grounding Score", f"{score:.1%}")
    with col2:
        prov = result.xai_report.get("provenance", {}).get("provenance_score", 0)
        st.metric("Provenance Score", f"{prov:.1%}")
    with col3:
        n_hal = result.summary.get("num_hallucinated", 0)
        total = result.summary.get("total_sentences", 0)
        st.metric("Hallucinated Claims", f"{n_hal} / {total}")
    with col4:
        latency = result.summary.get("latency", 0.0)
        st.metric("Query Latency", f"{latency:.2f}s")
    with col5:
        n_cited = len(result.response.cited_sources) if result.response else 0
        st.metric("Sources Cited", n_cited)

    st.markdown("---")

    # 2. JSON Export & Table
    col_dl, col_lbl = st.columns([1, 4])
    with col_dl:
        import json
        export_data = {
            "query": result.query,
            "raw_answer": result.raw_answer,
            "overall_support_score": result.summary.get("overall_support_score"),
            "provenance_score": result.xai_report.get("provenance", {}).get("provenance_score"),
            "latency_seconds": result.summary.get("latency"),
            "verdicts": [v.to_dict() for v in result.verdicts] if result.verdicts else [],
            "retrieved_chunks": result.retrieved_chunks,
        }
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="📥 Export Audit JSON",
            data=json_str,
            file_name=f"audit_report_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_lbl:
        st.markdown("**📊 Per-Claim Detection Log**")

    if result.verdicts:
        rows = []
        for v in result.verdicts:
            rows.append({
                "#": v.sentence_idx + 1,
                "Claim": v.sentence[:60] + "…",
                "NLI (DeBERTa)": f"{v.nli_score:.3f}",
                "BERTScore": f"{v.bertscore_f1:.3f}",
                "Cosine": f"{v.cosine_score:.3f}",
                "Final Score": f"{v.support_score:.3f}",
                "Verdict": v.classification,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 3. Charts side-by-side
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**🔍 Radar: Evidence Breakdown**")
        if result.verdicts:
            fig_radar = go.Figure()
            colors = {"SUPPORTED": "#22c55e", "WEAKLY_SUPPORTED": "#f59e0b", "HALLUCINATED": "#ef4444"}
            for v in result.verdicts:
                fig_radar.add_trace(go.Scatterpolar(
                    r=[v.nli_score, v.bertscore_f1, v.cosine_score, v.support_score, v.nli_score],
                    theta=["NLI", "BERTScore", "Cosine", "Aggregate", "NLI"],
                    name=f"Claim {v.sentence_idx+1}",
                    line=dict(color=colors.get(v.classification, "#64748b")),
                    fill="toself", opacity=0.15,
                ))
            fig_radar.update_layout(
                polar=dict(bgcolor="rgba(0,0,0,0)"),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                height=250, margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with chart_col2:
        st.markdown("**📊 Per-Claim Support Scores**")
        if result.verdicts:
            fig_bar = go.Figure()
            colors_map = {"SUPPORTED": "#22c55e", "WEAKLY_SUPPORTED": "#f59e0b", "HALLUCINATED": "#ef4444"}
            y_labels = [f"Claim {v.sentence_idx+1}" for v in result.verdicts]
            scores = [v.support_score for v in result.verdicts]
            bar_colors = [colors_map.get(v.classification, "#64748b") for v in result.verdicts]
            hover_texts = [v.sentence for v in result.verdicts]
            
            fig_bar.add_trace(go.Bar(
                x=scores,
                y=y_labels,
                orientation='h',
                marker_color=bar_colors,
                text=[f"{s:.2f}" for s in scores],
                textposition='auto',
                hovertext=hover_texts,
                hoverinfo='text+x',
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Support Score", range=[0, 1.05], gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(autorange="reversed"),
                font=dict(color="#94a3b8"),
                height=250,
                margin=dict(l=60, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # 4. Word-Level Saliency maps
    st.markdown("---")
    st.markdown("**🔥 Word-Level Saliency Heatmaps (Leave-One-Out Embedding)**")
    
    for attr in result.xai_report.get("attributions", []):
        s_idx = attr.get("sentence_idx", 0)
        saliency_data = attr.get("word_saliency", [])
        
        if saliency_data:
            # Build HTML string
            html_parts = []
            for word_info in saliency_data:
                word = word_info.get("word", "")
                sal = word_info.get("saliency", 0.0)
                
                # Style background color based on saliency magnitude and sign
                if sal > 0.002:
                    # Positive saliency: word anchors to evidence
                    alpha = min(0.65, 0.08 + sal * 4.0)
                    bg_style = f"background-color: rgba(34, 197, 94, {alpha:.2f}); color: #ffffff; border-radius: 3px; padding: 1px 3px; border-bottom: 1.5px solid rgba(34, 197, 94, 0.6);"
                elif sal < -0.002:
                    # Negative saliency: word pulls sentence away
                    alpha = min(0.65, 0.08 + abs(sal) * 4.0)
                    bg_style = f"background-color: rgba(239, 68, 68, {alpha:.2f}); color: #ffffff; border-radius: 3px; padding: 1px 3px; border-bottom: 1.5px solid rgba(239, 68, 68, 0.6);"
                else:
                    # Near zero
                    bg_style = "color: #94a3b8; border-radius: 3px; padding: 1px 3px;"
                    
                html_parts.append(f'<span style="{bg_style}">{word}</span>')
                
            sentence_html = " ".join(html_parts)
            st.markdown(f"**Claim {s_idx+1}:**", unsafe_allow_html=True)
            st.markdown(f'<div style="line-height: 2.0; margin-bottom: 15px; background-color: #1a1e24; padding: 12px; border-radius: 5px; border-left: 3px solid #6366f1;">{sentence_html}</div>', unsafe_allow_html=True)

    # 5. Retrieved Chunks
    st.markdown("---")
    st.markdown("**📄 Retrieved Document Chunks**")
    if result.retrieved_chunks:
        for i, chunk in enumerate(result.retrieved_chunks):
            sim = chunk.get("similarity_score", 0)
            src = chunk.get('source_file', '?')
            pg = chunk.get('page_num', '?')
            with st.expander(f"Chunk {i+1} — {src} (Sim: {sim:.2f})"):
                st.caption(f"Page: {pg}")
                st.write(chunk.get("text", ""))


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    
    corpus_dir = st.text_input("Corpus Directory", value="data/corpus")
    
    if st.button("🔨 Build Document Index", use_container_width=True, type="primary"):
        with st.spinner("Loading models & indexing documents..."):
            try:
                from src.pipeline import RAGHallucinationPipeline
                if st.session_state.pipeline is None:
                    st.session_state.pipeline = RAGHallucinationPipeline()
                
                stats = st.session_state.pipeline.index_documents(corpus_dir=corpus_dir, force_reindex=True)
                st.session_state.index_stats = stats
                st.session_state.models_loaded = True
                st.success(f"✅ Indexed {stats.get('num_chunks', '?')} chunks!")
            except Exception as e:
                st.error(f"❌ Indexing failed: {e}")

    if st.session_state.index_stats:
        st.info(f"Database active: {st.session_state.index_stats.get('num_chunks')} chunks.")
    
    st.markdown("---")
    st.markdown("### 🎛️ Detection Tuning")
    w_nli = st.slider("NLI Weight", 0.0, 1.0, 0.45, 0.05)
    w_bert = st.slider("BERTScore Weight", 0.0, 1.0, 0.35, 0.05)
    w_cos = st.slider("Cosine Weight", 0.0, 1.0, 0.20, 0.05)
    
    st.markdown("### 🎯 Decision Thresholds")
    high_threshold = st.slider("High Support (Cite)", 0.0, 1.0, 0.75, 0.05)
    medium_threshold = st.slider("Medium Support (Warn)", 0.0, 1.0, 0.50, 0.05)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main Chat Interface
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-container">
     <div class="main-title">🛡️ Explainability-Driven RAG</div>
     <div class="sub-title">Explainability-Driven Hallucination Detection Framework for Offline RAG</div>
</div>
""", unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("result") is not None:
            # It's an AI response with pipeline data
            res = msg["result"]
            resp = res.response
            
            # 1. Render Status Badge
            if resp.state == "ANSWER_WITH_CITATIONS":
                st.markdown('<div class="status-badge badge-cite">✅ Cited Answer</div>', unsafe_allow_html=True)
            elif resp.state == "PARTIAL_ANSWER":
                st.markdown('<div class="status-badge badge-warn">⚠️ Partial Answer</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge badge-block">🚫 Blocked</div>', unsafe_allow_html=True)
            
            # 2. Render Color-Coded Text or Blocked Evidence Egress
            if resp.state == "BLOCKED":
                st.error(resp.warning_message or "Answer blocked due to high hallucination risk.")
                st.markdown("**🛡️ Safe Egress: Retrieved Evidence & Source Provenance**")
                if resp.retrieved_chunks:
                    for i, chunk in enumerate(resp.retrieved_chunks):
                        st.info(f"**Evidence {i+1}** — {chunk.get('source_file')}, p.{chunk.get('page_num')} (similarity: {chunk.get('similarity_score', 0.0):.3f})\n\n{chunk.get('text', '')}")
                else:
                    st.write("No retrieved chunks available.")
            else:
                if resp.warning_message:
                    st.warning(resp.warning_message)
                
                # Build HTML for sentences
                html_out = ""
                for v in res.verdicts:
                    if v.classification == "SUPPORTED":
                        html_out += f'<div class="sent-supported">{v.sentence} [{v.source_file}, p.{v.source_page}]</div> '
                    elif v.classification == "WEAKLY_SUPPORTED":
                        html_out += f'<div class="sent-weak">⚠ {v.sentence}</div> '
                    else:
                        html_out += f'<div class="sent-hallucinated">{v.sentence}</div> '
                
                st.markdown(html_out, unsafe_allow_html=True)
                
                if resp.cited_sources:
                    st.caption(f"**Sources:** {', '.join(resp.cited_sources)}")

            # 3. Render XAI Expander
            with st.expander("🔬 View Explainability & Detection Logs"):
                render_explainability_panel(res)
                
        else:
            # Standard text message (user or initial greeting)
            st.markdown(msg["content"])


# Handle new user input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Display user prompt immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process AI response
    with st.chat_message("assistant"):
        with st.spinner("Loading models and checking for hallucinations... (takes ~25s)"):
            if st.session_state.pipeline is None:
                from src.pipeline import RAGHallucinationPipeline
                st.session_state.pipeline = RAGHallucinationPipeline()
                st.session_state.pipeline._load_models()
                st.session_state.models_loaded = True

            pipe = st.session_state.pipeline
            
            # Update thresholds dynamically
            pipe.hall_engine.w_nli = w_nli
            pipe.hall_engine.w_bert = w_bert
            pipe.hall_engine.w_cos = w_cos
            pipe.hall_engine.high_thr = high_threshold
            pipe.hall_engine.medium_thr = medium_threshold
            pipe.controller.high_thr = high_threshold
            pipe.controller.medium_thr = medium_threshold

            # Run query
            start = time.time()
            result = pipe.query(prompt)
            elapsed = time.time() - start

            if result.error:
                st.error(f"Error: {result.error}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {result.error}", "result": None})
            else:
                # Store query latency in summary
                result.summary["latency"] = elapsed
                # Add to history and rerun to render properly via the loop above
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "", # Content is generated dynamically from the result
                    "result": result
                })
                st.rerun()
