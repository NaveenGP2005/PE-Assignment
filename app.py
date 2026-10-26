"""
Streamlit Web Application: LangChain RAG Research QA & Document Assistant
Designed for 10-Mark Prompt Engineering for GenAI Coursework & Live Demonstration.
"""
import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Add current directory to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from src.config import (
    SAMPLE_DATA_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    OPENAI_API_KEY
)
from src.document_loader import load_document
from src.text_splitter import split_documents
from src.vector_store import build_vector_store
from src.rag_engine import execute_rag_query, generate_document_summary
from src.prompt_templates import PERSONAS, get_prompt_template

# Page configuration
st.set_page_config(
    page_title="LangChain RAG Assistant | Prompt Engineering for GenAI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling from Stitch Design System (ArXivLens / ScholarPulse AI)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">

<style>
    /* Global Typography from Stitch */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    code, pre, .font-mono, [data-testid="stCodeBlock"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Canvas & Backgrounds */
    .stApp {
        background-color: #0c1324 !important;
        color: #dce1fb !important;
    }
    
    /* Top Header Banner with Stitch Glassmorphism */
    .main-header {
        background: linear-gradient(135deg, rgba(25, 31, 49, 0.85) 0%, rgba(35, 41, 60, 0.7) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(70, 69, 84, 0.45);
        padding: 24px 28px;
        border-radius: 12px;
        color: #dce1fb;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(2, 6, 23, 0.5);
    }
    .main-header h1 {
        color: #f8fafc !important;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #c7c4d7;
        font-size: 0.98rem;
        margin: 0;
    }
    
    /* Stitch Pills & Telemetry Badges */
    .telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(78, 222, 163, 0.12);
        color: #4edea3;
        border: 1px solid rgba(78, 222, 163, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #4edea3;
        display: inline-block;
        box-shadow: 0 0 8px #4edea3;
    }

    /* Citation Tags from Stitch */
    .citation-tag {
        display: inline-flex;
        align-items: center;
        background: rgba(128, 131, 255, 0.14);
        color: #c0c1ff;
        border: 1px solid rgba(128, 131, 255, 0.35);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 4px;
    }

    /* Retrieved Chunk Cards */
    .chunk-card {
        background: #151b2d;
        border: 1px solid rgba(70, 69, 84, 0.4);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .chunk-card:hover {
        border-color: rgba(128, 131, 255, 0.6);
        background: #191f31;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #070d1f !important;
        border-right: 1px solid rgba(70, 69, 84, 0.35) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stSlider {
        margin-bottom: 10px;
    }
    
    /* Streamlit Tab Navigation Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #151b2d;
        padding: 6px 8px 0 8px;
        border-radius: 8px 8px 0 0;
        border-bottom: 1px solid rgba(70, 69, 84, 0.4);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.86rem;
        font-weight: 500;
        color: #908fa0 !important;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0c1324 !important;
        color: #c0c1ff !important;
        border-bottom: 2px solid #8083ff !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border: 1px solid rgba(128, 131, 255, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.5) !important;
    }

    /* Markdown Table Formatting (Fixes Column Text Overlap & Clipping) */
    table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 14px 0 !important;
        overflow-x: auto !important;
        display: block !important;
        border-radius: 6px !important;
    }
    th {
        background-color: #191f31 !important;
        color: #c0c1ff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        padding: 10px 14px !important;
        border: 1px solid rgba(70, 69, 84, 0.5) !important;
        text-align: left !important;
        white-space: nowrap !important;
    }
    td {
        background-color: #151b2d !important;
        color: #dce1fb !important;
        font-size: 0.85rem !important;
        padding: 10px 14px !important;
        border: 1px solid rgba(70, 69, 84, 0.3) !important;
        line-height: 1.5 !important;
        min-width: 140px !important;
    }
    tr:nth-child(even) td {
        background-color: #111728 !important;
    }

    /* Comparison Result Container */
    .compare-container {
        background: #151b2d;
        border: 1px solid rgba(70, 69, 84, 0.45);
        border-radius: 8px;
        padding: 18px 22px;
        margin-top: 12px;
        color: #dce1fb;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary_text" not in st.session_state:
    st.session_state.summary_text = None
if "comparison_data" not in st.session_state:
    st.session_state.comparison_data = None


# ---------------- SIDEBAR: CONTROLS & CONFIGURATION ----------------
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    
    # Section 1: Pre-bundled Demo Paper
    st.subheader("1. Quick Demo Setup")
    sample_pdf_path = SAMPLE_DATA_DIR / "sample_paper_rag.pdf"
    sample_md_path = SAMPLE_DATA_DIR / "sample_paper_rag.md"
    
    if st.button("🚀 Load Pre-Indexed Sample Paper", use_container_width=True, type="primary"):
        with st.spinner("Indexing sample paper into FAISS vector database..."):
            target_path = sample_pdf_path if sample_pdf_path.exists() else sample_md_path
            raw_docs = load_document(target_path)
            chunks = split_documents(raw_docs, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
            st.session_state.vector_store = build_vector_store(chunks)
            st.session_state.doc_metadata = {
                "source": target_path.name,
                "pages": len(raw_docs),
                "chunks": len(chunks)
            }
            st.session_state.chat_history = []
            st.session_state.summary_text = None
            st.success(f"Loaded '{target_path.name}' ({len(chunks)} chunks)!")

    # Section 2: Upload Custom Document
    st.subheader("Or Upload Custom Document")
    uploaded_file = st.file_uploader("Upload PDF, TXT, or MD", type=["pdf", "txt", "md"])
    if uploaded_file is not None:
        if st.button("Process & Index Uploaded File", use_container_width=True):
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                save_dest = SAMPLE_DATA_DIR / uploaded_file.name
                with open(save_dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                raw_docs = load_document(save_dest)
                chunks = split_documents(raw_docs, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
                st.session_state.vector_store = build_vector_store(chunks)
                st.session_state.doc_metadata = {
                    "source": uploaded_file.name,
                    "pages": len(raw_docs),
                    "chunks": len(chunks)
                }
                st.session_state.chat_history = []
                st.session_state.summary_text = None
                st.success(f"Indexed {len(chunks)} chunks!")

    st.divider()

    # Section 3: Prompt Engineering Parameters (CRUCIAL FOR COURSEWORK)
    st.subheader("2. Prompt Engineering Controls")
    
    selected_persona = st.selectbox(
        "🎭 System Persona / Role:",
        options=list(PERSONAS.keys()),
        index=0,
        help="Defines the behavior, tone, and guardrails injected into the system prompt."
    )
    
    selected_strategy = st.selectbox(
        "🧠 Prompting Strategy:",
        options=[
            "Zero-Shot (Direct Answer)",
            "Few-Shot (In-Context Exemplars)",
            "Chain-of-Thought (Step-by-Step Reasoning)"
        ],
        index=0,
        help="Selects the prompt engineering methodology."
    )
    
    temperature = st.slider("🌡️ Temperature (Creativity):", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    top_k = st.slider("🔍 Retrieval Top-K Chunks:", min_value=1, max_value=8, value=DEFAULT_TOP_K, step=1)

    st.divider()

    # Reload env dynamically on each run so changes to .env reflect immediately
    load_dotenv(override=True)
    groq_key_from_env = os.getenv("GROQ_API_KEY", "")
    gemini_key_from_env = os.getenv("GOOGLE_API_KEY", "")
    openai_key_from_env = os.getenv("OPENAI_API_KEY", "")

    # Section 4: Model Provider Configuration
    st.subheader("3. Model Provider")
    provider_options = [
        "Groq (Ultra-Fast Cloud)" if groq_key_from_env else "Offline (Local / Extractive Mode)",
        "Offline (Local / Extractive Mode)" if groq_key_from_env else "Groq (Ultra-Fast Cloud)",
        "Google Gemini",
        "OpenAI"
    ]
    # Remove duplicates preserving order
    provider_options = list(dict.fromkeys(provider_options))

    provider = st.selectbox(
        "Select LLM Provider:",
        options=provider_options,
        index=0,
        help="Groq offers ultra-fast cloud inference. Offline mode works 100% locally with zero internet requirements."
    )
    
    api_key = ""
    model_name = ""
    
    if "Groq" in provider:
        api_key = st.text_input("Groq API Key:", value=groq_key_from_env, type="password", help="Get free key from console.groq.com")
        model_name = st.selectbox(
            "Groq Model (Latest Cloud):",
            [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "qwen/qwen3.8-27b",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant"
            ],
            index=0,
            help="openai/gpt-oss-120b & 20b and Qwen are active high-speed models on GroqCloud."
        )
    elif provider == "Google Gemini":
        api_key = st.text_input("Gemini API Key:", value=gemini_key_from_env, type="password", help="Get free key from aistudio.google.com")
        model_name = st.selectbox("Gemini Model:", ["gemini-1.5-flash", "gemini-1.5-pro"])
    elif provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key:", value=openai_key_from_env, type="password")
        model_name = st.selectbox("OpenAI Model:", ["gpt-4o-mini", "gpt-4o"])


# ---------------- MAIN CONTENT AREA (STITCH DESIGN SYSTEM) ----------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                <span class="telemetry-pill"><span class="pulse-dot"></span> RAG CORE v3.1 ACTIVE</span>
                <span style="font-family: 'JetBrains Mono'; font-size: 0.78rem; color: #a78bfa; background: rgba(167, 139, 250, 0.12); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(167, 139, 250, 0.25);">ArXivLens Engine</span>
            </div>
            <h1>ScholarPulse AI • Research Synthesis Dashboard</h1>
            <p>AI-Powered Research Paper Summarization & Multi-Strategy Prompt Engineering QA</p>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
            <a href="file:///d:/Study/Assignment/PEGenAI/web/stitch_dashboard.html" target="_blank" style="text-decoration: none;">
                <span style="display: inline-flex; align-items: center; gap: 6px; background: rgba(128, 131, 255, 0.2); color: #c0c1ff; border: 1px solid rgba(128, 131, 255, 0.4); padding: 6px 12px; border-radius: 6px; font-family: 'JetBrains Mono'; font-size: 0.8rem; font-weight: 500;">
                    🌐 Open Full Stitch HTML Dashboard
                </span>
            </a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Status bar with Stitch cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.session_state.doc_metadata:
        st.markdown(f"""
        <div style="background: #151b2d; border: 1px solid rgba(70, 69, 84, 0.4); border-left: 3px solid #8083ff; padding: 10px 14px; border-radius: 6px;">
            <div style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #908fa0; text-transform: uppercase;">Active Document</div>
            <div style="font-weight: 600; color: #f8fafc; font-size: 0.95rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{st.session_state.doc_metadata['source']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #151b2d; border: 1px solid rgba(70, 69, 84, 0.4); border-left: 3px solid #908fa0; padding: 10px 14px; border-radius: 6px;">
            <div style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #908fa0; text-transform: uppercase;">Active Document</div>
            <div style="font-weight: 600; color: #908fa0; font-size: 0.95rem;">No Document Loaded</div>
        </div>
        """, unsafe_allow_html=True)
with col2:
    chunk_val = st.session_state.doc_metadata['chunks'] if st.session_state.doc_metadata else 0
    st.markdown(f"""
    <div style="background: #151b2d; border: 1px solid rgba(70, 69, 84, 0.4); border-left: 3px solid #4edea3; padding: 10px 14px; border-radius: 6px;">
        <div style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #908fa0; text-transform: uppercase;">FAISS Vector Index</div>
        <div style="font-weight: 600; color: #4edea3; font-size: 0.95rem;">{chunk_val} Chunks (384-d Dense)</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div style="background: #151b2d; border: 1px solid rgba(70, 69, 84, 0.4); border-left: 3px solid #cebdff; padding: 10px 14px; border-radius: 6px;">
        <div style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #908fa0; text-transform: uppercase;">Prompt Strategy</div>
        <div style="font-weight: 600; color: #cebdff; font-size: 0.95rem;">{selected_strategy.split(' ')[0]}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div style="background: #151b2d; border: 1px solid rgba(70, 69, 84, 0.4); border-left: 3px solid #8083ff; padding: 10px 14px; border-radius: 6px;">
        <div style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #908fa0; text-transform: uppercase;">Active Inference</div>
        <div style="font-weight: 600; color: #c0c1ff; font-size: 0.95rem;">{provider.split(' ')[0]} (120B)</div>
    </div>
    """, unsafe_allow_html=True)


# Main Tabs
tab_chat, tab_summary, tab_compare, tab_arch = st.tabs([
    "💬 Interactive QA & Inspection",
    "📊 Executive Summary",
    "🧪 Prompt Strategy Comparison",
    "📐 Architecture & LaTeX Report"
])


# ---------------- TAB 1: INTERACTIVE CHAT & INSPECTION ----------------
with tab_chat:
    if st.session_state.vector_store is None:
        st.info("👋 **Welcome to the GenAI Demo!** Click **'🚀 Load Pre-Indexed Sample Paper'** in the left sidebar to start instantly, or upload your own PDF.")
    else:
        # Sample Question Pills
        st.markdown("**💡 Quick Demonstration Questions:**")
        q_cols = st.columns(3)
        sample_q = None
        with q_cols[0]:
            if st.button("What problem does RAG solve?"):
                sample_q = "What are the fundamental limitations of pretrained models that RAG solves?"
        with q_cols[1]:
            if st.button("How does DPR compute scores?"):
                sample_q = "Explain the Dense Passage Retriever (DPR) and how similarity scores are calculated."
        with q_cols[2]:
            if st.button("What were the Exact Match results?"):
                sample_q = "What benchmark datasets were evaluated and what were the Exact Match results on Natural Questions?"

        # Chat query input
        user_query = st.chat_input("Ask a question about the document...")
        query_to_run = user_query or sample_q

        if query_to_run:
            with st.spinner("Retrieving vector embeddings and executing prompt pipeline..."):
                result = execute_rag_query(
                    query=query_to_run,
                    vector_store=st.session_state.vector_store,
                    persona_name=selected_persona,
                    strategy_name=selected_strategy,
                    provider=provider,
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    top_k=top_k
                )
                
                # Append to history
                st.session_state.chat_history.append({
                    "query": query_to_run,
                    "answer": result["answer"],
                    "formatted_prompt": result["formatted_prompt"],
                    "retrieved_chunks": result["retrieved_chunks"],
                    "model_used": result["model_used"],
                    "execution_time": result["execution_time"]
                })

        # Display Chat History (most recent on top)
        if st.session_state.chat_history:
            for idx, item in enumerate(reversed(st.session_state.chat_history)):
                with st.chat_message("user"):
                    st.write(item["query"])
                with st.chat_message("assistant"):
                    st.markdown(item["answer"])
                    st.caption(f"⚡ Latency: {item['execution_time']}s | Model: {item['model_used']}")
                    
                    # Collapsible Prompt Inspector (HIGH IMPACT FOR VIVA!)
                    with st.expander("🔍 [VIVA DEMO] Inspect Formatted Prompt Injected to Model"):
                        st.markdown("**Actual System & Few-Shot Prompt Built by LangChain:**")
                        st.code(item["formatted_prompt"], language="markdown")
                    
                    # Collapsible Retrieved Chunks
                    with st.expander(f"📑 Retrieved Chunks ({len(item['retrieved_chunks'])} passages)"):
                        for c_idx, (c_doc, score) in enumerate(item["retrieved_chunks"], 1):
                            p = c_doc.metadata.get("page", 1)
                            src = c_doc.metadata.get("source", "Doc")
                            st.markdown(f"""
                            <div class="chunk-card">
                                <span class="citation-tag">Chunk #{c_idx}</span>
                                <span class="citation-tag">Page {p}</span>
                                <span class="citation-tag">Distance: {score:.4f}</span>
                                <p style="margin-top: 8px; font-size: 0.9rem; color: #334155;">{c_doc.page_content}</p>
                            </div>
                            """, unsafe_allow_html=True)
                st.divider()


# ---------------- TAB 2: EXECUTIVE SUMMARY ----------------
with tab_summary:
    st.subheader("📄 Automated Structured Document Summary")
    st.write("Generates an executive synopsis covering Core Objectives, Methodology, Results, and Takeaways.")
    
    if st.session_state.vector_store is None:
        st.warning("Please index a document first from the sidebar.")
    else:
        if st.button("⚡ Generate Structured Executive Summary", type="primary"):
            with st.spinner("Synthesizing multi-chunk document summary..."):
                sum_res = generate_document_summary(
                    vector_store=st.session_state.vector_store,
                    persona_name="Executive Summarizer",
                    provider=provider,
                    api_key=api_key,
                    model_name=model_name
                )
                st.session_state.summary_text = sum_res["summary"]
                
        if st.session_state.summary_text:
            st.markdown(st.session_state.summary_text)


# ---------------- TAB 3: PROMPT STRATEGY BENCHMARK ----------------
with tab_compare:
    st.subheader("🧪 Prompt Strategy Comparison & Benchmark")
    st.write(
        "Demonstrate the practical difference between **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought (CoT)** "
        "on the exact same query and document context."
    )
    
    if st.session_state.vector_store is None:
        st.warning("Please index a document first from the sidebar.")
    else:
        # Pre-configured recommended queries for convenience
        default_q = "Compare CityGML, IndoorGML, and BIM in terms of geometric forms, semantic information, and application scenarios."
        
        col_q, col_quick = st.columns([3, 1])
        with col_q:
            compare_query = st.text_input(
                "Comparison Query:",
                value=default_q,
                help="Enter any complex analytical question to compare prompt strategies."
            )
        with col_quick:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            view_mode = st.radio(
                "Layout:",
                ["📑 Tabbed View", "📊 3-Column Grid"],
                horizontal=True,
                help="Tabbed view gives 100% width for wide comparison tables and mathematical formulas."
            )

        if st.button("🚀 Run Multi-Strategy Comparison", type="primary"):
            with st.spinner("Executing comparative inference across 3 prompt techniques with Groq..."):
                strategies = [
                    "Zero-Shot (Direct Answer)",
                    "Few-Shot (In-Context Exemplars)",
                    "Chain-of-Thought (Step-by-Step Reasoning)"
                ]
                
                results_list = []
                for strat in strategies:
                    res = execute_rag_query(
                        query=compare_query,
                        vector_store=st.session_state.vector_store,
                        persona_name=selected_persona,
                        strategy_name=strat,
                        provider=provider,
                        api_key=api_key,
                        model_name=model_name,
                        top_k=top_k
                    )
                    results_list.append({
                        "strategy": strat,
                        "answer": res["answer"],
                        "formatted_prompt": res["formatted_prompt"],
                        "latency": res["execution_time"],
                        "model": res["model_used"]
                    })
                
                st.session_state.comparison_data = {
                    "query": compare_query,
                    "results": results_list
                }

        # Render Comparison Results
        if st.session_state.comparison_data:
            comp = st.session_state.comparison_data
            st.markdown(f"**Evaluated Query:** *\"{comp['query']}\"*")
            
            if "Tabbed" in view_mode:
                # 100% Full-Width Tabbed View (ZERO TEXT OVERLAP)
                subtab1, subtab2, subtab3 = st.tabs([
                    "🎯 Zero-Shot (Direct)",
                    "📝 Few-Shot (In-Context Exemplars)",
                    "🧠 Chain-of-Thought (CoT Step-by-Step)"
                ])
                subtabs = [subtab1, subtab2, subtab3]
                
                for idx, item in enumerate(comp["results"]):
                    with subtabs[idx]:
                        col_m1, col_m2 = st.columns([1, 1])
                        with col_m1:
                            st.caption(f"⚡ Latency: **{item['latency']}s** | Engine: **{item['model']}**")
                        with col_m2:
                            st.caption(f"🏷️ Strategy: **{item['strategy']}**")
                        
                        st.markdown(f"""
                        <div class="compare-container">
                        """, unsafe_allow_html=True)
                        st.markdown(item["answer"])
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        with st.expander(f"🔍 Inspect Prompt Template for {item['strategy'].split(' ')[0]}"):
                            st.code(item["formatted_prompt"], language="markdown")
            else:
                # 3-Column Side-by-Side Grid with scrollable table support
                cols = st.columns(3)
                for idx, item in enumerate(comp["results"]):
                    with cols[idx]:
                        st.markdown(f"#### {item['strategy'].split(' ')[0]}")
                        st.caption(f"⚡ {item['latency']}s • {item['model'].split(' ')[0]}")
                        st.markdown(f"""
                        <div class="compare-container" style="padding: 12px; font-size: 0.9rem;">
                        """, unsafe_allow_html=True)
                        st.markdown(item["answer"])
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        with st.expander("View Prompt Format"):
                            st.code(item["formatted_prompt"][:350] + "\n...", language="markdown")


# ---------------- TAB 4: ARCHITECTURE & REPORT ----------------
with tab_arch:
    st.subheader("📐 System Architecture & Evaluation Report")
    
    st.markdown("""
    ### 1. End-to-End LangChain RAG Pipeline
    ```
    [PDF / Research Paper] 
            │
            ▼
    [PyPDF Loader] ──► Text Extraction & Page Metadata
            │
            ▼
    [RecursiveCharacterTextSplitter] ──► 800-char chunks (150 overlap)
            │
            ▼
    [Sentence-Transformers Embeddings] ──► Local 384-d dense vectors
            │
            ▼
    [FAISS Vector Index] ──► MIPS Index on Local Disk
            │
            ▼ (At Query Time)
    [User Query] ──► Vector Search ──► Top-K Relevant Chunks
            │                                  │
            └──────────────┬───────────────────┘
                           ▼
    [Prompt Engineering Engine]
      • Persona Injection (Researcher / Summarizer / Fact-Checker)
      • Strategy (Zero-Shot / Few-Shot / Chain-of-Thought)
      • Strict Citation & Anti-Hallucination Guardrails
                           │
                           ▼
    [LLM Inference (Gemini / Groq / OpenAI / Local Fallback)]
                           │
                           ▼
    [Cited Response + Page Provenance + Prompt Inspector]
    ```
    """)
    
    st.markdown("### 2. LaTeX Report Details")
    st.info(
        "A complete, ready-to-compile academic LaTeX report has been generated in `report/report.tex`. "
        "It includes the abstract, system architecture, prompt templates, mathematical formulation, and evaluation tables."
    )
