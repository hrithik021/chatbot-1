"""NEXUS RAG — Enterprise Document Intelligence Workspace
Three-column enterprise architecture inspired by the reference design.
Light enterprise palette (#F7F9FC / #FFFFFF) with deep navy sidebar (#0F2547).
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

import streamlit as st

# Custom modules
from components.icons import icons as ico
from rag.engine import RAGPipeline
from rag.llm import OllamaClient, GroqClient, GROQ_MODELS
from rag.prompts import (
    SYSTEM_PROMPT,
    format_rag_context,
    SUGGESTED_PROMPTS,
    DEMO_QUESTIONS,
    INSUFFICIENT_EVIDENCE_REPLY
)

# ──────────────────────────────────────────────
# PAGE CONFIGURATION & METADATA
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS RAG — Document Intelligence Workspace",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%232563EB' stroke-width='2'><polygon points='12 2 19 8 12 14 5 8'/><polygon points='12 10 19 16 12 22 5 16'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# DESIGN SYSTEM & PRODUCTION STYLESHEET
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-app:        #F7F9FC;
    --bg-surface:    #FFFFFF;
    --bg-sidebar:    #0F2547;
    --bg-sidebar-alt:#17335F;
    --bg-sidebar-hov:#1E3E73;
    --border-color:  #E4E9F1;
    --border-dark:   #D1D9E4;
    --text-primary:  #172033;
    --text-secondary:#667085;
    --text-muted:    #8B9BB4;
    --accent:        #2563EB;
    --accent-light:  #EFF6FF;
    --accent-hover:  #1D4ED8;
    --user-bubble:   #EBF3FE;
    --success:       #16A34A;
    --success-bg:    #F0FDF4;
    --warning:       #D97706;
    --warning-bg:    #FFFBEB;
    --error:         #DC2626;
    --radius-sm:     6px;
    --radius-md:     8px;
    --radius-lg:     12px;
    --radius-xl:     16px;
    --shadow-sm:     0 1px 3px rgba(16, 24, 40, 0.05);
    --shadow-md:     0 4px 8px -2px rgba(16, 24, 40, 0.06), 0 2px 4px -2px rgba(16, 24, 40, 0.04);
    --font-sans:     'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono:     'JetBrains Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    background-color: var(--bg-app) !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}

/* Hide Streamlit default decorations */
#MainMenu, footer, header, [data-testid="stToolbar"], .stDeployButton, [data-testid="stDecoration"] {
    display: none !important;
}

.main .block-container {
    padding: 1rem 1.5rem 2rem !important;
    max-width: 1480px;
}

/* ── DEEP NAVY SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.2rem 0.9rem !important;
}

.sidebar-brand-box {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.4rem 0.2rem 1.1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 0.8rem;
}
.sidebar-brand-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #FFFFFF !important;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.sidebar-brand-sub {
    font-size: 0.65rem;
    font-weight: 500;
    color: #8B9BB4 !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* Sidebar primary action button */
.sidebar-new-btn > div > button {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    height: 42px !important;
    width: 100% !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.35) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    transition: all 120ms ease !important;
}
.sidebar-new-btn > div > button:hover {
    background: var(--accent-hover) !important;
    transform: translateY(-1px);
}

.sidebar-nav-label {
    font-size: 0.62rem;
    font-weight: 700;
    color: #8B9BB4 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.2rem 0 0.4rem 0.2rem;
}

/* Chat history items */
.sidebar-chat-btn > div > button {
    background: transparent !important;
    color: #E2E8F0 !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.45rem 0.65rem !important;
    width: 100% !important;
    transition: all 100ms ease !important;
}
.sidebar-chat-btn > div > button:hover {
    background: var(--bg-sidebar-alt) !important;
    color: #FFFFFF !important;
}
.sidebar-chat-btn.active > div > button {
    background: var(--bg-sidebar-alt) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-left: 3px solid var(--accent) !important;
}

/* Sidebar bottom card */
.sidebar-bottom-promo {
    background: var(--bg-sidebar-alt);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-lg);
    padding: 0.85rem;
    margin-top: 1.5rem;
    text-align: center;
}
.promo-title {
    font-size: 0.76rem;
    font-weight: 700;
    color: #FFFFFF !important;
    margin-top: 6px;
}
.promo-sub {
    font-size: 0.65rem;
    color: #8B9BB4 !important;
    margin-top: 3px;
    line-height: 1.4;
}

/* ── MAIN WORKSPACE TOP HEADER ── */
.top-header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 0.85rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
}
.header-left-title {
    font-size: 0.95rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0;
    line-height: 1.2;
}
.header-left-sub {
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 2px;
}
.header-right-meta {
    display: flex;
    align-items: center;
    gap: 10px;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-app);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 0.3rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* ── CHAT CONTAINER & MESSAGES ── */
.chat-stream-box {
    min-height: 480px;
}

.msg-wrapper {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 1.25rem;
}

.avatar-circle {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.avatar-user {
    background: #DBEAFE;
    color: var(--accent);
}
.avatar-assistant {
    background: #0F2547;
    color: #FFFFFF;
}

/* User message bubble */
.bubble-user {
    background: var(--user-bubble);
    border: 1px solid #D0E1FD;
    border-radius: var(--radius-lg);
    padding: 0.85rem 1.1rem;
    max-width: 85%;
    font-size: 0.86rem;
    color: var(--text-primary);
    line-height: 1.55;
    position: relative;
}
.bubble-user-time {
    font-size: 0.64rem;
    color: var(--text-secondary);
    text-align: right;
    margin-top: 4px;
}

/* Assistant message card */
.card-assistant {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1rem 1.25rem;
    max-width: 95%;
    box-shadow: var(--shadow-sm);
    position: relative;
}
.assistant-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    border-bottom: 1px solid #F1F4F9;
    padding-bottom: 6px;
}
.assistant-name {
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 6px;
}
.assistant-time {
    font-size: 0.65rem;
    color: var(--text-muted);
}
.assistant-text {
    font-size: 0.86rem;
    line-height: 1.68;
    color: var(--text-primary);
}

/* Sources footer inside assistant card */
.card-sources-block {
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px dashed var(--border-color);
    font-size: 0.74rem;
}
.sources-title {
    font-weight: 700;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 5px;
    margin-bottom: 6px;
}
.source-citation-line {
    font-family: var(--font-mono);
    color: var(--accent);
    font-weight: 600;
    font-size: 0.72rem;
    margin-bottom: 2px;
}
.source-snippet-quote {
    color: var(--text-secondary);
    font-size: 0.72rem;
    line-height: 1.45;
    background: #F8FAFC;
    padding: 0.4rem 0.6rem;
    border-radius: var(--radius-sm);
    border: 1px solid #EEF2F6;
    margin-top: 3px;
}

/* ── EMPTY STATE ── */
.empty-box {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 2.2rem 1.8rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    margin-bottom: 1rem;
}
.empty-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
}
.empty-sub {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin: 6px auto 1.5rem;
    max-width: 520px;
    line-height: 1.55;
}

/* Suggestion cards */
.suggest-card-btn > div > button {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.75rem 0.9rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    height: 72px !important;
    width: 100% !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 120ms ease !important;
}
.suggest-card-btn > div > button:hover {
    border-color: var(--accent) !important;
    background: var(--accent-light) !important;
    transform: translateY(-1px);
}

/* ── COMPOSER ── */
.stChatInput {
    background: var(--bg-surface) !important;
    border-top: 1px solid var(--border-color) !important;
    padding: 0.75rem 1rem !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stChatInput textarea, .stChatInput input {
    background: #F8FAFC !important;
    border: 1px solid var(--border-dark) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-size: 0.85rem !important;
}
.stChatInput textarea:focus, .stChatInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12) !important;
}

/* ── RIGHT PANEL (DOCUMENTS & CONTEXT) ── */
.right-panel-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
}
.panel-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.85rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #F1F4F9;
}
.panel-card-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Document Row */
.doc-row-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.75rem;
    background: #F8FAFC;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    margin-bottom: 8px;
}
.doc-row-left {
    display: flex;
    align-items: center;
    gap: 8px;
    overflow: hidden;
}
.doc-row-name {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 170px;
}
.doc-row-meta {
    font-size: 0.68rem;
    color: var(--text-secondary);
}

/* Retrieved Context Card */
.retrieved-item-box {
    background: #F8FAFC;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 0.75rem 0.85rem;
    margin-bottom: 8px;
    font-size: 0.73rem;
    line-height: 1.5;
}
.retrieved-item-text {
    color: var(--text-primary);
    margin-bottom: 6px;
}
.retrieved-item-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.66rem;
    color: var(--accent);
    font-weight: 600;
    border-top: 1px dashed var(--border-color);
    padding-top: 5px;
}

/* Badge styles */
.badge-clean {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 7px;
    border-radius: 12px;
    font-size: 0.64rem;
    font-weight: 700;
}
.badge-clean-green {
    background: #DCFCE7;
    color: #15803D;
}
.badge-clean-amber {
    background: #FEF3C7;
    color: #B45309;
}
.badge-clean-blue {
    background: #DBEAFE;
    color: #1D4ED8;
}

/* Button general overrides */
.stButton > button {
    border-radius: var(--radius-md) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    transition: all 120ms ease !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────
def init_session_state():
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()

    if "conversations" not in st.session_state:
        default_id = str(uuid.uuid4())[:8]
        st.session_state.conversations = {
            default_id: {
                "id": default_id,
                "title": "Welcome Discussion",
                "created_at": datetime.now(),
                "messages": []
            }
        }
        st.session_state.active_chat_id = default_id

    if "chat_search" not in st.session_state:
        st.session_state.chat_search = ""

    if "settings" not in st.session_state:
        st.session_state.settings = {
            "backend": "ollama",
            "ollama_model": "mistral",
            "groq_key": "",
            "groq_model": GROQ_MODELS[0],
            "top_k": 4,
            "similarity_threshold": 0.28,
            "temperature": 0.1,
            "chunk_size": 500,
            "chunk_overlap": 80,
            "view_mode": "chat",  # "chat" | "evaluation" | "settings"
        }

    if "latest_inspection" not in st.session_state:
        st.session_state.latest_inspection = None

    if "eval_results" not in st.session_state:
        st.session_state.eval_results = {}

init_session_state()

pipeline: RAGPipeline = st.session_state.rag_pipeline
settings: Dict[str, Any] = st.session_state.settings
active_chat_id = st.session_state.active_chat_id
current_chat = st.session_state.conversations.get(active_chat_id, {
    "title": "New Chat", "messages": []
})

# ──────────────────────────────────────────────
# CONVERSATION MANAGEMENT HELPERS
# ──────────────────────────────────────────────
def create_new_chat(title: str = "New Conversation") -> str:
    new_id = str(uuid.uuid4())[:8]
    st.session_state.conversations[new_id] = {
        "id": new_id,
        "title": title,
        "created_at": datetime.now(),
        "messages": []
    }
    st.session_state.active_chat_id = new_id
    st.rerun()

def delete_chat(chat_id: str):
    if chat_id in st.session_state.conversations:
        del st.session_state.conversations[chat_id]
        if not st.session_state.conversations:
            create_new_chat()
        else:
            st.session_state.active_chat_id = list(st.session_state.conversations.keys())[0]
        st.rerun()

def load_sample_document_action():
    sample_path = "sample_docs/AI_and_ML_Overview.txt"
    try:
        with open(sample_path, "rb") as f:
            sdata = f.read()
        res = pipeline.ingest_document(sdata, "AI_and_ML_Overview.txt")
        if res["success"]:
            st.rerun()
    except Exception as e:
        st.error(f"Failed to read sample document: {e}")

# ══════════════════════════════════════════════
#  COLUMN 1: DEEP NAVY SIDEBAR (#0F2547)
# ══════════════════════════════════════════════
with st.sidebar:
    # ── Top Brand Header ──
    st.markdown(f"""
    <div class="sidebar-brand-box">
        <div>{ico.nexus_logo(24, "#6EA8FE")}</div>
        <div>
            <div class="sidebar-brand-title">NEXUS RAG</div>
            <div class="sidebar-brand-sub">Document Intelligence Workspace</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Primary Action: + New Chat ──
    st.markdown('<div class="sidebar-new-btn">', unsafe_allow_html=True)
    if st.button("+ New Chat", key="btn_side_new"):
        create_new_chat()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Chat History Header & Search ──
    st.markdown(f'<div class="sidebar-nav-label">Chat History</div>', unsafe_allow_html=True)

    search_q = st.text_input(
        "Search history...",
        placeholder="Search conversations...",
        label_visibility="collapsed",
        key="chat_search_input"
    )

    # ── Conversation List ──
    for cid, chat in sorted(
        st.session_state.conversations.items(),
        key=lambda item: item[1].get("created_at", datetime.now()),
        reverse=True
    ):
        title = chat.get("title", "Untitled")
        if search_q and search_q.lower() not in title.lower():
            continue

        is_active = (cid == active_chat_id)
        col_c1, col_c2 = st.columns([8.2, 1.8])
        with col_c1:
            cls_name = "sidebar-chat-btn active" if is_active else "sidebar-chat-btn"
            st.markdown(f'<div class="{cls_name}">', unsafe_allow_html=True)
            icon_char = "●" if is_active else "▸"
            if st.button(f"{icon_char} {title[:22]}", key=f"chat_{cid}"):
                st.session_state.active_chat_id = cid
                st.session_state.settings["view_mode"] = "chat"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col_c2:
            if st.button("×", key=f"del_{cid}", help="Delete chat"):
                delete_chat(cid)

    # ── Navigation Items ──
    st.markdown(f'<div class="sidebar-nav-label">Workspace</div>', unsafe_allow_html=True)

    if st.button("💬 Chat Workspace", key="nav_chat"):
        st.session_state.settings["view_mode"] = "chat"
        st.rerun()

    if st.button("🧪 Evaluation Suite (5 Tests)", key="nav_eval"):
        st.session_state.settings["view_mode"] = "evaluation"
        st.rerun()

    if st.button("⚙️ Settings & Inference", key="nav_settings"):
        st.session_state.settings["view_mode"] = "settings"
        st.rerun()

    # ── Bottom Knowledge Assistant Card ──
    st.markdown(f"""
    <div class="sidebar-bottom-promo">
        <div>{ico.nexus_logo(28, "#6EA8FE")}</div>
        <div class="promo-title">Document Knowledge Assistant</div>
        <div class="promo-sub">Grounded RAG architecture with verifiable citations</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  COLUMNS 2 & 3: MAIN CHAT & DOCUMENT PANEL
# ══════════════════════════════════════════════
col_main, col_panel = st.columns([6.8, 3.2], gap="medium")

# ──────────────────────────────────────────────
# COLUMN 2: CENTER WORKSPACE (CHAT / EVALUATION)
# ──────────────────────────────────────────────
with col_main:
    # ── Main Header ──
    is_indexed = pipeline.is_indexed
    ollama_live = OllamaClient.is_available()
    active_model = settings["ollama_model"] if settings["backend"] == "ollama" else settings["groq_model"]

    status_dot_color = "#16A34A" if ollama_live else "#DC2626"
    status_text = "Ollama Ready" if ollama_live else "Ollama Offline"

    doc_status_dot = "#16A34A" if is_indexed else "#D97706"
    doc_status_text = "Document Ready" if is_indexed else "No Document"

    st.markdown(f"""
    <div class="top-header-bar">
        <div>
            <h1 class="header-left-title">Document-Based AI Chatbot</h1>
            <div class="header-left-sub">Ask questions and get accurate answers derived directly from your documents</div>
        </div>
        <div class="header-right-meta">
            <div class="status-pill">
                {ico.dot(8, doc_status_dot)}
                <span>{doc_status_text}</span>
            </div>
            <div class="status-pill">
                {ico.dot(8, status_dot_color)}
                <span>Local LLM: <b>{active_model}</b></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── VIEW 1: CHAT WORKSPACE ──
    if settings["view_mode"] == "chat":
        messages = current_chat.get("messages", [])

        if not messages:
            # ── Clean Compact Empty State ──
            st.markdown(f"""
            <div class="empty-box">
                <div style="margin-bottom:0.6rem">{ico.nexus_logo(36, "#2563EB")}</div>
                <h2 class="empty-title">NEXUS RAG Workspace</h2>
                <div class="empty-sub">
                    Upload a PDF or TXT document to explore its contents with precision retrieval,
                    grounded answers, and verifiable source citations.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Suggestion cards (2x2 grid, each card auto-submits when clicked!)
            st.markdown('<div style="font-size:0.75rem;font-weight:700;color:var(--text-secondary);text-transform:uppercase;margin:0.5rem 0 0.5rem;letter-spacing:0.06em">Suggested Prompts</div>', unsafe_allow_html=True)

            sg_cols = st.columns(2)
            for idx, card in enumerate(SUGGESTED_PROMPTS):
                sg_col = sg_cols[idx % 2]
                with sg_col:
                    st.markdown('<div class="suggest-card-btn">', unsafe_allow_html=True)
                    btn_text = f"✦ {card['title']}\n{card['prompt'][:55]}..."
                    if st.button(btn_text, key=f"sg_btn_{idx}"):
                        st.session_state["queued_query"] = card["prompt"]
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            if not is_indexed:
                st.markdown('<div style="margin-top:1.2rem;text-align:center">', unsafe_allow_html=True)
                if st.button("📥 Use Sample Document (5-Page AI Overview)", key="btn_sample_empty"):
                    load_sample_document_action()
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            # ── Message Stream (Styled after Reference Image) ──
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                ts = msg.get("timestamp", "")
                sources = msg.get("sources", [])
                status_type = msg.get("status_type", "GROUNDED")
                top_score = msg.get("top_score", 0.0)

                if role == "user":
                    st.markdown(f"""
                    <div class="msg-wrapper">
                        <div class="avatar-circle avatar-user">
                            {ico.user(16, "#2563EB")}
                        </div>
                        <div class="bubble-user">
                            <div>{content}</div>
                            <div class="bubble-user-time">{ts}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Assistant Card
                    badge_markup = ""
                    if status_type == "GROUNDED":
                        badge_markup = f'<span class="badge-clean badge-clean-green">{ico.check_circle(11, "#16A34A")} Grounded (Score: {top_score:.3f})</span>'
                    elif status_type == "INSUFFICIENT_EVIDENCE":
                        badge_markup = f'<span class="badge-clean badge-clean-amber">{ico.alert_triangle(11, "#D97706")} Insufficient Evidence</span>'
                    elif status_type == "DOCUMENT_REQUIRED":
                        badge_markup = f'<span class="badge-clean badge-clean-blue">{ico.info(11, "#2563EB")} Document Required</span>'

                    sources_html = ""
                    if sources:
                        citations = []
                        for s_idx, s in enumerate(sources, 1):
                            src_name = s.get("source", "Document")
                            pg = s.get("page", 1)
                            cid = s.get("chunk_id", s_idx)
                            score = s.get("score", 0.0)
                            text_snip = s.get("text", "")[:180].replace("<", "&lt;").replace(">", "&gt;")
                            if len(s.get("text", "")) > 180:
                                text_snip += "..."

                            citations.append(
                                f'<div style="margin-bottom:6px">'
                                f'<div class="source-citation-line">{src_name} – Page {pg} (Chunk #{cid} · Relevance {score:.2f})</div>'
                                f'<div class="source-snippet-quote">"{text_snip}"</div>'
                                f'</div>'
                            )
                        all_cites = "".join(citations)
                        sources_html = f"""
                        <div class="card-sources-block">
                            <div class="sources-title">{ico.file_text(12, "#2563EB")} Sources:</div>
                            {all_cites}
                        </div>
                        """

                    formatted_answer = content.replace("\n", "<br>")
                    st.markdown(f"""
                    <div class="msg-wrapper">
                        <div class="avatar-circle avatar-assistant">
                            {ico.bot(16, "#FFFFFF")}
                        </div>
                        <div class="card-assistant" style="width:100%">
                            <div class="assistant-header">
                                <div class="assistant-name">
                                    <span>NEXUS Assistant</span>
                                    {badge_markup}
                                </div>
                                <div class="assistant-time">{ts}</div>
                            </div>
                            <div class="assistant-text">{formatted_answer}</div>
                            {sources_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Bottom Composer ──
        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

        query_to_process = None
        if "queued_query" in st.session_state and st.session_state["queued_query"]:
            query_to_process = st.session_state["queued_query"]
            st.session_state["queued_query"] = None

        placeholder_text = (
            "Type your question here..." if is_indexed
            else "Upload a document to start asking questions..."
        )

        chat_input = st.chat_input(placeholder=placeholder_text)
        if chat_input:
            query_to_process = chat_input.strip()

        if query_to_process:
            if not current_chat["messages"]:
                short_title = query_to_process[:28].strip()
                current_chat["title"] = f"{short_title}..." if len(query_to_process) > 28 else short_title

            curr_ts = datetime.now().strftime("%I:%M %p")

            # 1. Append User Message
            current_chat["messages"].append({
                "role": "user",
                "content": query_to_process,
                "timestamp": curr_ts
            })

            # 2. Check Document Lifecycle State
            if not is_indexed:
                # UX FIX: Distinct "DOCUMENT REQUIRED" state
                current_chat["messages"].append({
                    "role": "assistant",
                    "content": "A document is required to answer questions. Please upload a PDF/TXT document in the right panel or click 'Use Sample Document' to get started.",
                    "timestamp": datetime.now().strftime("%I:%M %p"),
                    "status_type": "DOCUMENT_REQUIRED",
                    "sources": [],
                    "top_score": 0.0
                })
            else:
                # 3. RAG Retrieval & LLM Generation
                with st.spinner("Retrieving document context and generating answer..."):
                    ret_res = pipeline.retrieve(
                        query=query_to_process,
                        top_k=settings["top_k"],
                        threshold=settings["similarity_threshold"]
                    )

                    retrieved_chunks = ret_res["chunks"]
                    is_grounded = ret_res["is_grounded"]
                    top_score = ret_res["top_score"]
                    context_str = format_rag_context(retrieved_chunks)

                    if not is_grounded:
                        # Grounded refusal
                        ans_text = INSUFFICIENT_EVIDENCE_REPLY
                        llm_ms = 0.0
                        status_type = "INSUFFICIENT_EVIDENCE"
                    else:
                        if settings["backend"] == "ollama":
                            gen_res = OllamaClient.generate(
                                model=settings["ollama_model"],
                                context=context_str,
                                question=query_to_process,
                                temperature=settings["temperature"]
                            )
                        else:
                            gen_res = GroqClient.generate(
                                api_key=settings["groq_key"],
                                model=settings["groq_model"],
                                context=context_str,
                                question=query_to_process,
                                temperature=settings["temperature"]
                            )
                        ans_text = gen_res["answer"]
                        llm_ms = gen_res["latency_ms"]
                        status_type = "GROUNDED"

                # Save inspection state
                st.session_state.latest_inspection = {
                    "query": query_to_process,
                    "context": context_str,
                    "retrieval_ms": ret_res["retrieval_latency_ms"],
                    "llm_ms": llm_ms,
                    "top_score": top_score,
                    "is_grounded": is_grounded,
                    "chunks": retrieved_chunks
                }

                # 4. Append Assistant Message
                current_chat["messages"].append({
                    "role": "assistant",
                    "content": ans_text,
                    "timestamp": datetime.now().strftime("%I:%M %p"),
                    "status_type": status_type,
                    "sources": retrieved_chunks if is_grounded else [],
                    "top_score": top_score
                })

            st.rerun()

    # ── VIEW 2: EVALUATION SUITE (5 TEST QUESTIONS) ──
    elif settings["view_mode"] == "evaluation":
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E4E9F1;border-radius:12px;padding:1.2rem;margin-bottom:1rem">
            <h3 style="font-size:1.1rem;font-weight:800;margin:0 0 4px">Assignment Evaluation Suite</h3>
            <div style="font-size:0.78rem;color:#667085">
                Execute the 5 required test questions directly against the RAG pipeline.
                Test #5 specifically tests strict anti-hallucination refusal.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not is_indexed:
            st.warning("Please upload or load the sample document first before running evaluation tests.")
            if st.button("Load Sample Document to Enable Tests", key="btn_eval_load_sample"):
                load_sample_document_action()
        else:
            for item in DEMO_QUESTIONS:
                qid = item["id"]
                qtext = item["question"]
                qtype = item["type"]
                should_refuse = item["should_refuse"]

                with st.container():
                    st.markdown(f"""
                    <div style="background:#FFFFFF;border:1px solid #E4E9F1;border-radius:8px;padding:0.9rem;margin-bottom:8px">
                        <div style="display:flex;align-items:center;justify-content:space-between">
                            <span style="font-weight:700;font-size:0.85rem">[{qid}] {qtype}</span>
                            <span style="font-size:0.72rem;color:#667085">{item['target_page']}</span>
                        </div>
                        <div style="font-size:0.84rem;margin:5px 0 8px;color:#172033">{qtext}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    c_run, c_res = st.columns([2.5, 7.5])
                    with c_run:
                        if st.button(f"▶ Run {qid}", key=f"run_test_{qid}"):
                            with st.spinner(f"Evaluating {qid}..."):
                                r = pipeline.retrieve(qtext, top_k=settings["top_k"], threshold=settings["similarity_threshold"])
                                is_grounded = r["is_grounded"]
                                top_sc = r["top_score"]

                                passed = (not is_grounded) if should_refuse else is_grounded

                                if not is_grounded:
                                    ans = INSUFFICIENT_EVIDENCE_REPLY
                                else:
                                    ctx = format_rag_context(r["chunks"])
                                    ans = OllamaClient.generate(settings["ollama_model"], ctx, qtext)["answer"]

                                st.session_state.eval_results[qid] = {
                                    "passed": passed,
                                    "score": top_sc,
                                    "chunks": len(r["chunks"]),
                                    "answer": ans,
                                    "latency_ms": r["retrieval_latency_ms"]
                                }

                    with c_res:
                        if qid in st.session_state.eval_results:
                            res_data = st.session_state.eval_results[qid]
                            p_color = "#16A34A" if res_data["passed"] else "#DC2626"
                            p_label = "PASS (Correct Refusal)" if (should_refuse and res_data["passed"]) else ("PASS" if res_data["passed"] else "FAIL")

                            st.markdown(f"""
                            <div style="background:#F8FAFC;border:1px solid #E4E9F1;border-radius:6px;padding:0.5rem 0.75rem;font-size:0.75rem">
                                <b style="color:{p_color}">{p_label}</b> · Score: <b>{res_data['score']:.3f}</b> · Retrieval: {res_data['latency_ms']}ms<br>
                                <span style="color:#667085">Answer:</span> {res_data['answer'][:140]}...
                            </div>
                            """, unsafe_allow_html=True)

    # ── VIEW 3: SETTINGS VIEW ──
    elif settings["view_mode"] == "settings":
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E4E9F1;border-radius:12px;padding:1.2rem;margin-bottom:1rem">
            <h3 style="font-size:1.1rem;font-weight:800;margin:0 0 4px">Intelligence & Inference Configuration</h3>
            <div style="font-size:0.78rem;color:#667085">Configure the local Ollama runtime, retrieval parameters, and embedding models.</div>
        </div>
        """, unsafe_allow_html=True)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### LLM Inference")
            b_choice = st.selectbox(
                "Runtime",
                ["Ollama (Local Daemon)", "Groq Cloud (Free Tier)"],
                index=0 if settings["backend"] == "ollama" else 1
            )
            settings["backend"] = "ollama" if "Ollama" in b_choice else "groq"

            if settings["backend"] == "ollama":
                models_avail = OllamaClient.get_available_models()
                settings["ollama_model"] = st.selectbox("Ollama Model", models_avail, index=0)
                if not ollama_live:
                    st.error("Ollama runtime is not reachable. Run 'ollama serve' in your terminal.")
            else:
                settings["groq_key"] = st.text_input("Groq API Key", type="password", value=settings["groq_key"])
                settings["groq_model"] = st.selectbox("Groq Model", GROQ_MODELS, index=0)

        with col_s2:
            st.markdown("#### Retrieval & Chunking")
            settings["top_k"] = st.slider("Top-K Retrieved Chunks", 1, 8, settings["top_k"])
            settings["similarity_threshold"] = st.slider("Grounding Confidence Threshold", 0.10, 0.60, settings["similarity_threshold"], 0.02)
            st.caption("Minimum cosine similarity required before generating an answer. Prevents hallucination.")

        st.markdown("---")
        if st.button("← Return to Chat", key="btn_back_chat"):
            settings["view_mode"] = "chat"
            st.rerun()

# ──────────────────────────────────────────────
# COLUMN 3: RIGHT PANEL (DOCUMENTS & CONTEXT)
# ──────────────────────────────────────────────
with col_panel:
    # ── 1. Uploaded Documents Card ──
    st.markdown(f"""
    <div class="right-panel-card">
        <div class="panel-card-header">
            <div class="panel-card-title">{ico.file_text(15, "#2563EB")} Uploaded Documents</div>
            <span class="badge-clean badge-clean-blue">{len(pipeline.documents)} File</span>
        </div>
    """, unsafe_allow_html=True)

    if pipeline.documents:
        for fname, dmeta in pipeline.documents.items():
            badge_icon = ico.badge_pdf() if fname.lower().endswith(".pdf") else ico.badge_txt()
            st.markdown(f"""
            <div class="doc-row-item">
                <div class="doc-row-left">
                    {badge_icon}
                    <div>
                        <div class="doc-row-name" title="{fname}">{fname}</div>
                        <div class="doc-row-meta">{dmeta['page_count']} pages · {dmeta['chunk_count']} chunks</div>
                    </div>
                </div>
                <div>
                    <span class="badge-clean badge-clean-green">● Indexed</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑 Unload Document", key="btn_unload_right"):
            pipeline.clear()
            st.rerun()
    else:
        st.markdown("""
        <div style="font-size:0.75rem;color:#667085;padding:0.3rem 0 0.8rem;line-height:1.5">
            No document currently indexed. Upload a PDF or TXT to enable semantic search.
        </div>
        """, unsafe_allow_html=True)

        # Upload component
        uploaded_doc = st.file_uploader(
            "Upload Document",
            type=["pdf", "txt"],
            label_visibility="collapsed",
            key="right_uploader"
        )
        if uploaded_doc is not None:
            f_bytes = uploaded_doc.read()
            f_name = uploaded_doc.name
            if f_name not in pipeline.documents:
                with st.spinner("Indexing document..."):
                    res = pipeline.ingest_document(f_bytes, f_name)
                    if res["success"]:
                        st.success(f"Indexed {res['page_count']} pages")
                        st.rerun()

        st.markdown('<div style="margin-top:6px">', unsafe_allow_html=True)
        if st.button("📥 Use Sample Document", key="btn_use_sample_right"):
            load_sample_document_action()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 2. Retrieved Context Card (Reference Design Exact Match) ──
    latest_chunks = []
    if st.session_state.latest_inspection and st.session_state.latest_inspection.get("chunks"):
        latest_chunks = st.session_state.latest_inspection["chunks"]

    st.markdown(f"""
    <div class="right-panel-card">
        <div class="panel-card-header">
            <div class="panel-card-title">{ico.layers(15, "#7C3AED")} Retrieved Context</div>
            <span class="badge-clean badge-clean-blue">{len(latest_chunks)} Passages</span>
        </div>
    """, unsafe_allow_html=True)

    if latest_chunks:
        for idx, chunk in enumerate(latest_chunks[:3], 1):
            src = chunk.get("source", "Document")
            pg = chunk.get("page", 1)
            score = chunk.get("score", 0.0)
            text_p = chunk.get("text", "")[:190].replace("<", "&lt;").replace(">", "&gt;")
            if len(chunk.get("text", "")) > 190:
                text_p += "..."

            st.markdown(f"""
            <div class="retrieved-item-box">
                <div class="retrieved-item-text">"{text_p}"</div>
                <div class="retrieved-item-footer">
                    <span>{src} – Page {pg}</span>
                    <span>Relevance: {score:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:0.74rem;color:#8B9BB4;padding:0.4rem 0.1rem;line-height:1.5">
            Submit a query to inspect live semantic passages retrieved from FAISS vectors.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 3. RAG Inspector Telemetry (Collapsible) ──
    with st.expander("Telemetry & Observability", expanded=False):
        insp = st.session_state.latest_inspection
        if insp:
            st.metric("Retrieval Latency", f"{insp['retrieval_ms']} ms")
            st.metric("LLM Latency", f"{insp['llm_ms']} ms")
            st.metric("Top Similarity", f"{insp['top_score']:.3f}")
            st.caption(f"Embedding: {pipeline.model_name}")
            st.caption(f"LLM: {active_model}")
        else:
            st.caption("No queries executed yet.")
