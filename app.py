"""NEXUS RAG — Enterprise Document Intelligence Workspace
Three-column enterprise architecture inspired by the reference design.
Light enterprise palette (#F7F9FC / #FFFFFF) with deep navy sidebar (#0F2547).
"""

import time
import uuid
import html
from datetime import datetime
from typing import Dict, List, Any, Optional

import streamlit as st

# Custom modules
from components.icons import icons as ico
from rag.engine import RAGPipeline, get_relevance_label, is_summary_query
from rag.llm import OllamaClient, GroqClient, GROQ_MODELS
from rag.prompts import (
    SYSTEM_PROMPT,
    format_rag_context,
    SUGGESTED_PROMPTS,
    DEMO_QUESTIONS,
    INSUFFICIENT_EVIDENCE_REPLY
)

# ──────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS RAG — Document Intelligence Workspace",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%232563EB' stroke-width='2'><polygon points='12 2 19 8 12 14 5 8'/><polygon points='12 10 19 16 12 22 5 16'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# STYLESHEET: LIGHT ENTERPRISE UI & NAVY SIDEBAR
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-app:        #F7F9FC;
    --bg-surface:    #FFFFFF;
    --bg-sidebar:    #0F2547;
    --bg-sidebar-alt:#17335F;
    --border-color:  #E4E9F1;
    --border-dark:   #CBD5E1;
    --text-primary:  #172033;
    --text-secondary:#667085;
    --text-muted:    #8B9BB4;
    --accent:        #2563EB;
    --accent-light:  #EFF6FF;
    --accent-hover:  #1D4ED8;
    --user-bubble:   #EBF3FE;
    --success:       #16A34A;
    --warning:       #D97706;
    --error:         #DC2626;
    --radius-sm:     6px;
    --radius-md:     8px;
    --radius-lg:     12px;
    --radius-xl:     16px;
    --shadow-sm:     0 1px 3px rgba(16, 24, 40, 0.05);
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

/* ── REMOVE STREAMLIT DEFAULT CHROME & TOP WHITESPACE ── */
#MainMenu, footer, header, [data-testid="stToolbar"], .stDeployButton, [data-testid="stDecoration"] {
    display: none !important;
}

/* HIDE SIDEBAR COLLAPSE / EXPAND TOGGLE COMPLETELY */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[kind="header"] {
    display: none !important;
}

/* TIGHTEN TOP PADDING */
.main .block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1480px !important;
}

/* ── DEEP NAVY SIDEBAR (#0F2547) ── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    min-width: 260px !important;
    max-width: 275px !important;
}
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1rem 0.85rem !important;
}

.sidebar-brand-box {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.2rem 0.1rem 0.9rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 0.75rem;
}
.sidebar-brand-title {
    font-size: 1rem;
    font-weight: 800;
    color: #FFFFFF !important;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.sidebar-brand-sub {
    font-size: 0.62rem;
    font-weight: 500;
    color: #8B9BB4 !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* New Chat Primary Action Button */
.sidebar-new-btn > div > button {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    height: 40px !important;
    width: 100% !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.35) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
}
.sidebar-new-btn > div > button:hover {
    background: var(--accent-hover) !important;
}

.sidebar-nav-label {
    font-size: 0.6rem;
    font-weight: 700;
    color: #8B9BB4 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1rem 0 0.35rem 0.2rem;
}

/* Chat history items */
.sidebar-chat-btn > div > button {
    background: transparent !important;
    color: #E2E8F0 !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.77rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.4rem 0.6rem !important;
    width: 100% !important;
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

/* Promo Card in Sidebar */
.sidebar-bottom-promo {
    background: var(--bg-sidebar-alt);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-lg);
    padding: 0.75rem;
    margin-top: 1.5rem;
    text-align: center;
}
.promo-title {
    font-size: 0.74rem;
    font-weight: 700;
    color: #FFFFFF !important;
    margin-top: 5px;
}
.promo-sub {
    font-size: 0.63rem;
    color: #8B9BB4 !important;
    margin-top: 2px;
    line-height: 1.4;
}

/* ── COMPACT MAIN WORKSPACE HEADER ── */
.top-header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 0.75rem 1.15rem;
    margin-bottom: 0.85rem;
    box-shadow: var(--shadow-sm);
    min-height: 56px;
}
.header-left-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0;
    line-height: 1.2;
}
.header-left-sub {
    font-size: 0.7rem;
    color: var(--text-secondary);
    margin-top: 2px;
}
.header-right-meta {
    display: flex;
    align-items: center;
    gap: 8px;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--bg-app);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 0.25rem 0.65rem;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* ── CHAT MESSAGES ── */
.msg-wrapper {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 1rem;
}
.avatar-circle {
    width: 32px;
    height: 32px;
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

/* User Bubble */
.bubble-user {
    background: var(--user-bubble);
    border: 1px solid #D0E1FD;
    border-radius: var(--radius-lg);
    padding: 0.75rem 1rem;
    max-width: 82%;
    font-size: 0.86rem;
    color: var(--text-primary);
    line-height: 1.55;
}
.bubble-user-time {
    font-size: 0.62rem;
    color: var(--text-secondary);
    text-align: right;
    margin-top: 4px;
}

/* Assistant Card */
.card-assistant {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 0.95rem 1.15rem;
    max-width: 95%;
    width: 100%;
    box-shadow: var(--shadow-sm);
}
.assistant-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
    border-bottom: 1px solid #F1F4F9;
    padding-bottom: 5px;
}
.assistant-name {
    font-size: 0.76rem;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 6px;
}
.assistant-time {
    font-size: 0.63rem;
    color: var(--text-muted);
}
.assistant-text {
    font-size: 0.86rem;
    line-height: 1.65;
    color: var(--text-primary);
}

/* Sources block */
.card-sources-block {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px dashed var(--border-color);
    font-size: 0.73rem;
}
.sources-title {
    font-weight: 700;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 5px;
    margin-bottom: 5px;
}
.source-citation-line {
    font-family: var(--font-mono);
    color: var(--accent);
    font-weight: 600;
    font-size: 0.7rem;
    margin-bottom: 2px;
}
.source-snippet-quote {
    color: var(--text-secondary);
    font-size: 0.7rem;
    line-height: 1.45;
    background: #F8FAFC;
    padding: 0.35rem 0.55rem;
    border-radius: var(--radius-sm);
    border: 1px solid #EEF2F6;
    margin-top: 2px;
}

/* ── EMPTY STATE ── */
.empty-box {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1.8rem 1.5rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    margin-bottom: 0.85rem;
}
.empty-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
}
.empty-sub {
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin: 4px auto 1.2rem;
    max-width: 480px;
    line-height: 1.5;
}

/* Suggestion cards */
.suggest-card-btn > div > button {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.65rem 0.8rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    height: 68px !important;
    width: 100% !important;
    box-shadow: var(--shadow-sm) !important;
}
.suggest-card-btn > div > button:hover {
    border-color: var(--accent) !important;
    background: var(--accent-light) !important;
}

/* ── COMPOSER (CLEAN SINGLE BORDER, NO FLOATING ARTIFACTS) ── */
.stChatInput {
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.stChatInput > div {
    background: #FFFFFF !important;
    border: 1.5px solid var(--border-dark) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color 140ms ease !important;
}
.stChatInput > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
}
.stChatInput textarea {
    background: transparent !important;
    border: none !important;
    color: var(--text-primary) !important;
    font-size: 0.86rem !important;
    padding: 10px 14px !important;
    box-shadow: none !important;
}

/* ── RIGHT PANEL (INDEPENDENTLY SCROLLABLE) ── */
.right-panel-scroll {
    max-height: calc(100vh - 80px);
    overflow-y: auto;
    padding-right: 2px;
}
.right-panel-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 0.85rem 1rem;
    margin-bottom: 0.85rem;
    box-shadow: var(--shadow-sm);
}
.panel-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.7rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #F1F4F9;
}
.panel-card-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 6px;
}

.doc-row-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0.7rem;
    background: #F8FAFC;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    margin-bottom: 6px;
}
.doc-row-left {
    display: flex;
    align-items: center;
    gap: 7px;
    overflow: hidden;
}
.doc-row-name {
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 160px;
}
.doc-row-meta {
    font-size: 0.65rem;
    color: var(--text-secondary);
}

.retrieved-item-box {
    background: #F8FAFC;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 0.65rem 0.75rem;
    margin-bottom: 6px;
    font-size: 0.72rem;
    line-height: 1.48;
}
.retrieved-item-text {
    color: var(--text-primary);
    margin-bottom: 5px;
}
.retrieved-item-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.65rem;
    color: var(--accent);
    font-weight: 600;
    border-top: 1px dashed var(--border-color);
    padding-top: 4px;
}

/* Compact Telemetry Grid */
.telemetry-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-top: 4px;
}
.telemetry-tile {
    background: #F8FAFC;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    padding: 0.45rem 0.55rem;
    text-align: center;
}
.telemetry-val {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-primary);
    font-family: var(--font-mono);
}
.telemetry-lbl {
    font-size: 0.58rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

/* Badges */
.badge-clean {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 6px;
    border-radius: 12px;
    font-size: 0.62rem;
    font-weight: 700;
}
.badge-clean-green { background: #DCFCE7; color: #15803D; }
.badge-clean-amber { background: #FEF3C7; color: #B45309; }
.badge-clean-blue  { background: #DBEAFE; color: #1D4ED8; }

.stButton > button {
    border-radius: var(--radius-md) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SINGLE SOURCE OF TRUTH: SESSION STATE
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
# ACTION HELPERS
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

# ──────────────────────────────────────────────
# CLEAN HTML RENDERING HELPERS (NO RAW </div> LEAK)
# ──────────────────────────────────────────────
def render_assistant_html(content: str, ts: str, badge_markup: str, sources: list) -> str:
    """Builds a contiguous, single-block HTML string with no internal blank lines, preventing Markdown parser leakage."""
    # Escape special characters
    safe_content = html.escape(content).replace("\n", "<br>")

    sources_markup = ""
    if sources:
        citations = []
        for s_idx, s in enumerate(sources[:3], 1):
            src_name = html.escape(s.get("source", "Document"))
            pg = s.get("page", 1)
            cid = s.get("chunk_id", s_idx)
            score = s.get("score", 0.0)
            text_snip = html.escape(s.get("text", "")[:180])
            if len(s.get("text", "")) > 180:
                text_snip += "..."

            citations.append(
                f'<div style="margin-bottom:6px;">'
                f'<div class="source-citation-line">{src_name} – Page {pg} (Chunk #{cid} · Relevance {score:.2f})</div>'
                f'<div class="source-snippet-quote">"{text_snip}"</div>'
                f'</div>'
            )
        all_cites = "".join(citations)
        sources_markup = (
            f'<div class="card-sources-block">'
            f'<div class="sources-title">{ico.file_text(12, "#2563EB")} Sources:</div>'
            f'{all_cites}'
            f'</div>'
        )

    # Return pure contiguous HTML with ZERO blank lines
    return (
        f'<div class="msg-wrapper">'
        f'<div class="avatar-circle avatar-assistant">{ico.bot(16, "#FFFFFF")}</div>'
        f'<div class="card-assistant">'
        f'<div class="assistant-header">'
        f'<div class="assistant-name"><span>NEXUS Assistant</span>{badge_markup}</div>'
        f'<div class="assistant-time">{ts}</div>'
        f'</div>'
        f'<div class="assistant-text">{safe_content}</div>'
        f'{sources_markup}'
        f'</div>'
        f'</div>'
    )

def render_user_html(content: str, ts: str) -> str:
    safe_content = html.escape(content).replace("\n", "<br>")
    return (
        f'<div class="msg-wrapper">'
        f'<div class="avatar-circle avatar-user">{ico.user(16, "#2563EB")}</div>'
        f'<div class="bubble-user">'
        f'<div>{safe_content}</div>'
        f'<div class="bubble-user-time">{ts}</div>'
        f'</div>'
        f'</div>'
    )

# ══════════════════════════════════════════════
#  COLUMN 1: DEEP NAVY SIDEBAR (#0F2547)
# ══════════════════════════════════════════════
with st.sidebar:
    # ── Brand Header ──
    st.markdown(f"""
    <div class="sidebar-brand-box">
        <div>{ico.nexus_logo(22, "#6EA8FE")}</div>
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
    st.markdown('<div class="sidebar-nav-label">Chat History</div>', unsafe_allow_html=True)

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
            if st.button(f"{icon_char} {title[:20]}", key=f"chat_{cid}"):
                st.session_state.active_chat_id = cid
                st.session_state.settings["view_mode"] = "chat"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col_c2:
            if st.button("×", key=f"del_{cid}", help="Delete chat"):
                delete_chat(cid)

    # ── Navigation Items ──
    st.markdown('<div class="sidebar-nav-label">Workspace</div>', unsafe_allow_html=True)

    if st.button("💬 Chat Workspace", key="nav_chat"):
        st.session_state.settings["view_mode"] = "chat"
        st.rerun()

    if st.button("🧪 Evaluation Suite (5 Tests)", key="nav_eval"):
        st.session_state.settings["view_mode"] = "evaluation"
        st.rerun()

    if st.button("⚙️ Settings & Inference", key="nav_settings"):
        st.session_state.settings["view_mode"] = "settings"
        st.rerun()

    # ── Bottom Promo Card ──
    st.markdown(f"""
    <div class="sidebar-bottom-promo">
        <div>{ico.nexus_logo(24, "#6EA8FE")}</div>
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
    # ── Single Source of Truth for Document State ──
    is_indexed = pipeline.is_indexed
    ollama_live = OllamaClient.is_available()
    active_model = settings["ollama_model"] if settings["backend"] == "ollama" else settings["groq_model"]

    status_dot_color = "#16A34A" if ollama_live else "#DC2626"
    status_label = f"Ollama · {active_model}" if ollama_live else "Ollama Offline"

    doc_status_dot = "#16A34A" if is_indexed else "#D97706"
    doc_status_text = "Document Ready" if is_indexed else "No Document"

    # ── Compact Top Header ──
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
                <span>{status_label}</span>
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
                <div style="margin-bottom:0.4rem">{ico.nexus_logo(32, "#2563EB")}</div>
                <h2 class="empty-title">NEXUS RAG Workspace</h2>
                <div class="empty-sub">
                    Upload a PDF or TXT document to explore its contents with precision retrieval,
                    grounded answers, and verifiable source citations.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Suggestion cards (2x2 grid, auto-submits upon click)
            st.markdown('<div style="font-size:0.72rem;font-weight:700;color:var(--text-secondary);text-transform:uppercase;margin:0.2rem 0 0.4rem;letter-spacing:0.06em">Suggested Prompts</div>', unsafe_allow_html=True)

            sg_cols = st.columns(2)
            for idx, card in enumerate(SUGGESTED_PROMPTS):
                sg_col = sg_cols[idx % 2]
                with sg_col:
                    st.markdown('<div class="suggest-card-btn">', unsafe_allow_html=True)
                    btn_text = f"✦ {card['title']}\n{card['prompt'][:52]}..."
                    if st.button(btn_text, key=f"sg_btn_{idx}"):
                        st.session_state["queued_query"] = card["prompt"]
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            if not is_indexed:
                st.markdown('<div style="margin-top:0.8rem;text-align:center">', unsafe_allow_html=True)
                if st.button("📥 Use Sample Document (5-Page AI Overview)", key="btn_sample_empty"):
                    load_sample_document_action()
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            # ── Message Stream (100% Valid Clean HTML, Zero </div> Leakage) ──
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                ts = msg.get("timestamp", "")
                sources = msg.get("sources", [])
                status_type = msg.get("status_type", "GROUNDED")
                top_score = msg.get("top_score", 0.0)

                if role == "user":
                    user_markup = render_user_html(content, ts)
                    st.markdown(user_markup, unsafe_allow_html=True)
                else:
                    badge_markup = ""
                    if status_type == "GROUNDED":
                        strength_lbl, strength_color = get_relevance_label(top_score)
                        badge_markup = f'<span class="badge-clean badge-clean-green">{ico.check_circle(11, "#16A34A")} Grounded ({top_score:.2f} · {strength_lbl})</span>'
                    elif status_type == "INSUFFICIENT_EVIDENCE":
                        badge_markup = f'<span class="badge-clean badge-clean-amber">{ico.alert_triangle(11, "#D97706")} Insufficient Evidence</span>'
                    elif status_type == "DOCUMENT_REQUIRED":
                        badge_markup = f'<span class="badge-clean badge-clean-blue">{ico.info(11, "#2563EB")} Document Required</span>'

                    assistant_markup = render_assistant_html(content, ts, badge_markup, sources)
                    st.markdown(assistant_markup, unsafe_allow_html=True)

        # ── Bottom Composer ──
        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

        query_to_process = None
        if "queued_query" in st.session_state and st.session_state["queued_query"]:
            query_to_process = st.session_state["queued_query"]
            st.session_state["queued_query"] = None

        placeholder_text = (
            "Ask a question about your document..." if is_indexed
            else "Upload a document to start asking questions..."
        )

        chat_input = st.chat_input(placeholder=placeholder_text)
        if chat_input:
            query_to_process = chat_input.strip()

        if query_to_process:
            if not current_chat["messages"]:
                short_title = query_to_process[:26].strip()
                current_chat["title"] = f"{short_title}..." if len(query_to_process) > 26 else short_title

            curr_ts = datetime.now().strftime("%I:%M %p")

            # 1. Append User Message
            current_chat["messages"].append({
                "role": "user",
                "content": query_to_process,
                "timestamp": curr_ts
            })

            # 2. Strict Document Lifecycle Check (Single Source of Truth)
            if not pipeline.is_indexed:
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
                    is_summary = ret_res.get("is_summary", False)
                    context_str = format_rag_context(retrieved_chunks)

                    llm_ms = None  # None if LLM skipped

                    if not is_grounded:
                        # Grounded refusal
                        ans_text = INSUFFICIENT_EVIDENCE_REPLY
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

                # Save inspection state (Do not show fake 0.0 ms if LLM was skipped!)
                st.session_state.latest_inspection = {
                    "query": query_to_process,
                    "context": context_str,
                    "retrieval_ms": ret_res["retrieval_latency_ms"],
                    "llm_ms": llm_ms,  # None if skipped
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

    # ── VIEW 2: EVALUATION SUITE (5 TESTS) ──
    elif settings["view_mode"] == "evaluation":
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E4E9F1;border-radius:12px;padding:1rem 1.15rem;margin-bottom:0.85rem">
            <h3 style="font-size:1.05rem;font-weight:800;margin:0 0 3px">Assignment Evaluation Suite</h3>
            <div style="font-size:0.76rem;color:#667085">
                Execute the 5 required test questions directly against the RAG pipeline.
                Test #5 specifically validates strict anti-hallucination refusal.
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
                    <div style="background:#FFFFFF;border:1px solid #E4E9F1;border-radius:8px;padding:0.75rem 0.85rem;margin-bottom:6px">
                        <div style="display:flex;align-items:center;justify-content:space-between">
                            <span style="font-weight:700;font-size:0.82rem">[{qid}] {qtype}</span>
                            <span style="font-size:0.7rem;color:#667085">{item['target_page']}</span>
                        </div>
                        <div style="font-size:0.82rem;margin:4px 0 6px;color:#172033">{qtext}</div>
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
                            <div style="background:#F8FAFC;border:1px solid #E4E9F1;border-radius:6px;padding:0.45rem 0.65rem;font-size:0.73rem">
                                <b style="color:{p_color}">{p_label}</b> · Score: <b>{res_data['score']:.3f}</b> · Retrieval: {res_data['latency_ms']}ms<br>
                                <span style="color:#667085">Answer:</span> {res_data['answer'][:130]}...
                            </div>
                            """, unsafe_allow_html=True)

    # ── VIEW 3: SETTINGS VIEW ──
    elif settings["view_mode"] == "settings":
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E4E9F1;border-radius:12px;padding:1rem 1.15rem;margin-bottom:0.85rem">
            <h3 style="font-size:1.05rem;font-weight:800;margin:0 0 3px">Intelligence & Inference Configuration</h3>
            <div style="font-size:0.76rem;color:#667085">Configure the local Ollama runtime, retrieval parameters, and embedding models.</div>
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
    st.markdown('<div class="right-panel-scroll">', unsafe_allow_html=True)

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
        <div style="font-size:0.74rem;color:#667085;padding:0.2rem 0 0.6rem;line-height:1.45">
            No document currently indexed. Upload a PDF or TXT to enable semantic search.
        </div>
        """, unsafe_allow_html=True)

        uploaded_doc = st.file_uploader(
            "Upload Document",
            type=["pdf", "txt"],
            label_visibility="collapsed",
            key="right_uploader"
        )
        if uploaded_doc is not None:
            f_bytes = uploaded_doc.read()
            f_name = uploaded_doc.name
            if f_name not in pipeline.documents and len(f_bytes) > 0:
                with st.spinner("Indexing document..."):
                    res = pipeline.ingest_document(f_bytes, f_name)
                    if res["success"]:
                        st.success(f"Indexed {res['page_count']} pages")
                        st.rerun()

        st.markdown('<div style="margin-top:4px">', unsafe_allow_html=True)
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
            strength_lbl, strength_color = get_relevance_label(score)
            text_p = html.escape(chunk.get("text", "")[:180])
            if len(chunk.get("text", "")) > 180:
                text_p += "..."

            st.markdown(f"""
            <div class="retrieved-item-box">
                <div class="retrieved-item-text">"{text_p}"</div>
                <div class="retrieved-item-footer">
                    <span>{src} – Page {pg}</span>
                    <span style="color:{strength_color}">{score:.2f} ({strength_lbl})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:0.73rem;color:#8B9BB4;padding:0.3rem 0.1rem;line-height:1.45">
            Submit a query to inspect live semantic passages retrieved from FAISS vectors.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 3. Compact Professional Telemetry ──
    insp = st.session_state.latest_inspection
    retrieval_txt = f"{insp['retrieval_ms']:.0f} ms" if insp else "—"
    llm_txt = f"{insp['llm_ms']/1000:.2f} s" if (insp and insp.get("llm_ms") is not None) else "Not invoked"
    top_score_txt = f"{insp['top_score']:.2f}" if insp else "—"

    st.markdown(f"""
    <div class="right-panel-card">
        <div class="panel-card-header">
            <div class="panel-card-title">{ico.cpu(15, "#2563EB")} Telemetry</div>
            <span class="badge-clean badge-clean-blue">{active_model}</span>
        </div>
        <div class="telemetry-grid">
            <div class="telemetry-tile">
                <div class="telemetry-val">{retrieval_txt}</div>
                <div class="telemetry-lbl">Retrieval</div>
            </div>
            <div class="telemetry-tile">
                <div class="telemetry-val">{llm_txt}</div>
                <div class="telemetry-lbl">LLM Inference</div>
            </div>
        </div>
        <div class="telemetry-grid" style="margin-top:6px">
            <div class="telemetry-tile">
                <div class="telemetry-val">{top_score_txt}</div>
                <div class="telemetry-lbl">Top Match</div>
            </div>
            <div class="telemetry-tile">
                <div class="telemetry-val">all-MiniLM</div>
                <div class="telemetry-lbl">Embedder</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
