"""NEXUS RAG — Document Intelligence Workspace
Production-grade Retrieval-Augmented Generation application.
"""

import time
import uuid
from datetime import datetime, timedelta
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
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236EA8FE' stroke-width='2'><polygon points='12 2 19 8 12 14 5 8'/><polygon points='12 10 19 16 12 22 5 16'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# DESIGN SYSTEM & PRODUCTION STYLESHEET
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-base:        #0B0D10;
    --bg-secondary:   #111418;
    --bg-panel:       #15191F;
    --bg-card:        #1A2027;
    --bg-hover:       #212832;
    --border-subtle:  #1E242C;
    --border-strong:  #252B33;
    --border-focus:   #3B4450;
    --text-primary:   #F5F7FA;
    --text-secondary: #9AA4B2;
    --text-muted:     #606B7B;
    --accent:         #6EA8FE;
    --accent-glow:    rgba(110, 168, 254, 0.12);
    --success:        #3DDC97;
    --success-glow:   rgba(61, 220, 151, 0.12);
    --warning:        #F5C451;
    --warning-glow:   rgba(245, 196, 81, 0.12);
    --error:          #FF6B6B;
    --error-glow:     rgba(255, 107, 107, 0.12);
    --radius-sm:      6px;
    --radius-md:      8px;
    --radius-lg:      12px;
    --radius-xl:      16px;
    --font-sans:      'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono:      'JetBrains Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}

/* Hide Streamlit Default UI Elements */
#MainMenu, footer, header, [data-testid="stToolbar"], .stDeployButton, [data-testid="stDecoration"] {
    display: none !important;
}

.main .block-container {
    padding: 1.25rem 2rem 3rem !important;
    max-width: 1140px;
}

/* ── SIDEBAR STYLING ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.2rem 0.9rem !important;
}

/* ── BRANDING ── */
.nexus-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.5rem 0.2rem 1.1rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 0.8rem;
}
.nexus-brand-icon {
    width: 36px;
    height: 36px;
    background: var(--bg-panel);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.nexus-brand-text h1 {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}
.nexus-brand-text p {
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    margin: 2px 0 0;
    text-transform: uppercase;
}

/* ── SIDEBAR SECTION HEADERS ── */
.sidebar-section-title {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.4rem 0 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── CONVERSATION LIST (CHATGPT STYLE) ── */
.chat-history-group {
    font-size: 0.62rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0.75rem 0 0.3rem 0.3rem;
}

.chat-history-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.65rem;
    margin-bottom: 2px;
    border-radius: var(--radius-sm);
    font-size: 0.78rem;
    color: var(--text-secondary);
    background: transparent;
    border: 1px solid transparent;
    transition: all 120ms ease;
    cursor: pointer;
    text-decoration: none;
}
.chat-history-item:hover {
    background: var(--bg-panel);
    color: var(--text-primary);
}
.chat-history-item.active {
    background: var(--bg-card);
    border-color: var(--border-strong);
    color: var(--text-primary);
    font-weight: 600;
}
.chat-title-truncate {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 175px;
}

/* ── DOCUMENT LIBRARY CARD ── */
.doc-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-md);
    padding: 0.65rem 0.75rem;
    margin-top: 0.4rem;
}
.doc-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.doc-stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-top: 8px;
}
.doc-stat-pill {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.35rem 0.2rem;
    text-align: center;
}
.doc-stat-val {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
    line-height: 1;
}
.doc-stat-lbl {
    font-size: 0.55rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 3px;
}

/* ── BUTTON STYLES ── */
.stButton > button {
    background: var(--bg-panel) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 0.9rem !important;
    width: 100% !important;
    transition: all 140ms ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
}
.stButton > button:hover {
    background: var(--bg-card) !important;
    border-color: var(--border-focus) !important;
    color: var(--text-primary) !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* Primary Accent Button for + New Chat */
div[data-testid="stSidebar"] div.primary-action-btn > div > button {
    background: var(--accent) !important;
    color: #0B0D10 !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(110, 168, 254, 0.25) !important;
}
div[data-testid="stSidebar"] div.primary-action-btn > div > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
}

/* ── INPUT / SELECT STYLING ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-size: 0.8rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg-panel) !important;
    border: 1.5px dashed var(--border-strong) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.4rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--border-focus) !important;
}

/* ── CHAT TOPBAR ── */
.chat-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.25rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}
.chat-topbar-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--text-primary);
}
.chat-meta-badges {
    display: flex;
    align-items: center;
    gap: 8px;
}
.nexus-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 0.2rem 0.55rem;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-grounded {
    background: var(--success-glow);
    color: var(--success);
    border: 1px solid rgba(61, 220, 151, 0.25);
}
.badge-insufficient {
    background: var(--warning-glow);
    color: var(--warning);
    border: 1px solid rgba(245, 196, 81, 0.25);
}
.badge-model {
    background: var(--bg-card);
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
}

/* ── CHAT CONTAINER ── */
.chat-viewport {
    background: var(--bg-secondary);
    border: 1px solid var(--border-strong);
    border-top: none;
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
    min-height: 460px;
    padding: 0.5rem 0 1rem;
}

/* ── MESSAGES ── */
.message-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 1.1rem 1.4rem;
    border-bottom: 1px solid var(--border-subtle);
    transition: background 120ms ease;
}
.message-row:last-child {
    border-bottom: none;
}
.message-row:hover {
    background: rgba(255, 255, 255, 0.012);
}
.message-row.user-msg {
    background: rgba(21, 25, 31, 0.5);
}

.msg-avatar {
    width: 32px;
    height: 32px;
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.avatar-bot {
    background: var(--bg-card);
    border: 1px solid var(--border-strong);
}
.avatar-user {
    background: var(--accent);
}

.msg-body {
    flex: 1;
    max-width: 92%;
}
.msg-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 5px;
}
.msg-sender {
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text-primary);
}
.msg-timestamp {
    font-size: 0.65rem;
    color: var(--text-muted);
}
.msg-content {
    font-size: 0.88rem;
    line-height: 1.68;
    color: var(--text-primary);
}

/* ── SOURCES BLOCK ── */
.sources-card {
    margin-top: 12px;
    background: var(--bg-base);
    border: 1px solid var(--border-strong);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-md);
    padding: 0.75rem 0.95rem;
}
.sources-header {
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
}
.source-snippet {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.5rem 0.65rem;
    margin-bottom: 6px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-secondary);
    line-height: 1.55;
}
.source-badge {
    color: var(--accent);
    font-weight: 600;
}

/* ── EMPTY STATE ── */
.empty-workspace {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3.5rem 2rem 2.5rem;
    text-align: center;
}
.empty-hero-icon {
    width: 60px;
    height: 60px;
    background: var(--bg-panel);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-xl);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.empty-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
}
.empty-subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
    max-width: 440px;
    line-height: 1.6;
    margin: 6px 0 2rem;
}

.starter-cards-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    width: 100%;
    max-width: 680px;
}
.starter-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-lg);
    padding: 0.9rem 1rem;
    text-align: left;
    transition: all 140ms ease;
    cursor: pointer;
}
.starter-card:hover {
    background: var(--bg-card);
    border-color: var(--border-focus);
    transform: translateY(-1px);
}
.starter-card-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 3px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.starter-card-desc {
    font-size: 0.72rem;
    color: var(--text-muted);
    line-height: 1.45;
}

/* ── COMPOSER STYLING ── */
.stChatInput {
    background: var(--bg-secondary) !important;
    border-top: 1px solid var(--border-strong) !important;
    padding: 0.8rem 1.2rem !important;
}
.stChatInput textarea, .stChatInput input {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-size: 0.85rem !important;
    font-family: var(--font-sans) !important;
}
.stChatInput textarea:focus, .stChatInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

/* ── SLIDER & PROGRESS ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
}
.stProgress > div > div {
    background: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────
def init_session_state():
    """Initializes multi-conversation state, pipeline, and config."""
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
            "show_inspector": False,
            "show_eval_suite": False,
        }

    if "latest_inspection" not in st.session_state:
        st.session_state.latest_inspection = None

init_session_state()

# Helper accessors
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

def categorize_chats() -> Dict[str, List[Dict[str, Any]]]:
    """Sorts and buckets conversations by recency."""
    now = datetime.now()
    buckets = {
        "Today": [],
        "Yesterday": [],
        "Previous 7 Days": [],
        "Older": []
    }
    search_q = st.session_state.chat_search.lower().strip()

    for cid, chat in sorted(
        st.session_state.conversations.items(),
        key=lambda item: item[1].get("created_at", now),
        reverse=True
    ):
        title = chat.get("title", "Untitled")
        if search_q and search_q not in title.lower():
            continue

        created = chat.get("created_at", now)
        delta = now.date() - created.date()

        if delta.days == 0:
            buckets["Today"].append(chat)
        elif delta.days == 1:
            buckets["Yesterday"].append(chat)
        elif delta.days <= 7:
            buckets["Previous 7 Days"].append(chat)
        else:
            buckets["Older"].append(chat)

    return buckets

# ══════════════════════════════════════════════
#  SIDEBAR: NAVIGATION & WORKSPACE DRAWER
# ══════════════════════════════════════════════
with st.sidebar:
    # ── Brand Header ──
    st.markdown(f"""
    <div class="nexus-brand">
        <div class="nexus-brand-icon">
            {ico.nexus_logo(20, "#6EA8FE")}
        </div>
        <div class="nexus-brand-text">
            <h1>NEXUS RAG</h1>
            <p>Document Intelligence Workspace</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Primary Action: + New Chat ──
    st.markdown('<div class="primary-action-btn">', unsafe_allow_html=True)
    if st.button("+ New Conversation", key="btn_new_chat"):
        create_new_chat()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Search Conversations ──
    search_input = st.text_input(
        "Search chats",
        placeholder="Search history...",
        label_visibility="collapsed",
        key="chat_search"
    )

    # ── Chronological Conversation History ──
    st.markdown(f'<div class="sidebar-section-title">{ico.message(12, "#606B7B")} Conversations</div>', unsafe_allow_html=True)

    chat_buckets = categorize_chats()
    has_any = any(chat_buckets.values())

    if not has_any:
        st.markdown('<div style="font-size:0.75rem;color:var(--text-muted);padding:0.4rem 0.2rem">No conversations match search.</div>', unsafe_allow_html=True)
    else:
        for bucket_name, chats in chat_buckets.items():
            if not chats:
                continue
            st.markdown(f'<div class="chat-history-group">{bucket_name}</div>', unsafe_allow_html=True)
            for c in chats:
                is_active = (c["id"] == active_chat_id)
                col_title, col_del = st.columns([8.5, 1.5])
                with col_title:
                    btn_label = f"{'▸ ' if is_active else ''}{c.get('title', 'Untitled')[:22]}"
                    if st.button(btn_label, key=f"sel_chat_{c['id']}"):
                        st.session_state.active_chat_id = c["id"]
                        st.rerun()
                with col_del:
                    if st.button("×", key=f"del_chat_{c['id']}", help="Delete chat"):
                        delete_chat(c["id"])

    # ── Document Library ──
    st.markdown(f'<div class="sidebar-section-title">{ico.file_text(12, "#606B7B")} Document Library</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf", "txt"],
        label_visibility="collapsed",
        help="Upload multi-page PDF or TXT to build vector index"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name

        # Check if already indexed
        if filename not in pipeline.documents:
            with st.spinner("Processing document..."):
                prog_bar = st.progress(0, text="Extracting text...")
                time.sleep(0.1)
                prog_bar.progress(35, text="Chunking with sentence boundaries...")
                time.sleep(0.1)
                prog_bar.progress(65, text="Generating SentenceTransformers embeddings...")

                result = pipeline.ingest_document(
                    data=file_bytes,
                    filename=filename,
                    chunk_size=settings["chunk_size"],
                    chunk_overlap=settings["chunk_overlap"]
                )

                if result["success"]:
                    prog_bar.progress(100, text="Indexed successfully")
                    time.sleep(0.3)
                    prog_bar.empty()
                    st.success(f"Indexed {result['page_count']} pages ({result['chunk_count']} chunks) in {result['indexing_time']}s")
                    st.rerun()
                else:
                    prog_bar.empty()
                    st.error(f"Processing failed: {result['error']}")

    # Render Document Cards if loaded
    if pipeline.documents:
        for fname, dmeta in pipeline.documents.items():
            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-card-header">
                    {ico.file_text(14, "#6EA8FE")}
                    <span title="{fname}">{fname[:24]}{'...' if len(fname) > 24 else ''}</span>
                </div>
                <div class="doc-stats-grid">
                    <div class="doc-stat-pill">
                        <div class="doc-stat-val">{dmeta['page_count']}</div>
                        <div class="doc-stat-lbl">Pages</div>
                    </div>
                    <div class="doc-stat-pill">
                        <div class="doc-stat-val">{dmeta['chunk_count']}</div>
                        <div class="doc-stat-lbl">Chunks</div>
                    </div>
                    <div class="doc-stat-pill">
                        <div class="doc-stat-val">{dmeta['indexing_time']}s</div>
                        <div class="doc-stat-lbl">Index Time</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Unload Document", key=f"unload_{fname}"):
                pipeline.clear()
                st.rerun()
    else:
        st.markdown(f"""
        <div style="font-size:0.75rem;color:var(--text-muted);padding:0.4rem 0.2rem;line-height:1.5">
            {ico.info(13, "#606B7B")} No document indexed. Upload a PDF/TXT or use the sample file.
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load Sample Document", key="btn_load_sample"):
            sample_path = "sample_docs/AI_and_ML_Overview.txt"
            try:
                with open(sample_path, "rb") as f:
                    sdata = f.read()
                res = pipeline.ingest_document(sdata, "AI_and_ML_Overview.txt")
                if res["success"]:
                    st.success("Loaded 5-page AI & ML Overview document")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to read sample document: {e}")

    # ── Model & Runtime Settings ──
    st.markdown(f'<div class="sidebar-section-title">{ico.settings(12, "#606B7B")} Intelligence Settings</div>', unsafe_allow_html=True)

    backend_opt = st.selectbox(
        "LLM Runtime",
        ["Ollama (Local Runtime)", "Groq Cloud (Free Tier)"],
        index=0 if settings["backend"] == "ollama" else 1,
        label_visibility="collapsed"
    )
    settings["backend"] = "ollama" if "Ollama" in backend_opt else "groq"

    if settings["backend"] == "ollama":
        ollama_models = OllamaClient.get_available_models()
        sel_model = st.selectbox(
            "Local Model",
            ollama_models,
            index=0 if settings["ollama_model"] not in ollama_models else ollama_models.index(settings["ollama_model"]),
            label_visibility="collapsed"
        )
        settings["ollama_model"] = sel_model
        ollama_live = OllamaClient.is_available()
        dot_color = "#3DDC97" if ollama_live else "#FF6B6B"
        lbl_status = "Ollama Active" if ollama_live else "Ollama Unreachable ('ollama serve')"
        st.markdown(f"""
        <div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px;display:flex;align-items:center;gap:6px">
            {ico.dot(7, dot_color)} <span>{lbl_status}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        groq_k = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_... (free at console.groq.com)",
            value=settings["groq_key"],
            label_visibility="collapsed"
        )
        settings["groq_key"] = groq_k
        settings["groq_model"] = st.selectbox(
            "Groq Model",
            GROQ_MODELS,
            index=GROQ_MODELS.index(settings["groq_model"]) if settings["groq_model"] in GROQ_MODELS else 0,
            label_visibility="collapsed"
        )

    # Retrieval Tuning
    st.markdown('<div style="font-size:0.72rem;font-weight:600;color:var(--text-secondary);margin:0.8rem 0 0.2rem">Top-K Passages</div>', unsafe_allow_html=True)
    settings["top_k"] = st.slider(
        "Top-K",
        min_value=1,
        max_value=8,
        value=settings["top_k"],
        label_visibility="collapsed"
    )

    st.markdown('<div style="font-size:0.72rem;font-weight:600;color:var(--text-secondary);margin:0.6rem 0 0.2rem">Grounding Confidence Threshold</div>', unsafe_allow_html=True)
    settings["similarity_threshold"] = st.slider(
        "Grounding Threshold",
        min_value=0.10,
        max_value=0.60,
        value=settings["similarity_threshold"],
        step=0.02,
        label_visibility="collapsed"
    )

    # ── Workspace Controls ──
    st.markdown(f'<div class="sidebar-section-title">{ico.activity(12, "#606B7B")} Tools & Observability</div>', unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("RAG Inspector", key="btn_toggle_insp"):
            settings["show_inspector"] = not settings["show_inspector"]
            st.rerun()
    with col_t2:
        if st.button("Demo Suite", key="btn_toggle_eval"):
            settings["show_eval_suite"] = not settings["show_eval_suite"]
            st.rerun()

    if st.button("Clear Conversation Messages", key="btn_clear_chat"):
        current_chat["messages"] = []
        st.rerun()

# ══════════════════════════════════════════════
#  MAIN APPLICATION VIEWPORT
# ══════════════════════════════════════════════

# ── Dynamic Top Bar ──
is_indexed = pipeline.is_indexed
doc_count = len(pipeline.documents)
active_model = settings["ollama_model"] if settings["backend"] == "ollama" else settings["groq_model"]

st.markdown(f"""
<div class="chat-topbar">
    <div class="chat-topbar-title">
        {ico.bot(18, "#6EA8FE")}
        <span>{current_chat.get('title', 'New Chat')}</span>
    </div>
    <div class="chat-meta-badges">
        <span class="nexus-badge {'badge-grounded' if is_indexed else 'badge-insufficient'}">
            {ico.dot(6, '#3DDC97' if is_indexed else '#F5C451')}
            {f"{doc_count} Doc Indexed" if is_indexed else "No Document"}
        </span>
        <span class="nexus-badge badge-model">
            {ico.cpu(12, '#9AA4B2')}
            {active_model}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── RAG Inspector Drawer (Optional Observability) ──
if settings["show_inspector"]:
    with st.expander("RAG Pipeline Inspector (Full Observability)", expanded=True):
        insp = st.session_state.latest_inspection
        if insp:
            col_i1, col_i2, col_i3, col_i4 = st.columns(4)
            with col_i1:
                st.metric("Retrieval Latency", f"{insp['retrieval_ms']} ms")
            with col_i2:
                st.metric("LLM Latency", f"{insp['llm_ms']} ms")
            with col_i3:
                st.metric("Top Similarity Score", f"{insp['top_score']:.4f}")
            with col_i4:
                status_txt = "Grounded" if insp["is_grounded"] else "Insufficient Evidence"
                st.metric("Grounding Status", status_txt)

            st.markdown(f"**Query**: `{insp['query']}`")
            st.markdown("**Retrieved Chunks Context**:")
            st.code(insp["context"], language="markdown")
            st.markdown(f"**System Prompt Enforcement**:")
            st.code(SYSTEM_PROMPT, language="markdown")
        else:
            st.info("Submit a question to view live embedding retrieval, cosine similarity vectors, and latency telemetry.")

# ── Evaluation & Demo Suite (Optional Grading Tool) ──
if settings["show_eval_suite"]:
    with st.expander("Assignment Evaluation Suite (5 Test Questions)", expanded=True):
        st.markdown("""
        Click any test question below to execute it against the current RAG pipeline.
        Test #5 tests refusal of unanswerable information (strict anti-hallucination requirement).
        """)
        for item in DEMO_QUESTIONS:
            col_q, col_run = st.columns([8.2, 1.8])
            with col_q:
                st.markdown(f"**[{item['id']}] {item['type']}**: {item['question']}")
                st.caption(f"Expected: {item['expected']}")
            with col_run:
                if st.button("Run Test", key=f"run_{item['id']}"):
                    # Inject as user query
                    st.session_state["queued_query"] = item["question"]
                    st.rerun()

# ── Chat Messages Viewport ──
messages = current_chat.get("messages", [])

if not messages:
    # ── EMPTY STATE ──
    st.markdown(f"""
    <div class="empty-workspace">
        <div class="empty-hero-icon">
            {ico.nexus_logo(32, "#6EA8FE")}
        </div>
        <h2 class="empty-title">NEXUS RAG Workspace</h2>
        <p class="empty-subtitle">
            Grounded document intelligence powered by SentenceTransformers embeddings and local FAISS vector search.
            Ask questions about your uploaded documents with verifiable source citations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Suggested Prompts (2x2 Grid)
    cols = st.columns(2)
    for idx, card in enumerate(SUGGESTED_PROMPTS):
        col = cols[idx % 2]
        with col:
            st.markdown(f"""
            <div class="starter-card">
                <div class="starter-card-title">{ico.zap(14, "#6EA8FE")} {card['title']}</div>
                <div class="starter-card-desc">{card['prompt']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Ask: {card['title']}", key=f"starter_btn_{idx}"):
                st.session_state["queued_query"] = card["prompt"]
                st.rerun()
else:
    # ── MESSAGE STREAM ──
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        ts = msg.get("timestamp", "")
        sources = msg.get("sources", [])
        is_grounded = msg.get("is_grounded", True)
        top_score = msg.get("top_score", 0.0)

        if role == "user":
            st.markdown(f"""
            <div class="message-row user-msg">
                <div class="msg-avatar avatar-user">
                    {ico.user(16, "#0B0D10")}
                </div>
                <div class="msg-body">
                    <div class="msg-meta">
                        <span class="msg-sender">You</span>
                        <span class="msg-timestamp">{ts}</span>
                    </div>
                    <div class="msg-content">{content}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Assistant Message with Grounding Badge & Source Citations
            grounding_badge = (
                f'<span class="nexus-badge badge-grounded">{ico.check_circle(12, "#3DDC97")} Grounded in Document ({top_score:.3f})</span>'
                if is_grounded
                else f'<span class="nexus-badge badge-insufficient">{ico.alert_triangle(12, "#F5C451")} Insufficient Evidence</span>'
            )

            sources_html = ""
            if sources:
                snips = []
                for s_idx, s in enumerate(sources, 1):
                    src_file = s.get("source", "Document")
                    pg = s.get("page", 1)
                    cid = s.get("chunk_id", s_idx)
                    score = s.get("score", 0.0)
                    text_prev = s.get("text", "")[:240].replace("<", "&lt;").replace(">", "&gt;")
                    if len(s.get("text", "")) > 240:
                        text_prev += "..."

                    snips.append(
                        f'<div class="source-snippet">'
                        f'<span class="source-badge">[{s_idx:02d}]</span> '
                        f'<b>{src_file}</b> · Page {pg} · Chunk #{cid} (Score: {score:.3f})<br>'
                        f'{text_prev}'
                        f'</div>'
                    )
                all_snips = "".join(snips)
                sources_html = f"""
                <div class="sources-card">
                    <div class="sources-header">
                        {ico.bookmark(12, "#6EA8FE")}
                        Retrieved Context Passages ({len(sources)} Chunks)
                    </div>
                    {all_snips}
                </div>
                """

            formatted_text = content.replace("\n", "<br>")
            st.markdown(f"""
            <div class="message-row">
                <div class="msg-avatar avatar-bot">
                    {ico.bot(16, "#6EA8FE")}
                </div>
                <div class="msg-body">
                    <div class="msg-meta">
                        <span class="msg-sender">NEXUS Assistant</span>
                        <span class="msg-timestamp">{ts}</span>
                        {grounding_badge}
                    </div>
                    <div class="msg-content">{formatted_text}</div>
                    {sources_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# QUERY EXECUTION PIPELINE
# ──────────────────────────────────────────────
# Check if a queued query was triggered from starter cards or evaluation suite
query_to_process = None
if "queued_query" in st.session_state and st.session_state["queued_query"]:
    query_to_process = st.session_state["queued_query"]
    st.session_state["queued_query"] = None

chat_input = st.chat_input(
    placeholder="Ask a question about your indexed documents..." if is_indexed else "Upload a document to enable querying...",
    disabled=not is_indexed
)

if chat_input:
    query_to_process = chat_input.strip()

if query_to_process:
    # Auto-generate conversation title from first question
    if not current_chat["messages"]:
        clean_title = query_to_process[:32].strip()
        current_chat["title"] = f"{clean_title}..." if len(query_to_process) > 32 else clean_title

    curr_time = datetime.now().strftime("%I:%M %p")

    # 1. Append User Message
    current_chat["messages"].append({
        "role": "user",
        "content": query_to_process,
        "timestamp": curr_time
    })

    # 2. Execute RAG Retrieval
    with st.spinner("Retrieving semantic passages & synthesizing answer..."):
        retrieval_res = pipeline.retrieve(
            query=query_to_process,
            top_k=settings["top_k"],
            threshold=settings["similarity_threshold"]
        )

        retrieved_chunks = retrieval_res["chunks"]
        is_grounded = retrieval_res["is_grounded"]
        top_score = retrieval_res["top_score"]
        context_str = format_rag_context(retrieved_chunks)

        # 3. LLM Generation
        if not is_grounded:
            # Anti-hallucination safeguard: If similarity is below threshold, refuse early
            answer_text = INSUFFICIENT_EVIDENCE_REPLY
            llm_latency = 0.0
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
            answer_text = gen_res["answer"]
            llm_latency = gen_res["latency_ms"]

    # 4. Save Inspection Data
    st.session_state.latest_inspection = {
        "query": query_to_process,
        "context": context_str,
        "retrieval_ms": retrieval_res["retrieval_latency_ms"],
        "llm_ms": llm_latency,
        "top_score": top_score,
        "is_grounded": is_grounded,
        "chunks": retrieved_chunks
    }

    # 5. Append Assistant Message
    current_chat["messages"].append({
        "role": "assistant",
        "content": answer_text,
        "timestamp": datetime.now().strftime("%I:%M %p"),
        "sources": retrieved_chunks if is_grounded else [],
        "is_grounded": is_grounded,
        "top_score": top_score
    })

    st.rerun()
