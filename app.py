"""
RAG-Based AI Chatbot — Professional Production App
Subject: Prompt Engineering | Python | Streamlit
Cloud-ready: Groq (free) + fallback Ollama (local)
"""

import os
import time
import hashlib
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot — Prompt Engineering",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL CUSTOM CSS  (Black & White Professional)
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary:   #0a0a0a;
    --bg-secondary: #111111;
    --bg-card:      #161616;
    --bg-hover:     #1e1e1e;
    --border:       #2a2a2a;
    --border-light: #333333;
    --text-primary: #f5f5f5;
    --text-secondary:#a0a0a0;
    --text-muted:   #666666;
    --accent:       #ffffff;
    --accent-dim:   #cccccc;
    --code-bg:      #0d0d0d;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1200px; }

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

.sidebar-section {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--text-muted);
    margin: 1.5rem 0 0.6rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

.brand-header {
    display: flex; align-items: center; gap: 12px;
    padding: 0.5rem 0 1.2rem; border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
}
.brand-icon {
    width: 38px; height: 38px; background: var(--accent); border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.brand-title { font-size: 1rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.01em; line-height: 1.2; }
.brand-sub   { font-size: 0.68rem; color: var(--text-muted); font-weight: 400; letter-spacing: 0.03em; }

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-light) !important;
    border-radius: 10px !important; padding: 0.5rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-dim) !important; }

.stButton > button {
    background: var(--accent) !important; color: #000 !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.82rem !important;
    letter-spacing: 0.03em !important; padding: 0.55rem 1.2rem !important;
    width: 100% !important; transition: opacity 0.15s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

.stSelectbox > div > div {
    background: var(--bg-card) !important; border: 1px solid var(--border-light) !important;
    border-radius: 8px !important; color: var(--text-primary) !important; font-size: 0.83rem !important;
}

.stTextInput > div > div > input {
    background: var(--bg-card) !important; border: 1px solid var(--border-light) !important;
    border-radius: 8px !important; color: var(--text-primary) !important; font-size: 0.83rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-dim) !important;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.06) !important;
}

.chat-wrapper {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; overflow: hidden;
}
.chat-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 1.4rem; border-bottom: 1px solid var(--border);
    background: var(--bg-secondary);
}
.chat-topbar-title {
    font-size: 0.88rem; font-weight: 600; color: var(--text-primary);
    display: flex; align-items: center; gap: 8px;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #4ade80; display: inline-block; }
.status-dot.offline { background: #555; }
.chat-meta { font-size: 0.7rem; color: var(--text-muted); }

.msg-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 1rem 1.4rem; transition: background 0.1s;
}
.msg-row:hover { background: rgba(255,255,255,0.015); }
.msg-row.user-row { flex-direction: row-reverse; }

.avatar {
    width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 700;
}
.avatar.user-av { background: var(--accent); color: #000; font-size: 12px; }
.avatar.bot-av  { background: var(--border-light); color: var(--accent); font-size: 14px; }

.bubble {
    max-width: 78%; padding: 0.75rem 1rem; border-radius: 12px;
    font-size: 0.88rem; line-height: 1.65; color: var(--text-primary); border: 1px solid var(--border);
}
.bubble.user-bubble { background: #1a1a1a; border-color: var(--border-light); border-top-right-radius: 3px; }
.bubble.bot-bubble  { background: #0f0f0f; border-color: var(--border); border-top-left-radius: 3px; }

.msg-time { font-size: 0.65rem; color: var(--text-muted); margin-top: 4px; padding: 0 4px; }
.user-row .msg-time { text-align: right; }

.source-block {
    margin-top: 10px; padding: 0.7rem 0.9rem; background: #0a0a0a;
    border: 1px solid var(--border); border-left: 3px solid var(--accent-dim);
    border-radius: 6px; font-size: 0.75rem; color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace; line-height: 1.6;
}
.source-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: 5px;
}

.empty-chat {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 4rem 2rem; text-align: center; gap: 12px;
}
.empty-icon { font-size: 2.5rem; opacity: 0.3; }
.empty-title { font-size: 1rem; font-weight: 600; color: var(--text-secondary); }
.empty-sub { font-size: 0.8rem; color: var(--text-muted); max-width: 320px; line-height: 1.6; }

.stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin-top: 0.5rem; }
.stat-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.7rem 0.8rem; text-align: center;
}
.stat-val { font-size: 1.3rem; font-weight: 700; color: var(--text-primary); }
.stat-lbl { font-size: 0.6rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }

.stProgress > div > div { background: var(--accent) !important; }
hr { border-color: var(--border) !important; }

.stChatInput { background: var(--bg-secondary) !important; border-top: 1px solid var(--border) !important; padding: 0.8rem 1.2rem !important; }
.stChatInput textarea, .stChatInput input {
    background: var(--bg-card) !important; border: 1px solid var(--border-light) !important;
    border-radius: 10px !important; color: var(--text-primary) !important; font-size: 0.88rem !important;
}
.stChatInput textarea:focus, .stChatInput input:focus {
    border-color: var(--accent-dim) !important; box-shadow: 0 0 0 2px rgba(255,255,255,0.06) !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }

.chip {
    display: inline-block; padding: 0.2rem 0.55rem; background: #1e1e1e;
    border: 1px solid var(--border-light); border-radius: 20px;
    font-size: 0.68rem; color: var(--text-secondary); margin: 2px 3px 2px 0;
}

.api-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 0.3rem 0.7rem; border-radius: 6px; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.03em;
}
.api-badge.groq  { background: #0f1f0f; border: 1px solid #1a3a1a; color: #4ade80; }
.api-badge.ollama{ background: #1a1a0f; border: 1px solid #333310; color: #facc15; }
.api-badge.none  { background: #1f0f0f; border: 1px solid #3a1a1a; color: #f87171; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 80
TOP_K_DEFAULT   = 4

SYSTEM_PROMPT = """You are a precise document assistant. Answer ONLY using the provided context below.

Rules (follow strictly):
1. If the answer is in the context, give a clear, accurate, well-structured response.
2. If the answer is NOT in the context, respond EXACTLY: "I could not find this information in the uploaded document."
3. Never invent, speculate, or use knowledge outside the provided context.
4. Be concise and professional.

Context:
{context}

---
Question: {question}
Answer:"""


# ──────────────────────────────────────────────
# LLM BACKENDS
# ──────────────────────────────────────────────
def ask_groq(api_key: str, model: str, context: str, question: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": SYSTEM_PROMPT.format(context=context, question=question)}],
        temperature=0.2,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()

def ask_ollama(model: str, context: str, question: str) -> str:
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": SYSTEM_PROMPT.format(context=context, question=question)}],
    )
    return resp["message"]["content"].strip()

def get_ollama_models() -> list:
    try:
        import ollama
        models = ollama.list()
        return [m["name"] for m in models.get("models", [])] or ["mistral"]
    except Exception:
        return ["mistral", "llama3", "phi3", "gemma2"]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
def init_state():
    defaults = {
        "messages":      [],
        "index":         None,
        "chunks":        [],
        "embedder":      None,
        "doc_name":      None,
        "doc_hash":      None,
        "total_chunks":  0,
        "total_pages":   0,
        "ready":         False,
        "query_count":   0,
        "backend":       "groq",       # "groq" | "ollama"
        "groq_key":      "",
        "groq_model":    GROQ_MODELS[0],
        "ollama_model":  "mistral",
        "top_k":         TOP_K_DEFAULT,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ──────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────
def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def timestamp() -> str:
    return time.strftime("%H:%M")

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

def extract_text(data: bytes, filename: str):
    if filename.endswith(".pdf"):
        with fitz.open(stream=data, filetype="pdf") as doc:
            pages = len(doc)
            text  = "\n\n".join(p.get_text() for p in doc)
    else:
        text  = data.decode("utf-8", errors="replace")
        pages = max(1, len(text) // 2000)
    return text, pages

def chunk_text(text: str) -> list:
    chunks, start = [], 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def build_index(chunks: list, embedder) -> faiss.IndexFlatL2:
    vecs = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    idx  = faiss.IndexFlatL2(vecs.shape[1])
    idx.add(vecs.astype(np.float32))
    return idx

def retrieve(query: str, index, chunks: list, embedder, k: int) -> list:
    qvec = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    _, I = index.search(qvec, k)
    return [chunks[i] for i in I[0] if i < len(chunks)]


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div class="brand-icon">🤖</div>
        <div>
            <div class="brand-title">RAG Chatbot</div>
            <div class="brand-sub">Prompt Engineering · Python</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Document ──
    st.markdown('<div class="sidebar-section">Document</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed")

    if uploaded:
        raw  = uploaded.read()
        fhsh = file_hash(raw)
        if fhsh != st.session_state.doc_hash:
            with st.spinner("Processing…"):
                prog = st.progress(0, text="Extracting…")
                text, pages = extract_text(raw, uploaded.name)
                prog.progress(30, text="Chunking…")
                chunks = chunk_text(text)
                prog.progress(55, text="Loading embedder…")
                embedder = load_embedder()
                prog.progress(72, text="Embedding…")
                index = build_index(chunks, embedder)
                prog.progress(100, text="Done ✓")
                time.sleep(0.3); prog.empty()
                st.session_state.update({
                    "index": index, "chunks": chunks, "embedder": embedder,
                    "doc_name": uploaded.name, "doc_hash": fhsh,
                    "total_chunks": len(chunks), "total_pages": pages,
                    "ready": True, "messages": [], "query_count": 0,
                })
            st.success("Document indexed ✓")

    if st.session_state.ready:
        st.markdown(f"""
        <div style="font-size:0.72rem;color:#666;margin:0.5rem 0 0.3rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
             title="{st.session_state.doc_name}">📄 {st.session_state.doc_name}</div>
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-val">{st.session_state.total_pages}</div><div class="stat-lbl">Pages</div></div>
            <div class="stat-card"><div class="stat-val">{st.session_state.total_chunks}</div><div class="stat-lbl">Chunks</div></div>
            <div class="stat-card"><div class="stat-val">{st.session_state.query_count}</div><div class="stat-lbl">Queries</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── LLM Backend ──
    st.markdown('<div class="sidebar-section">LLM Backend</div>', unsafe_allow_html=True)
    backend = st.selectbox("Backend", ["Groq (Free Cloud)", "Ollama (Local)"],
                           index=0, label_visibility="collapsed")
    st.session_state.backend = "groq" if "Groq" in backend else "ollama"

    if st.session_state.backend == "groq":
        groq_key = st.text_input("Groq API Key", type="password",
                                  placeholder="gsk_…  (free at console.groq.com)",
                                  value=st.session_state.groq_key,
                                  label_visibility="collapsed")
        st.session_state.groq_key = groq_key
        groq_model = st.selectbox("Model", GROQ_MODELS,
                                   index=GROQ_MODELS.index(st.session_state.groq_model),
                                   label_visibility="collapsed")
        st.session_state.groq_model = groq_model
        st.markdown("""
        <div style="font-size:0.7rem;color:#555;margin-top:4px;line-height:1.7">
            Free key at <a href="https://console.groq.com" target="_blank"
            style="color:#888;text-decoration:none">console.groq.com</a><br>
            No credit card required ✓
        </div>""", unsafe_allow_html=True)
    else:
        ollama_models = get_ollama_models()
        ollama_model = st.selectbox("Ollama Model", ollama_models,
                                     index=0, label_visibility="collapsed")
        st.session_state.ollama_model = ollama_model

    # ── Retrieval ──
    st.markdown('<div class="sidebar-section">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K Chunks", 1, 8, st.session_state.top_k, label_visibility="collapsed")
    st.session_state.top_k = top_k
    st.markdown(f"""
    <div style="font-size:0.72rem;color:#555;margin-top:2px;line-height:1.6">
        <span class="chip">all-MiniLM-L6-v2</span> <span class="chip">FAISS FlatL2</span><br>
        Retrieving top <b style="color:#888">{top_k}</b> chunks per query
    </div>""", unsafe_allow_html=True)

    # ── Actions ──
    st.markdown('<div class="sidebar-section">Actions</div>', unsafe_allow_html=True)
    if st.button("🗑  Clear Chat"):
        st.session_state.messages    = []
        st.session_state.query_count = 0
        st.rerun()

    # ── Status ──
    st.markdown('<div class="sidebar-section">Status</div>', unsafe_allow_html=True)
    if st.session_state.backend == "groq":
        key_ok = bool(st.session_state.groq_key and st.session_state.groq_key.startswith("gsk_"))
        badge_class = "groq" if key_ok else "none"
        badge_text  = "Groq ready" if key_ok else "Key missing"
    else:
        try:
            import ollama; ollama.list(); badge_class = "ollama"; badge_text = "Ollama connected"
        except Exception:
            badge_class = "none"; badge_text = "Ollama offline"

    doc_status = "✓ Document loaded" if st.session_state.ready else "No document"
    st.markdown(f"""
    <div style="font-size:0.75rem;color:#555;line-height:2">
        <span class="api-badge {badge_class}">{badge_text}</span><br>
        📦 {doc_status}<br>
        🔑 No paid API required
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN AREA
# ──────────────────────────────────────────────
c1, c2 = st.columns([1, 11])
with c1:
    st.markdown('<div style="font-size:2.2rem;padding-top:0.1rem">🤖</div>', unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div style="padding-top:0.3rem">
        <div style="font-size:1.45rem;font-weight:800;letter-spacing:-0.03em;color:#f5f5f5">RAG Chatbot</div>
        <div style="font-size:0.78rem;color:#555;margin-top:1px">
            Retrieval-Augmented Generation · Prompt Engineering Assignment
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr style='margin:0.8rem 0 1.2rem'>", unsafe_allow_html=True)

is_ready   = st.session_state.ready
s_dot      = "status-dot" if is_ready else "status-dot offline"
s_text     = "Document indexed" if is_ready else "Awaiting document"
active_mdl = st.session_state.groq_model if st.session_state.backend == "groq" else st.session_state.ollama_model

st.markdown(f"""
<div class="chat-wrapper">
    <div class="chat-topbar">
        <div class="chat-topbar-title">
            <span class="{s_dot}"></span>
            Assistant &nbsp;·&nbsp; <span style="color:#666;font-weight:400">{active_mdl}</span>
        </div>
        <div class="chat-meta">{s_text} &nbsp;·&nbsp; {len(st.session_state.messages)} messages</div>
    </div>
""", unsafe_allow_html=True)

with st.container():
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-chat">
            <div class="empty-icon">💬</div>
            <div class="empty-title">No messages yet</div>
            <div class="empty-sub">
                Upload a document in the sidebar, then ask any question about its content.
                The assistant retrieves relevant sections and generates grounded answers.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            role, content, ts = msg["role"], msg["content"], msg.get("ts", "")
            sources = msg.get("sources", [])

            if role == "user":
                safe = content.replace("<","&lt;").replace(">","&gt;")
                st.markdown(f"""
                <div class="msg-row user-row">
                    <div class="avatar user-av">U</div>
                    <div>
                        <div class="bubble user-bubble">{safe}</div>
                        <div class="msg-time">{ts}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                src_html = ""
                if sources:
                    snips = "".join(
                        f'<div style="margin-bottom:6px"><b style="color:#888">[{i}]</b> '
                        f'{s[:220].replace("<","&lt;").replace(">","&gt;")}{"…" if len(s)>220 else ""}</div>'
                        for i, s in enumerate(sources, 1)
                    )
                    src_html = f"""
                    <div class="source-block">
                        <div class="source-label">📎 Retrieved Sources ({len(sources)} chunks)</div>
                        {snips}
                    </div>"""

                safe = content.replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
                st.markdown(f"""
                <div class="msg-row">
                    <div class="avatar bot-av">🤖</div>
                    <div style="max-width:78%">
                        <div class="bubble bot-bubble">{safe}</div>
                        {src_html}
                        <div class="msg-time">{ts}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Input ──
st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
user_input = st.chat_input(
    placeholder="Ask a question about your document…" if is_ready else "Upload a document first…",
    disabled=not is_ready,
)

# ──────────────────────────────────────────────
# QUERY PIPELINE
# ──────────────────────────────────────────────
if user_input and is_ready:
    q = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": q, "ts": timestamp()})

    with st.spinner("Thinking…"):
        sources = retrieve(q, st.session_state.index, st.session_state.chunks,
                           st.session_state.embedder, k=st.session_state.top_k)
        context = "\n\n---\n\n".join(sources)

        try:
            if st.session_state.backend == "groq":
                if not st.session_state.groq_key:
                    answer = "⚠ Please enter your Groq API key in the sidebar.\nGet a free key at https://console.groq.com"
                else:
                    answer = ask_groq(st.session_state.groq_key, st.session_state.groq_model, context, q)
            else:
                answer = ask_ollama(st.session_state.ollama_model, context, q)
        except Exception as e:
            answer = f"⚠ Error: {e}"

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "ts": timestamp(), "sources": sources,
    })
    st.session_state.query_count += 1
    st.rerun()
