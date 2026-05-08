"""
app/main.py
Streamlit front-end for the RAG Document Assistant.

Run locally:
    streamlit run app/main.py

Environment variables must be set (see .env.example).
"""

import logging
import os
import tempfile

import streamlit as st

from app.rag.chain import RAGChain
from app.rag.embeddings import EmbeddingManager
from app.rag.memory import ConversationMemory
from app.rag.retriever import DocumentRetriever
from app.storage.s3_handler import S3Handler
from app.utils.config import get_settings
from app.utils.logger import setup_logging
from app.utils.pdf_processor import PDFProcessor

# ── Bootstrap ─────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=settings.app_name,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global font & background ─────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* ── Sidebar ──────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0f0f0f;
        border-right: 1px solid #2a2a2a;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* ── Chat bubbles ─────────────────────────────── */
    .user-bubble {
        background: #1a1a2e;
        border-left: 3px solid #4f8ef7;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.95rem;
    }
    .assistant-bubble {
        background: #0d1f0d;
        border-left: 3px solid #3ddc84;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.95rem;
    }
    .source-chip {
        display: inline-block;
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-family: 'IBM Plex Mono', monospace;
        margin: 2px 4px 2px 0;
        color: #a0a0a0;
    }
    .stat-card {
        background: #111;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        margin-bottom: 10px;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4f8ef7;
        font-family: 'IBM Plex Mono', monospace;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Main header ─────────────────────────────── */
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }
    .main-subheader {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    /* ── Input area ──────────────────────────────── */
    textarea {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    /* ── Scrollable chat window ──────────────────── */
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding-right: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session-state initialisation ─────────────────────────────────────────────

def _init_session():
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager()
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if "retriever" not in st.session_state:
        st.session_state.retriever = DocumentRetriever(
            st.session_state.embedding_manager
        )
    if "chain" not in st.session_state:
        st.session_state.chain = RAGChain(
            st.session_state.retriever, st.session_state.memory
        )
    if "s3" not in st.session_state:
        st.session_state.s3 = S3Handler()
    if "processor" not in st.session_state:
        st.session_state.processor = PDFProcessor()
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files: list[dict] = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[dict] = []


_init_session()

em: EmbeddingManager = st.session_state.embedding_manager
memory: ConversationMemory = st.session_state.memory
chain: RAGChain = st.session_state.chain
s3: S3Handler = st.session_state.s3
processor: PDFProcessor = st.session_state.processor


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📚 Document Assistant")
    st.markdown("---")

    # ── Stats ──────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-value">{em.document_count}</div>
                <div class="stat-label">Vectors</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-value">{len(st.session_state.uploaded_files)}</div>
                <div class="stat-label">Docs</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### Upload Documents")

    uploaded = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        for uf in uploaded:
            # Skip already processed files
            already = [f["filename"] for f in st.session_state.uploaded_files]
            if uf.name in already:
                continue

            with st.spinner(f"Processing {uf.name}…"):
                try:
                    file_bytes = uf.read()

                    # Upload to S3 (optional)
                    s3_key = ""
                    if s3.is_available():
                        s3_key = s3.upload(file_bytes, uf.name)

                    # Parse PDF and embed
                    docs, meta = processor.process(file_bytes, uf.name, s3_key)
                    em.add_documents(docs)

                    st.session_state.uploaded_files.append(
                        {
                            "filename": uf.name,
                            "pages": meta.num_pages,
                            "chunks": len(docs),
                            "s3_key": s3_key,
                        }
                    )
                    st.success(
                        f"✅ {uf.name} — {meta.num_pages} pages, {len(docs)} chunks"
                    )
                except Exception as exc:
                    st.error(f"❌ Failed to process {uf.name}: {exc}")

    # ── Uploaded files list ─────────────────────────
    if st.session_state.uploaded_files:
        st.markdown("### Indexed Documents")
        for f in st.session_state.uploaded_files:
            with st.expander(f"📄 {f['filename']}", expanded=False):
                st.write(f"**Pages:** {f['pages']}")
                st.write(f"**Chunks:** {f['chunks']}")
                if f["s3_key"]:
                    url = s3.get_presigned_url(f["s3_key"])
                    if url:
                        st.markdown(f"[⬇ Download from S3]({url})")

    # ── Settings ────────────────────────────────────
    st.markdown("---")
    st.markdown("### Settings")
    top_k = st.slider("Retrieval depth (top-k)", 1, 8, 4)
    st.session_state.retriever.top_k = top_k

    if st.button("🗑 Clear Conversation"):
        memory.clear()
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Reset Everything"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.caption(f"Model: `{settings.openai_chat_model}`")
    st.caption(f"Embeddings: `{settings.openai_embedding_model}`")
    if s3.is_available():
        st.caption(f"S3: `{settings.s3_bucket_name}`")
    else:
        st.caption("S3: ⚠ not configured")


# ── Main panel ────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">📚 RAG Document Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-subheader">Upload PDFs, then ask questions — powered by FAISS + OpenAI</div>',
    unsafe_allow_html=True,
)

# ── Chat history ──────────────────────────────────────────────────────────────
chat_area = st.container()
with chat_area:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                chips = "".join(
                    f'<span class="source-chip">📄 {s["filename"]} p.{s["page"]}</span>'
                    for s in msg["sources"]
                )
                st.markdown(
                    f'<div style="margin-top:6px">Sources: {chips}</div>',
                    unsafe_allow_html=True,
                )

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("---")

if not em.is_ready:
    st.info("👈 Upload at least one PDF in the sidebar to begin asking questions.")

with st.form("chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([8, 1])
    with col_input:
        question = st.text_input(
            "Your question",
            placeholder="What are the key terms in this contract?",
            label_visibility="collapsed",
            disabled=not em.is_ready,
        )
    with col_btn:
        submitted = st.form_submit_button("Send", use_container_width=True)

if submitted and question.strip():
    if not em.is_ready:
        st.warning("Please upload at least one document first.")
    else:
        # Show user message immediately
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        # Stream the answer
        answer_placeholder = st.empty()
        full_answer = ""
        sources_collected = []

        with st.spinner("Thinking…"):
            response = chain.ask(question)
            full_answer = response.answer
            sources_collected = response.sources

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": full_answer,
                "sources": sources_collected,
            }
        )
        st.rerun()
