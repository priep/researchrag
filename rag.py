import streamlit as st
import os
import tempfile
import numpy as np
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
def get_api_key():
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    st.error("GROQ_API_KEY not found. Add it to Streamlit secrets.")
    st.stop()

MODEL_NAME      = "llama-3.1-8b-instant"
EMBED_MODEL     = "all-MiniLM-L6-v2"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVE  = 10   # retrieve more initially
TOP_K_RERANK    = 4    # keep top N after reranking
MAX_HISTORY     = 6
FAISS_DIR       = "/tmp/faiss_indexes"

os.makedirs(FAISS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchRAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .user-bubble {
        background: #1e3a5f; color: #e8f0fe;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0 6px 60px;
        font-size: 15px; line-height: 1.5;
    }
    .assistant-bubble {
        background: #1a1a2e; color: #e0e0e0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 60px 6px 0;
        font-size: 15px; line-height: 1.5;
        border-left: 3px solid #4a9eff;
    }
    .source-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px; color: #94a3b8;
    }
    .score-bar-wrap {
        background: #1e293b;
        border-radius: 4px;
        height: 6px;
        margin: 6px 0 2px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 6px;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .score-label {
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .source-tag {
        display: inline-block;
        padding: 1px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .rank-badge {
        display: inline-block;
        width: 20px; height: 20px;
        border-radius: 50%;
        background: #1e3a5f;
        color: #93c5fd;
        font-size: 11px; font-weight: 700;
        text-align: center; line-height: 20px;
        margin-right: 6px;
    }
    .app-header {
        padding: 10px 0 20px 0;
        border-bottom: 1px solid #2d3748;
        margin-bottom: 16px;
    }
    .status-badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 12px; font-size: 12px; font-weight: 500;
    }
    .status-ready   { background: #064e3b; color: #6ee7b7; }
    .status-waiting { background: #1f2937; color: #9ca3af; }
    .mode-pill {
        display: inline-block; padding: 3px 12px;
        border-radius: 20px; font-size: 12px; font-weight: 600; margin-bottom: 12px;
    }
    .mode-single  { background: #1e3a5f;  color: #93c5fd; }
    .mode-multi   { background: #1a1a2e;  color: #c4b5fd; }
    .mode-compare { background: #1a2e1a;  color: #6ee7b7; }
    .compare-col {
        background: #111827; border: 1px solid #2d3748;
        border-radius: 10px; padding: 14px;
    }
    .compare-header {
        font-size: 13px; font-weight: 600; color: #93c5fd;
        margin-bottom: 8px; border-bottom: 1px solid #2d3748; padding-bottom: 6px;
    }
    .pdf-chip {
        display: inline-block; padding: 3px 10px;
        border-radius: 12px; font-size: 11px; margin: 2px;
    }
    .rerank-badge {
        display:inline-block; padding:2px 8px; border-radius:8px;
        font-size:11px; font-weight:600;
        background:#0f2a0f; color:#4ade80; border:1px solid #166534;
        margin-left:6px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "chat_history": [],
    "papers": {},
    "active_papers": [],
    "mode": "single",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# CACHED MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(groq_api_key=get_api_key(), model_name=MODEL_NAME)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

@st.cache_resource
def get_reranker():
    return CrossEncoder(RERANK_MODEL)

# ─────────────────────────────────────────────
# PDF PROCESSING + FAISS PERSISTENCE
# ─────────────────────────────────────────────
def get_index_path(name):
    safe = "".join(c if c.isalnum() else "_" for c in name)
    return os.path.join(FAISS_DIR, safe)

def process_pdf(uploaded_file):
    index_path = get_index_path(uploaded_file.name)

    # Load from disk if already indexed
    if os.path.exists(index_path):
        vs = FAISS.load_local(
            index_path, get_embeddings(), allow_dangerous_deserialization=True
        )
        # Count chunks from docstore
        n = len(vs.docstore._dict)
        return vs, n, True   # True = loaded from cache

    # Otherwise process fresh
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["source_name"] = uploaded_file.name

    vs = FAISS.from_documents(chunks, get_embeddings())
    vs.save_local(index_path)   # persist to disk
    return vs, len(chunks), False

# ─────────────────────────────────────────────
# RERANKING WITH CONFIDENCE SCORES
# ─────────────────────────────────────────────
def retrieve_and_rerank(query, paper_names):
    """
    1. Retrieve TOP_K_RETRIEVE candidates from each paper via FAISS
    2. Rerank all candidates with CrossEncoder
    3. Return top TOP_K_RERANK with normalised confidence scores
    """
    reranker = get_reranker()
    all_docs = []

    for name in paper_names:
        vs = st.session_state.papers[name]["vectorstore"]
        docs = vs.similarity_search(query, k=TOP_K_RETRIEVE)
        for d in docs:
            d.metadata["source_name"] = name
        all_docs.extend(docs)

    if not all_docs:
        return []

    # Score every (query, chunk) pair
    pairs = [(query, doc.page_content) for doc in all_docs]
    raw_scores = reranker.predict(pairs)

    # Attach scores
    for doc, score in zip(all_docs, raw_scores):
        doc.metadata["rerank_score"] = float(score)

    # Sort descending
    all_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    top_docs = all_docs[:TOP_K_RERANK]

    # Normalise scores to 0-100% using min-max scaling
    scores = [d.metadata["rerank_score"] for d in top_docs]
    s_min, s_max = min(scores), max(scores)
    if s_max == s_min:
        norm = [75.0] * len(scores)
    else:
        norm = [round((s - s_min) / (s_max - s_min) * 100, 1) for s in scores]
    for doc, pct in zip(top_docs, norm):
        doc.metadata["confidence"] = pct

    return top_docs

# ─────────────────────────────────────────────
# SCORE → COLOUR
# ─────────────────────────────────────────────
def score_color(pct):
    if pct >= 75:
        return "#4ade80", "#0f2a0f"   # green fill, dark bg
    elif pct >= 50:
        return "#facc15", "#2a1f0f"   # amber
    else:
        return "#f87171", "#2a0f0f"   # red

# ─────────────────────────────────────────────
# ANSWER GENERATION
# ─────────────────────────────────────────────
def get_answer(question, history, paper_names):
    top_docs = retrieve_and_rerank(question, paper_names)

    context_parts = []
    for i, doc in enumerate(top_docs):
        label = doc.metadata.get("source_name", "Unknown")
        conf  = doc.metadata.get("confidence", 0)
        context_parts.append(
            f"[Rank {i+1} | Confidence {conf}% | From: {label}]\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    history_text = ""
    for msg in history[-(MAX_HISTORY * 2):]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    paper_list = ", ".join(paper_names)
    system_prompt = f"""You are a precise research assistant analyzing: {paper_list}.
Answer based strictly on the provided context (already ranked by relevance).
- Cite which paper each point comes from
- Be concise and factual
- Never fabricate information"""

    user_prompt = f"""Context (reranked by relevance):
{context}

Conversation so far:
{history_text if history_text else "(No prior conversation)"}

Question: {question}

Answer clearly, citing paper sources."""

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    sources = []
    for doc in top_docs:
        sources.append({
            "text":       doc.page_content[:300],
            "paper":      doc.metadata.get("source_name", "Unknown"),
            "confidence": doc.metadata.get("confidence", 0),
            "rerank_score": doc.metadata.get("rerank_score", 0),
        })
    return response.content, sources

def get_comparison(question, paper_names):
    results = {}
    for name in paper_names:
        docs = retrieve_and_rerank(question, [name])
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        system_prompt = f"You are analyzing: {name}. Answer strictly from this paper. Be concise."
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on this paper."
        response = get_llm().invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        sources = [{
            "text": d.page_content[:200],
            "confidence": d.metadata.get("confidence", 0),
        } for d in docs]
        results[name] = {"answer": response.content, "sources": sources}
    return results

# ─────────────────────────────────────────────
# PAPER COLOURS
# ─────────────────────────────────────────────
PAPER_COLORS = [
    "background:#1e3a5f;color:#93c5fd",
    "background:#1a2e1a;color:#6ee7b7",
    "background:#2e1a2e;color:#c4b5fd",
    "background:#2e2a1a;color:#fcd34d",
    "background:#2e1a1a;color:#fca5a5",
]
def paper_color(name):
    papers = list(st.session_state.papers.keys())
    idx = papers.index(name) % len(PAPER_COLORS) if name in papers else 0
    return PAPER_COLORS[idx]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 ResearchRAG")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload research papers (up to 5)",
        type="pdf", accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.papers:
                with st.spinner(f"Processing {uf.name}..."):
                    vs, n, cached = process_pdf(uf)
                    st.session_state.papers[uf.name] = {"vectorstore": vs, "chunks": n}
                    if uf.name not in st.session_state.active_papers:
                        st.session_state.active_papers.append(uf.name)
                label = "✓ Loaded from cache" if cached else f"✓ Indexed & saved — {n} chunks"
                st.success(f"{uf.name}: {label}")

    if st.session_state.papers:
        st.markdown("**Loaded papers**")
        to_remove = []
        for name, info in st.session_state.papers.items():
            col1, col2 = st.columns([5, 1])
            with col1:
                active = name in st.session_state.active_papers
                short = (name[:26] + "…") if len(name) > 26 else name
                check = st.checkbox(f"{short} ({info['chunks']})", value=active, key=f"chk_{name}")
                if check and name not in st.session_state.active_papers:
                    st.session_state.active_papers.append(name)
                elif not check and name in st.session_state.active_papers:
                    st.session_state.active_papers.remove(name)
            with col2:
                if st.button("✕", key=f"del_{name}"):
                    to_remove.append(name)

        for name in to_remove:
            del st.session_state.papers[name]
            if name in st.session_state.active_papers:
                st.session_state.active_papers.remove(name)
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")
        st.markdown("**Query mode**")
        mode = st.radio(
            "mode",
            options=["single", "multi", "compare"],
            format_func=lambda x: {
                "single":  "📄 Single paper",
                "multi":   "📚 All selected (merged)",
                "compare": "⚖️ Side-by-side compare",
            }[x],
            index=["single", "multi", "compare"].index(st.session_state.mode),
            label_visibility="collapsed",
        )
        st.session_state.mode = mode

        st.markdown("---")
        st.markdown("**Retrieval settings**")
        st.caption(f"Retrieve top {TOP_K_RETRIEVE} → rerank → keep top {TOP_K_RERANK}")
        st.markdown('<span class="rerank-badge">CrossEncoder reranking ON</span>', unsafe_allow_html=True)

    else:
        st.markdown('<span class="status-badge status-waiting">○ No papers loaded</span>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-top:14px;padding:12px;background:#111827;border-radius:8px;border:1px solid #2d3748">
            <div style="font-size:12px;color:#6b7280;margin-bottom:8px;font-weight:600">MODES AVAILABLE</div>
            <div style="font-size:12px;color:#9ca3af;margin-bottom:5px">📄 <b>Single</b> — Q&A on one paper</div>
            <div style="font-size:12px;color:#9ca3af;margin-bottom:5px">📚 <b>Multi</b> — query across papers</div>
            <div style="font-size:12px;color:#9ca3af">⚖️ <b>Compare</b> — side-by-side answers</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.chat_history:
        st.markdown("**Conversation**")
        for i, m in enumerate([m for m in st.session_state.chat_history if m["role"] == "user"]):
            short = m["content"][:40] + ("…" if len(m["content"]) > 40 else "")
            st.markdown(f"<small style='color:#9ca3af'>Q{i+1}: {short}</small>", unsafe_allow_html=True)
        st.markdown("")
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <small style='color:#6b7280'>
    <b>Stack:</b> LangChain · FAISS · CrossEncoder · Groq<br>
    <b>Retrieval:</b> FAISS → CrossEncoder rerank<br>
    <b>Model:</b> llama-3.1-8b-instant
    </small>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
mode   = st.session_state.mode
active = st.session_state.active_papers

mode_labels = {
    "single":  ("📄 Single Paper Q&A",    "mode-single"),
    "multi":   ("📚 Multi-Paper Q&A",     "mode-multi"),
    "compare": ("⚖️ Side-by-Side Compare", "mode-compare"),
}
label, pill_class = mode_labels[mode]
n_active = len(active)
pill_text = (
    "Single paper mode" if mode == "single"
    else f"{n_active} papers merged" if mode == "multi"
    else f"Comparing {n_active} papers"
)
st.markdown(f"""
<div class="app-header">
    <h2 style="margin:0 0 6px">{label}
        <span class="rerank-badge" style="font-size:12px;vertical-align:middle">
            ✦ CrossEncoder reranking
        </span>
    </h2>
    <span class="mode-pill {pill_class}">{pill_text}</span>
    <p style="margin:4px 0 0;color:#9ca3af;font-size:14px">
        Upload papers in the sidebar · select papers · pick a mode.
    </p>
</div>
""", unsafe_allow_html=True)

# ── EMPTY STATE ──
if not st.session_state.papers:
    col1, col2, col3 = st.columns(3)
    cards = [
        ("📄", "#93c5fd", "Single paper Q&A",    "Ask anything about one paper. Every answer grounded in the document."),
        ("📚", "#c4b5fd", "Multi-paper merged",   "Query across multiple papers. Answers cite which paper each point comes from."),
        ("⚖️", "#6ee7b7", "Side-by-side compare", "Same question to multiple papers, answers shown in columns."),
    ]
    for col, (icon, color, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #2d3748;border-radius:10px;padding:20px;text-align:center">
                <div style="font-size:28px;margin-bottom:8px">{icon}</div>
                <div style="font-size:14px;font-weight:600;color:{color};margin-bottom:6px">{title}</div>
                <div style="font-size:12px;color:#6b7280">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:28px 0 0;color:#6b7280">
        <p style="font-size:15px">Upload PDFs in the sidebar — up to 5 papers, with CrossEncoder reranking on every query.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not active:
    st.warning("No papers selected. Check at least one paper in the sidebar.")
    st.stop()

if mode == "compare" and len(active) < 2:
    st.warning("Compare mode needs at least 2 papers checked in the sidebar.")
    st.stop()

# ── COMPARE MODE ──
if mode == "compare":
    st.markdown("#### Ask a question to compare across papers")
    compare_q = st.text_input(
        "Comparison question",
        placeholder="e.g. What methodology does each paper use?",
        label_visibility="collapsed",
    )
    if st.button("⚖️ Compare", type="primary", disabled=not compare_q):
        with st.spinner("Reranking and generating per-paper answers..."):
            results = get_comparison(compare_q, active)

        st.markdown(f"**Q: {compare_q}**")
        st.markdown("---")
        cols = st.columns(len(active))
        for i, name in enumerate(active):
            with cols[i]:
                color = paper_color(name)
                short = (name[:32] + "…") if len(name) > 32 else name
                st.markdown(f"""
                <div class="compare-col">
                    <div class="compare-header" style="{color};padding:4px 8px;border-radius:6px;margin-bottom:8px">
                        📄 {short}
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(results[name]["answer"])
                with st.expander("Source chunks + confidence", expanded=False):
                    for j, src in enumerate(results[name]["sources"]):
                        pct = src["confidence"]
                        fill, bg = score_color(pct)
                        st.markdown(f"""
                        <div class="source-card">
                            <span class="rank-badge">{j+1}</span>
                            <span class="score-label" style="color:{fill}">Confidence: {pct}%</span>
                            <div class="score-bar-wrap">
                                <div class="score-bar-fill" style="width:{pct}%;background:{fill}"></div>
                            </div>
                            {src['text']}{"…" if len(src['text'])==200 else ""}
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        combined = "\n\n".join([f"**{n}:** {results[n]['answer']}" for n in active])
        st.session_state.chat_history.append({"role": "user", "content": compare_q, "sources": []})
        st.session_state.chat_history.append({"role": "assistant", "content": combined, "sources": []})

# ── SINGLE / MULTI CHAT MODE ──
else:
    query_papers = [active[0]] if mode == "single" else active

    tag_html = ""
    for name in query_papers:
        color = paper_color(name)
        short = (name[:20] + "…") if len(name) > 20 else name
        tag_html += f'<span class="pdf-chip" style="{color}">{short}</span> '
    st.markdown(f"<div style='margin-bottom:12px'>{tag_html}</div>", unsafe_allow_html=True)

    with st.container():
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center;padding:40px 0;color:#6b7280">
                <div style="font-size:32px">💬</div>
                <p>Papers ready! Ask your first question below.</p>
                <p style="font-size:12px">Every answer uses CrossEncoder reranking for higher accuracy.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-bubble">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                    if msg.get("sources"):
                        with st.expander(f"📎 {len(msg['sources'])} reranked chunks", expanded=False):
                            for j, src in enumerate(msg["sources"]):
                                pct   = src.get("confidence", 0)
                                paper = src.get("paper", "")
                                fill, bg = score_color(pct)
                                color = paper_color(paper)
                                short_p = (paper[:25] + "…") if len(paper) > 25 else paper
                                st.markdown(f"""
                                <div class="source-card">
                                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
                                        <span><span class="rank-badge">{j+1}</span>
                                        <span class="source-tag" style="{color}">📄 {short_p}</span></span>
                                        <span style="font-size:12px;font-weight:600;color:{fill}">{pct}%</span>
                                    </div>
                                    <div class="score-bar-wrap">
                                        <div class="score-bar-fill" style="width:{pct}%;background:{fill}"></div>
                                    </div>
                                    <div style="margin-top:6px;font-size:13px;color:#94a3b8">
                                        {src['text']}{"…" if len(src['text'])==300 else ""}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

    question = st.chat_input(
        f"Ask about {query_papers[0][:30] if mode == 'single' else f'{len(query_papers)} papers'}..."
    )
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})
        with st.spinner("Retrieving → reranking → generating..."):
            answer, sources = get_answer(question, st.session_state.chat_history[:-1], query_papers)
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()
