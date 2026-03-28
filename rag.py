import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

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

MODEL_NAME = "llama-3.1-8b-instant"
TOP_K = 4
MAX_HISTORY_TURNS = 6

# ─────────────────────────────────────────────
# PAGE SETUP
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
        background: #1e3a5f;
        color: #e8f0fe;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0 6px 60px;
        font-size: 15px;
        line-height: 1.5;
    }
    .assistant-bubble {
        background: #1a1a2e;
        color: #e0e0e0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 60px 6px 0;
        font-size: 15px;
        line-height: 1.5;
        border-left: 3px solid #4a9eff;
    }
    .source-card {
        background: #111827;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 13px;
        color: #9ca3af;
    }
    .source-tag {
        display: inline-block;
        padding: 1px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .app-header {
        padding: 10px 0 20px 0;
        border-bottom: 1px solid #2d3748;
        margin-bottom: 16px;
    }
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    .status-ready   { background: #064e3b; color: #6ee7b7; }
    .status-waiting { background: #1f2937; color: #9ca3af; }
    .mode-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .mode-single { background: #1e3a5f; color: #93c5fd; }
    .mode-multi  { background: #1a1a2e; color: #c4b5fd; }
    .mode-compare { background: #1a2e1a; color: #6ee7b7; }
    .compare-col {
        background: #111827;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 14px;
        height: 100%;
    }
    .compare-header {
        font-size: 13px;
        font-weight: 600;
        color: #93c5fd;
        margin-bottom: 8px;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 6px;
    }
    .pdf-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        margin: 2px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "chat_history": [],
    "papers": {},        # { filename: {"vectorstore": vs, "chunks": n} }
    "active_papers": [], # list of filenames currently selected for Q&A
    "mode": "single",    # "single" | "multi" | "compare"
    "compare_question": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(groq_api_key=get_api_key(), model_name=MODEL_NAME)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# PDF PROCESSING
# ─────────────────────────────────────────────
def process_pdf(uploaded_file):
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
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    # Tag each chunk with source filename
    for chunk in chunks:
        chunk.metadata["source_name"] = uploaded_file.name
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    return vectorstore, len(chunks)

# ─────────────────────────────────────────────
# ANSWER — single or multi-paper
# ─────────────────────────────────────────────
def get_answer(question, history, paper_names):
    """Retrieve from one or multiple papers, then generate a unified answer."""
    all_docs = []
    for name in paper_names:
        vs = st.session_state.papers[name]["vectorstore"]
        docs = vs.similarity_search(question, k=TOP_K)
        for d in docs:
            d.metadata["source_name"] = name
        all_docs.extend(docs)

    # Build context with paper labels
    context_parts = []
    for doc in all_docs:
        label = doc.metadata.get("source_name", "Unknown")
        context_parts.append(f"[From: {label}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    history_text = ""
    for msg in history[-(MAX_HISTORY_TURNS * 2):]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    paper_list = ", ".join(paper_names)
    system_prompt = f"""You are a precise research assistant analyzing {len(paper_names)} paper(s): {paper_list}.
Answer questions based strictly on the provided context.
- Each context chunk is labeled with its source paper in [From: ...] tags
- Be concise and factual
- Always mention which paper a piece of information comes from
- If comparing papers, structure your answer clearly with paper names as headers
- Never fabricate information"""

    user_prompt = f"""Context from paper(s):
{context}

Conversation so far:
{history_text if history_text else "(No prior conversation)"}

Question: {question}

Answer clearly. If multiple papers are involved, cite which paper each point comes from."""

    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    sources = []
    for doc in all_docs:
        sources.append({
            "text": doc.page_content[:280],
            "paper": doc.metadata.get("source_name", "Unknown"),
        })
    return response.content, sources

# ─────────────────────────────────────────────
# COMPARE — side-by-side answer per paper
# ─────────────────────────────────────────────
def get_comparison(question, paper_names):
    """Get a separate answer from each paper for side-by-side comparison."""
    results = {}
    for name in paper_names:
        vs = st.session_state.papers[name]["vectorstore"]
        docs = vs.similarity_search(question, k=3)
        context = "\n\n---\n\n".join([d.page_content for d in docs])

        system_prompt = f"""You are analyzing the paper: {name}.
Answer based strictly on this paper's content. Be concise — 3-5 sentences max."""

        user_prompt = f"""Context:
{context}

Question: {question}

Answer based only on this paper."""

        response = get_llm().invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        sources = [d.page_content[:200] for d in docs]
        results[name] = {"answer": response.content, "sources": sources}
    return results

# ─────────────────────────────────────────────
# COLOUR PALETTE FOR PAPER TAGS
# ─────────────────────────────────────────────
PAPER_COLORS = [
    ("background:#1e3a5f;color:#93c5fd", "border:1px solid #1e3a5f"),
    ("background:#1a2e1a;color:#6ee7b7", "border:1px solid #1a2e1a"),
    ("background:#2e1a2e;color:#c4b5fd", "border:1px solid #2e1a2e"),
    ("background:#2e2a1a;color:#fcd34d", "border:1px solid #2e2a1a"),
    ("background:#2e1a1a;color:#fca5a5", "border:1px solid #2e1a1a"),
]

def paper_color(name):
    papers = list(st.session_state.papers.keys())
    idx = papers.index(name) % len(PAPER_COLORS) if name in papers else 0
    return PAPER_COLORS[idx][0]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 ResearchRAG")
    st.markdown("---")

    # File uploader — allows multiple
    uploaded_files = st.file_uploader(
        "Upload research papers (up to 5)",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Process any newly uploaded files
    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.papers:
                with st.spinner(f"Processing {uf.name}..."):
                    vs, n = process_pdf(uf)
                    st.session_state.papers[uf.name] = {"vectorstore": vs, "chunks": n}
                    if uf.name not in st.session_state.active_papers:
                        st.session_state.active_papers.append(uf.name)
                st.success(f"✓ {uf.name} — {n} chunks")

    # Loaded papers list
    if st.session_state.papers:
        st.markdown("**Loaded papers**")
        to_remove = []
        for i, (name, info) in enumerate(st.session_state.papers.items()):
            col1, col2 = st.columns([5, 1])
            with col1:
                color = paper_color(name)
                active = name in st.session_state.active_papers
                check = st.checkbox(
                    f"{name[:28]}… ({info['chunks']} chunks)" if len(name) > 28 else f"{name} ({info['chunks']} chunks)",
                    value=active,
                    key=f"chk_{name}",
                )
                if check and name not in st.session_state.active_papers:
                    st.session_state.active_papers.append(name)
                elif not check and name in st.session_state.active_papers:
                    st.session_state.active_papers.remove(name)
            with col2:
                if st.button("✕", key=f"del_{name}", help="Remove paper"):
                    to_remove.append(name)

        for name in to_remove:
            del st.session_state.papers[name]
            if name in st.session_state.active_papers:
                st.session_state.active_papers.remove(name)
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")

        # Mode selector
        st.markdown("**Query mode**")
        n_active = len(st.session_state.active_papers)

        mode = st.radio(
            "mode",
            options=["single", "multi", "compare"],
            format_func=lambda x: {
                "single": "📄 Single paper",
                "multi":  "📚 All selected (merged)",
                "compare": "⚖️ Side-by-side compare",
            }[x],
            index=["single", "multi", "compare"].index(st.session_state.mode),
            label_visibility="collapsed",
        )
        st.session_state.mode = mode

        if mode == "single" and n_active > 1:
            st.caption("Single mode uses only the first checked paper.")
        if mode == "compare" and n_active < 2:
            st.warning("Select at least 2 papers to compare.")

    else:
        st.markdown("""
        <span class="status-badge status-waiting">○ No papers loaded</span>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:14px;padding:12px;background:#111827;border-radius:8px;border:1px solid #2d3748">
            <div style="font-size:12px;color:#6b7280;margin-bottom:8px;font-weight:600">MODES AVAILABLE</div>
            <div style="font-size:12px;color:#9ca3af;margin-bottom:5px">📄 <b>Single</b> — Q&A on one paper</div>
            <div style="font-size:12px;color:#9ca3af;margin-bottom:5px">📚 <b>Multi</b> — query across papers</div>
            <div style="font-size:12px;color:#9ca3af">⚖️ <b>Compare</b> — side-by-side answers</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Chat history
    if st.session_state.chat_history:
        st.markdown("**Conversation**")
        questions = [m["content"] for m in st.session_state.chat_history if m["role"] == "user"]
        for i, q in enumerate(questions):
            short = q[:40] + ("…" if len(q) > 40 else "")
            st.markdown(f"<small style='color:#9ca3af'>Q{i+1}: {short}</small>", unsafe_allow_html=True)
        st.markdown("")
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <small style='color:#6b7280'>
    <b>Stack:</b> LangChain · FAISS · MiniLM · Groq<br>
    <b>Model:</b> llama-3.1-8b-instant
    </small>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
mode = st.session_state.mode
active = st.session_state.active_papers

# Header
mode_labels = {
    "single":  ("📄 Single Paper Q&A",   "mode-single"),
    "multi":   ("📚 Multi-Paper Q&A",    "mode-multi"),
    "compare": ("⚖️ Side-by-Side Compare", "mode-compare"),
}
label, pill_class = mode_labels[mode]
st.markdown(f"""
<div class="app-header">
    <h2 style="margin:0 0 6px">{label}</h2>
    <span class="mode-pill {pill_class}">{
        "Single paper mode" if mode == "single"
        else f"{len(active)} papers merged" if mode == "multi"
        else f"Comparing {len(active)} papers"
    }</span>
    <p style="margin:4px 0 0;color:#9ca3af;font-size:14px">
        Upload papers in the sidebar · check which ones to include · pick a mode below.
    </p>
</div>
""", unsafe_allow_html=True)

# ── EMPTY STATES ──
if not st.session_state.papers:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#111827;border:1px solid #2d3748;border-radius:10px;padding:20px;text-align:center">
            <div style="font-size:28px;margin-bottom:8px">📄</div>
            <div style="font-size:14px;font-weight:600;color:#93c5fd;margin-bottom:6px">Single paper Q&A</div>
            <div style="font-size:12px;color:#6b7280">Ask anything about one paper. Every answer grounded in the document.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#111827;border:1px solid #2d3748;border-radius:10px;padding:20px;text-align:center">
            <div style="font-size:28px;margin-bottom:8px">📚</div>
            <div style="font-size:14px;font-weight:600;color:#c4b5fd;margin-bottom:6px">Multi-paper merged</div>
            <div style="font-size:12px;color:#6b7280">Query across multiple papers. Answers cite which paper each point comes from.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#111827;border:1px solid #2d3748;border-radius:10px;padding:20px;text-align:center">
            <div style="font-size:28px;margin-bottom:8px">⚖️</div>
            <div style="font-size:14px;font-weight:600;color:#6ee7b7;margin-bottom:6px">Side-by-side compare</div>
            <div style="font-size:12px;color:#6b7280">Same question to multiple papers, answers shown in columns.</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:28px 0 0;color:#6b7280">
        <p style="font-size:15px">Upload PDFs in the sidebar to unlock all three modes — up to 5 papers at once.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not active:
    st.warning("No papers selected. Check at least one paper in the sidebar.")
    st.stop()

if mode == "compare" and len(active) < 2:
    st.warning("Switch to Compare mode requires at least 2 papers checked in the sidebar.")
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
        with st.spinner("Querying each paper separately..."):
            results = get_comparison(compare_q, active)

        st.markdown(f"**Q: {compare_q}**")
        st.markdown("---")

        cols = st.columns(len(active))
        for i, name in enumerate(active):
            with cols[i]:
                color = paper_color(name)
                st.markdown(f"""
                <div class="compare-col">
                    <div class="compare-header" style="{color}; padding:4px 8px; border-radius:6px; margin-bottom:8px">
                        📄 {name[:35]}{"…" if len(name) > 35 else ""}
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(results[name]["answer"])
                with st.expander("Source chunks", expanded=False):
                    for src in results[name]["sources"]:
                        st.markdown(f"""
                        <div class="source-card">{src}{"…" if len(src)==200 else ""}</div>
                        """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # Save to chat history
        combined = "\n\n".join([
            f"**{name}:** {results[name]['answer']}" for name in active
        ])
        st.session_state.chat_history.append({"role": "user", "content": compare_q, "sources": []})
        st.session_state.chat_history.append({"role": "assistant", "content": combined, "sources": []})

# ── SINGLE / MULTI CHAT MODE ──
else:
    # Determine which papers to query
    query_papers = [active[0]] if mode == "single" else active

    # Show active paper tags
    tag_html = ""
    for name in query_papers:
        color = paper_color(name)
        short = name[:20] + ("…" if len(name) > 20 else "")
        tag_html += f'<span class="pdf-chip" style="{color}">{short}</span> '
    st.markdown(f"<div style='margin-bottom:12px'>{tag_html}</div>", unsafe_allow_html=True)

    # Chat history display
    with st.container():
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center;padding:40px 0;color:#6b7280">
                <div style="font-size:32px">💬</div>
                <p>Papers are ready! Ask your first question below.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-bubble">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                    if msg.get("sources"):
                        with st.expander(f"📎 {len(msg['sources'])} source chunks used", expanded=False):
                            for src in msg["sources"]:
                                paper = src.get("paper", "")
                                color = paper_color(paper)
                                short_paper = paper[:25] + ("…" if len(paper) > 25 else "")
                                st.markdown(f"""
                                <div class="source-card">
                                    <span class="source-tag" style="{color}">📄 {short_paper}</span><br>
                                    {src['text']}{"…" if len(src['text'])==280 else ""}
                                </div>
                                """, unsafe_allow_html=True)

    # Input
    question = st.chat_input(
        f"Ask about {query_papers[0][:30] if mode == 'single' else f'{len(query_papers)} papers'}..."
    )
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})
        with st.spinner("Searching and generating answer..."):
            answer, sources = get_answer(question, st.session_state.chat_history[:-1], query_papers)
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()
