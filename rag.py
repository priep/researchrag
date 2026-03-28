import streamlit as st
import os
import tempfile
import json
import csv
import io
import time
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

MODEL_NAME     = "llama-3.1-8b-instant"
EMBED_MODEL    = "all-MiniLM-L6-v2"
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVE = 10
TOP_K_RERANK   = 4
MAX_HISTORY    = 6
FAISS_DIR      = "/tmp/faiss_indexes"
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
        background:#1e3a5f;color:#e8f0fe;padding:12px 16px;
        border-radius:18px 18px 4px 18px;margin:6px 0 6px 60px;
        font-size:15px;line-height:1.5;
    }
    .assistant-bubble {
        background:#1a1a2e;color:#e0e0e0;padding:12px 16px;
        border-radius:18px 18px 18px 4px;margin:6px 60px 6px 0;
        font-size:15px;line-height:1.5;border-left:3px solid #4a9eff;
    }
    .source-card {
        background:#0f172a;border:1px solid #1e293b;border-radius:8px;
        padding:10px 14px;margin:6px 0;font-size:13px;color:#94a3b8;
    }
    .score-bar-wrap { background:#1e293b;border-radius:4px;height:6px;margin:6px 0 2px;overflow:hidden; }
    .score-bar-fill { height:6px;border-radius:4px; }
    .source-tag { display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;margin-bottom:4px; }
    .rank-badge {
        display:inline-block;width:20px;height:20px;border-radius:50%;
        background:#1e3a5f;color:#93c5fd;font-size:11px;font-weight:700;
        text-align:center;line-height:20px;margin-right:6px;
    }
    .app-header { padding:10px 0 20px 0;border-bottom:1px solid #2d3748;margin-bottom:16px; }
    .status-badge { display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:500; }
    .status-ready   { background:#064e3b;color:#6ee7b7; }
    .status-waiting { background:#1f2937;color:#9ca3af; }
    .mode-pill { display:inline-block;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600;margin-bottom:12px; }
    .mode-single  { background:#1e3a5f;color:#93c5fd; }
    .mode-multi   { background:#1a1a2e;color:#c4b5fd; }
    .mode-compare { background:#1a2e1a;color:#6ee7b7; }
    .compare-col  { background:#111827;border:1px solid #2d3748;border-radius:10px;padding:14px; }
    .pdf-chip     { display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;margin:2px; }
    .rerank-badge {
        display:inline-block;padding:2px 8px;border-radius:8px;font-size:11px;font-weight:600;
        background:#0f2a0f;color:#4ade80;border:1px solid #166534;margin-left:6px;
    }
    /* Benchmark tab styles */
    .bench-card {
        background:#0f172a;border:1px solid #1e293b;border-radius:10px;
        padding:16px;margin-bottom:10px;
    }
    .bench-metric {
        text-align:center;padding:16px 10px;background:#111827;
        border:1px solid #2d3748;border-radius:10px;
    }
    .bench-metric .val { font-size:28px;font-weight:600;margin-bottom:4px; }
    .bench-metric .lbl { font-size:12px;color:#6b7280; }
    .pass-row { border-left:3px solid #4ade80;padding:8px 12px;margin:4px 0;background:#0f1f0f;border-radius:0 6px 6px 0;font-size:13px; }
    .fail-row { border-left:3px solid #f87171;padding:8px 12px;margin:4px 0;background:#1f0f0f;border-radius:0 6px 6px 0;font-size:13px; }
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
    "bench_results": None,
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
    if os.path.exists(index_path):
        vs = FAISS.load_local(index_path, get_embeddings(), allow_dangerous_deserialization=True)
        n  = len(vs.docstore._dict)
        return vs, n, True
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader    = PyPDFLoader(tmp_path)
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
    vs.save_local(index_path)
    return vs, len(chunks), False

# ─────────────────────────────────────────────
# RERANKING
# ─────────────────────────────────────────────
def retrieve_and_rerank(query, paper_names):
    reranker = get_reranker()
    all_docs = []
    for name in paper_names:
        vs   = st.session_state.papers[name]["vectorstore"]
        docs = vs.similarity_search(query, k=TOP_K_RETRIEVE)
        for d in docs:
            d.metadata["source_name"] = name
        all_docs.extend(docs)
    if not all_docs:
        return []
    pairs      = [(query, doc.page_content) for doc in all_docs]
    raw_scores = get_reranker().predict(pairs)
    for doc, score in zip(all_docs, raw_scores):
        doc.metadata["rerank_score"] = float(score)
    all_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    top_docs = all_docs[:TOP_K_RERANK]
    scores   = [d.metadata["rerank_score"] for d in top_docs]
    s_min, s_max = min(scores), max(scores)
    if s_max == s_min:
        norm = [75.0] * len(scores)
    else:
        norm = [round((s - s_min) / (s_max - s_min) * 100, 1) for s in scores]
    for doc, pct in zip(top_docs, norm):
        doc.metadata["confidence"] = pct
    return top_docs

def score_color(pct):
    if pct >= 75:
        return "#4ade80", "#0f2a0f"
    elif pct >= 50:
        return "#facc15", "#2a1f0f"
    else:
        return "#f87171", "#2a0f0f"

# ─────────────────────────────────────────────
# ANSWER GENERATION
# ─────────────────────────────────────────────
def get_answer(question, history, paper_names):
    top_docs = retrieve_and_rerank(question, paper_names)
    context_parts = []
    for i, doc in enumerate(top_docs):
        label = doc.metadata.get("source_name", "Unknown")
        conf  = doc.metadata.get("confidence", 0)
        context_parts.append(f"[Rank {i+1} | Confidence {conf}% | From: {label}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)
    history_text = ""
    for msg in history[-(MAX_HISTORY * 2):]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    paper_list    = ", ".join(paper_names)
    system_prompt = f"""You are a precise research assistant analyzing: {paper_list}.
Answer based strictly on the provided context. Cite paper sources. Never fabricate."""
    user_prompt   = f"""Context (reranked by relevance):
{context}

Conversation so far:
{history_text if history_text else "(No prior conversation)"}

Question: {question}"""
    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    sources = [{
        "text":       doc.page_content[:300],
        "paper":      doc.metadata.get("source_name", "Unknown"),
        "confidence": doc.metadata.get("confidence", 0),
    } for doc in top_docs]
    return response.content, sources

def get_comparison(question, paper_names):
    results = {}
    for name in paper_names:
        docs      = retrieve_and_rerank(question, [name])
        context   = "\n\n---\n\n".join([d.page_content for d in docs])
        response  = get_llm().invoke([
            SystemMessage(content=f"You are analyzing: {name}. Answer strictly from this paper. Be concise."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ])
        results[name] = {
            "answer":  response.content,
            "sources": [{"text": d.page_content[:200], "confidence": d.metadata.get("confidence", 0)} for d in docs],
        }
    return results

# ─────────────────────────────────────────────
# ── BENCHMARKING ENGINE ──
# ─────────────────────────────────────────────
def hit_at_k(query, expected_keywords, paper_name, k=3):
    """
    Hit@K — checks if ANY of the top-K retrieved chunks
    contains at least one expected keyword.
    Returns (hit: bool, top_chunks: list, matched_keyword: str|None)
    """
    vs   = st.session_state.papers[paper_name]["vectorstore"]
    docs = vs.similarity_search(query, k=k)
    combined = " ".join([d.page_content.lower() for d in docs])
    for kw in expected_keywords:
        if kw.lower() in combined:
            return True, docs, kw
    return False, docs, None

def run_benchmark(paper_name, test_cases, k=3):
    """
    Runs the full benchmark suite against one paper.
    test_cases: list of {"question": str, "keywords": [str], "difficulty": str}
    Returns detailed results dict.
    """
    results   = []
    hits      = 0
    latencies = []

    for tc in test_cases:
        t0  = time.time()
        hit, docs, matched = hit_at_k(tc["question"], tc["keywords"], paper_name, k=k)
        lat = round((time.time() - t0) * 1000, 1)  # ms
        latencies.append(lat)
        if hit:
            hits += 1
        results.append({
            "question":   tc["question"],
            "keywords":   tc["keywords"],
            "difficulty": tc.get("difficulty", "medium"),
            "hit":        hit,
            "matched_kw": matched,
            "latency_ms": lat,
            "top_chunks": [d.page_content[:200] for d in docs],
        })

    n      = len(test_cases)
    acc    = round(hits / n * 100, 1) if n > 0 else 0
    avg_lat = round(sum(latencies) / len(latencies), 1) if latencies else 0

    # Per-difficulty breakdown
    by_diff = {}
    for r in results:
        d = r["difficulty"]
        if d not in by_diff:
            by_diff[d] = {"total": 0, "hits": 0}
        by_diff[d]["total"] += 1
        if r["hit"]:
            by_diff[d]["hits"] += 1

    return {
        "paper":       paper_name,
        "k":           k,
        "total":       n,
        "hits":        hits,
        "accuracy":    acc,
        "avg_latency": avg_lat,
        "by_difficulty": by_diff,
        "results":     results,
    }

def auto_generate_test_cases(paper_name):
    """
    Uses Groq to auto-generate 10 test Q&A pairs from the paper.
    Returns list of {"question", "keywords", "difficulty"}.
    """
    vs   = st.session_state.papers[paper_name]["vectorstore"]
    docs = vs.similarity_search("main contribution methodology results", k=6)
    sample_text = "\n\n".join([d.page_content for d in docs])[:3000]

    prompt = f"""You are creating a retrieval benchmark for this research paper excerpt.

Paper excerpt:
{sample_text}

Generate exactly 10 test questions that can be answered from this paper.
For each question, provide 2-3 short keyword phrases that MUST appear in the relevant passage.

Return ONLY valid JSON — no markdown, no explanation, just the JSON array:
[
  {{
    "question": "What is the main problem this paper addresses?",
    "keywords": ["problem", "challenge"],
    "difficulty": "easy"
  }},
  ...
]

difficulty must be one of: easy, medium, hard
easy = direct factual questions
medium = requires understanding a concept
hard = synthesis or comparison questions

Return exactly 10 items."""

    response = get_llm().invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    test_cases = json.loads(raw)
    return test_cases[:10]

def results_to_csv(bench):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Question", "Keywords", "Difficulty", "Hit", "Matched Keyword", "Latency (ms)"])
    for r in bench["results"]:
        writer.writerow([
            r["question"],
            "; ".join(r["keywords"]),
            r["difficulty"],
            "PASS" if r["hit"] else "FAIL",
            r.get("matched_kw") or "",
            r["latency_ms"],
        ])
    return buf.getvalue()

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
    idx    = papers.index(name) % len(PAPER_COLORS) if name in papers else 0
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
                st.success(f"{'Cache' if cached else 'Indexed'}: {uf.name} ({n} chunks)")

    if st.session_state.papers:
        st.markdown("**Loaded papers**")
        to_remove = []
        for name, info in st.session_state.papers.items():
            c1, c2 = st.columns([5, 1])
            with c1:
                active = name in st.session_state.active_papers
                short  = (name[:26] + "…") if len(name) > 26 else name
                chk    = st.checkbox(f"{short} ({info['chunks']})", value=active, key=f"chk_{name}")
                if chk and name not in st.session_state.active_papers:
                    st.session_state.active_papers.append(name)
                elif not chk and name in st.session_state.active_papers:
                    st.session_state.active_papers.remove(name)
            with c2:
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
        st.caption(f"Retrieve {TOP_K_RETRIEVE} → rerank → keep {TOP_K_RERANK}")
        st.markdown('<span class="rerank-badge">CrossEncoder ON</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-waiting">○ No papers loaded</span>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-top:14px;padding:12px;background:#111827;border-radius:8px;border:1px solid #2d3748">
            <div style="font-size:12px;color:#6b7280;margin-bottom:8px;font-weight:600">MODES AVAILABLE</div>
            <div style="font-size:12px;color:#9ca3af;margin-bottom:5px">📄 <b>Single</b> — Q&A on one paper</div>
            <div style="font-size:12px;color:#9ca3af;margin-bottom:5px">📚 <b>Multi</b> — query across papers</div>
            <div style="font-size:12px;color:#9ca3af;margin-bottom:5px">⚖️ <b>Compare</b> — side-by-side answers</div>
            <div style="font-size:12px;color:#9ca3af">📊 <b>Benchmark</b> — measure accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.chat_history:
        st.markdown("**Conversation**")
        for i, m in enumerate([m for m in st.session_state.chat_history if m["role"] == "user"]):
            short = m["content"][:38] + ("…" if len(m["content"]) > 38 else "")
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
# MAIN TABS
# ─────────────────────────────────────────────
tab_qa, tab_bench = st.tabs(["💬 Q&A", "📊 Benchmark"])

# ══════════════════════════════════════════════
# TAB 1 — Q&A (existing flow)
# ══════════════════════════════════════════════
with tab_qa:
    mode   = st.session_state.mode
    active = st.session_state.active_papers

    mode_labels = {
        "single":  ("📄 Single Paper Q&A",     "mode-single"),
        "multi":   ("📚 Multi-Paper Q&A",      "mode-multi"),
        "compare": ("⚖️ Side-by-Side Compare",  "mode-compare"),
    }
    label, pill_class = mode_labels[mode]
    pill_text = (
        "Single paper mode" if mode == "single"
        else f"{len(active)} papers merged" if mode == "multi"
        else f"Comparing {len(active)} papers"
    )
    st.markdown(f"""
    <div class="app-header">
        <h2 style="margin:0 0 6px">{label}
            <span class="rerank-badge" style="font-size:12px;vertical-align:middle">✦ CrossEncoder reranking</span>
        </h2>
        <span class="mode-pill {pill_class}">{pill_text}</span>
        <p style="margin:4px 0 0;color:#9ca3af;font-size:14px">
            Upload papers in the sidebar · select papers · pick a mode.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.papers:
        col1, col2, col3, col4 = st.columns(4)
        cards = [
            ("📄", "#93c5fd", "Single Q&A",     "Ask anything about one paper."),
            ("📚", "#c4b5fd", "Multi-paper",     "Query across multiple papers."),
            ("⚖️", "#6ee7b7", "Compare",         "Side-by-side answers per paper."),
            ("📊", "#fcd34d", "Benchmark",        "Measure retrieval accuracy."),
        ]
        for col, (icon, color, title, desc) in zip([col1, col2, col3, col4], cards):
            with col:
                st.markdown(f"""
                <div style="background:#111827;border:1px solid #2d3748;border-radius:10px;padding:16px;text-align:center">
                    <div style="font-size:24px;margin-bottom:6px">{icon}</div>
                    <div style="font-size:13px;font-weight:600;color:{color};margin-bottom:4px">{title}</div>
                    <div style="font-size:11px;color:#6b7280">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;padding:24px 0 0;color:#6b7280'><p>Upload PDFs in the sidebar to get started.</p></div>", unsafe_allow_html=True)
        st.stop()

    if not active:
        st.warning("No papers selected — check at least one in the sidebar.")
        st.stop()

    if mode == "compare" and len(active) < 2:
        st.warning("Compare mode needs at least 2 papers.")
        st.stop()

    # ── COMPARE ──
    if mode == "compare":
        st.markdown("#### Ask a question to compare across papers")
        compare_q = st.text_input("Question", placeholder="e.g. What methodology does each paper use?", label_visibility="collapsed")
        if st.button("⚖️ Compare", type="primary", disabled=not compare_q):
            with st.spinner("Reranking and generating per-paper answers..."):
                results = get_comparison(compare_q, active)
            st.markdown(f"**Q: {compare_q}**")
            st.markdown("---")
            cols = st.columns(len(active))
            for i, name in enumerate(active):
                with cols[i]:
                    color = paper_color(name)
                    short = (name[:30] + "…") if len(name) > 30 else name
                    st.markdown(f'<div class="compare-col"><div style="{color};padding:4px 8px;border-radius:6px;margin-bottom:8px;font-size:13px;font-weight:600">📄 {short}</div>', unsafe_allow_html=True)
                    st.markdown(results[name]["answer"])
                    with st.expander("Source chunks + confidence"):
                        for j, src in enumerate(results[name]["sources"]):
                            pct = src["confidence"]
                            fill, _ = score_color(pct)
                            st.markdown(f"""
                            <div class="source-card">
                                <span class="rank-badge">{j+1}</span>
                                <span style="font-size:12px;font-weight:600;color:{fill}">{pct}%</span>
                                <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{pct}%;background:{fill}"></div></div>
                                {src['text']}…
                            </div>""", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    # ── SINGLE / MULTI CHAT ──
    else:
        query_papers = [active[0]] if mode == "single" else active
        tag_html = "".join([
            f'<span class="pdf-chip" style="{paper_color(n)}">{(n[:20]+"…") if len(n)>20 else n}</span>'
            for n in query_papers
        ])
        st.markdown(f"<div style='margin-bottom:12px'>{tag_html}</div>", unsafe_allow_html=True)

        with st.container():
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="text-align:center;padding:40px 0;color:#6b7280">
                    <div style="font-size:32px">💬</div>
                    <p>Papers ready — ask your first question below.</p>
                    <p style="font-size:12px">Every answer uses CrossEncoder reranking for higher accuracy.</p>
                </div>""", unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="assistant-bubble">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                        if msg.get("sources"):
                            with st.expander(f"📎 {len(msg['sources'])} reranked chunks"):
                                for j, src in enumerate(msg["sources"]):
                                    pct   = src.get("confidence", 0)
                                    paper = src.get("paper", "")
                                    fill, _ = score_color(pct)
                                    color   = paper_color(paper)
                                    short_p = (paper[:25]+"…") if len(paper)>25 else paper
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
                                            <span><span class="rank-badge">{j+1}</span>
                                            <span class="source-tag" style="{color}">📄 {short_p}</span></span>
                                            <span style="font-size:12px;font-weight:600;color:{fill}">{pct}%</span>
                                        </div>
                                        <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{pct}%;background:{fill}"></div></div>
                                        <div style="margin-top:6px;font-size:13px;color:#94a3b8">{src['text']}{"…" if len(src['text'])==300 else ""}</div>
                                    </div>""", unsafe_allow_html=True)

        question = st.chat_input(f"Ask about {query_papers[0][:30] if mode=='single' else f'{len(query_papers)} papers'}...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})
            with st.spinner("Retrieving → reranking → generating..."):
                answer, sources = get_answer(question, st.session_state.chat_history[:-1], query_papers)
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
            st.rerun()

# ══════════════════════════════════════════════
# TAB 2 — BENCHMARK
# ══════════════════════════════════════════════
with tab_bench:
    st.markdown("""
    <div class="app-header">
        <h2 style="margin:0 0 6px">📊 Retrieval Benchmark</h2>
        <p style="margin:4px 0 0;color:#9ca3af;font-size:14px">
            Measure how accurately your RAG pipeline retrieves relevant chunks. Uses Hit@K metric.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.papers:
        st.info("Upload at least one PDF in the sidebar to run a benchmark.")
        st.stop()

    # ── WHAT IS HIT@K explanation ──
    with st.expander("ℹ️ How the benchmark works", expanded=False):
        st.markdown("""
**Hit@K** measures whether the correct answer chunk appears in the top K retrieved results.

- For each test question, we know what keywords **must** appear in the relevant passage
- We retrieve the top **K chunks** using FAISS similarity search  
- If **any** keyword appears in those K chunks → **PASS** ✅
- Otherwise → **FAIL** ❌

**Hit@3 = 80%** means: for 80% of questions, the answer was found in the top 3 retrieved chunks.

This is the standard metric used in production RAG evaluation (used by LlamaIndex, LangChain eval frameworks, etc.)
        """)

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Step 1 — Select paper to benchmark**")
        paper_options = list(st.session_state.papers.keys())
        bench_paper   = st.selectbox("Paper", paper_options, label_visibility="collapsed")

        st.markdown("**Step 2 — Choose K**")
        k_val = st.slider("Retrieve top K chunks", min_value=1, max_value=10, value=3,
                          help="Hit@3 is the standard. Higher K = easier to pass but less meaningful.")

        st.markdown("**Step 3 — Test questions**")
        test_mode = st.radio(
            "Test mode",
            ["🤖 Auto-generate 10 questions via Groq", "✏️ Enter my own questions"],
            label_visibility="collapsed",
        )

    with col_right:
        st.markdown("**What you'll get**")
        st.markdown("""
        <div class="bench-card">
            <div style="font-size:13px;color:#94a3b8;line-height:1.9">
                ✦ <b>Hit@K accuracy %</b> — the number for your resume<br>
                ✦ Pass/fail per question with matched keywords<br>
                ✦ Per-difficulty breakdown (easy / medium / hard)<br>
                ✦ Average retrieval latency in ms<br>
                ✦ <b>Download results as CSV</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── MANUAL QUESTIONS ──
    custom_cases = []
    if "✏️" in test_mode:
        st.markdown("**Enter your test questions** (one per row, with expected keywords)")
        st.caption("Format: one question per text area. Keywords are comma-separated words that MUST appear in the answer chunk.")

        n_manual = st.number_input("Number of questions", min_value=1, max_value=20, value=5)
        for i in range(int(n_manual)):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                q = st.text_input(f"Question {i+1}", key=f"bq_{i}", placeholder="What is the main contribution?")
            with c2:
                kws = st.text_input(f"Keywords {i+1}", key=f"bk_{i}", placeholder="contribution, system, approach")
            with c3:
                diff = st.selectbox("Diff", ["easy", "medium", "hard"], key=f"bd_{i}")
            if q and kws:
                custom_cases.append({
                    "question":   q,
                    "keywords":   [k.strip() for k in kws.split(",")],
                    "difficulty": diff,
                })

    # ── RUN BUTTON ──
    st.markdown("")
    run_col, _ = st.columns([2, 3])
    with run_col:
        run_btn = st.button("🚀 Run Benchmark", type="primary", use_container_width=True)

    if run_btn:
        if "🤖" in test_mode:
            with st.spinner("Generating test questions with Groq..."):
                try:
                    test_cases = auto_generate_test_cases(bench_paper)
                    st.success(f"Generated {len(test_cases)} test questions.")
                except Exception as e:
                    st.error(f"Failed to generate questions: {e}")
                    st.stop()
        else:
            if not custom_cases:
                st.warning("Enter at least one question with keywords.")
                st.stop()
            test_cases = custom_cases

        with st.spinner(f"Running Hit@{k_val} benchmark on {len(test_cases)} questions..."):
            bench = run_benchmark(bench_paper, test_cases, k=k_val)
            st.session_state.bench_results = bench

    # ── RESULTS ──
    if st.session_state.bench_results:
        bench = st.session_state.bench_results
        st.markdown("---")
        st.markdown("### Results")

        # Top metrics
        m1, m2, m3, m4 = st.columns(4)
        acc_color = "#4ade80" if bench["accuracy"] >= 70 else "#facc15" if bench["accuracy"] >= 50 else "#f87171"
        with m1:
            st.markdown(f"""
            <div class="bench-metric">
                <div class="val" style="color:{acc_color}">{bench['accuracy']}%</div>
                <div class="lbl">Hit@{bench['k']} Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="bench-metric">
                <div class="val" style="color:#93c5fd">{bench['hits']}/{bench['total']}</div>
                <div class="lbl">Questions Passed</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="bench-metric">
                <div class="val" style="color:#c4b5fd">{bench['avg_latency']}ms</div>
                <div class="lbl">Avg Retrieval Time</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="bench-metric">
                <div class="val" style="color:#fcd34d">{bench['k']}</div>
                <div class="lbl">K (chunks checked)</div>
            </div>""", unsafe_allow_html=True)

        # Resume-ready line
        st.markdown(f"""
        <div style="margin:16px 0;padding:14px 18px;background:#0f2a1a;border:1px solid #166534;border-radius:10px;font-size:14px;color:#4ade80">
            📋 <b>Resume bullet:</b> "Achieved <b>{bench['accuracy']}% Hit@{bench['k']}</b> retrieval accuracy 
            on {bench['total']} test questions across paper '{bench['paper']}' 
            with avg {bench['avg_latency']}ms retrieval latency."
        </div>
        """, unsafe_allow_html=True)

        # Per-difficulty breakdown
        if bench["by_difficulty"]:
            st.markdown("#### Breakdown by difficulty")
            diff_cols = st.columns(len(bench["by_difficulty"]))
            diff_colors = {"easy": "#4ade80", "medium": "#facc15", "hard": "#f87171"}
            for col, (diff, stats) in zip(diff_cols, bench["by_difficulty"].items()):
                pct = round(stats["hits"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
                c   = diff_colors.get(diff, "#94a3b8")
                with col:
                    st.markdown(f"""
                    <div class="bench-metric">
                        <div class="val" style="color:{c}">{pct}%</div>
                        <div class="lbl">{diff.capitalize()} ({stats['hits']}/{stats['total']})</div>
                    </div>""", unsafe_allow_html=True)

        # Per-question results
        st.markdown("#### Per-question results")
        for i, r in enumerate(bench["results"]):
            css   = "pass-row" if r["hit"] else "fail-row"
            icon  = "✅" if r["hit"] else "❌"
            kw_str = f" — matched: <b>{r['matched_kw']}</b>" if r.get("matched_kw") else ""
            st.markdown(f"""
            <div class="{css}">
                {icon} <b>Q{i+1}</b> [{r['difficulty']}] {r['question']}{kw_str}
                <span style="float:right;font-size:11px;color:#6b7280">{r['latency_ms']}ms</span>
            </div>""", unsafe_allow_html=True)

        # CSV download
        st.markdown("")
        csv_data = results_to_csv(bench)
        st.download_button(
            label="⬇️ Download results as CSV",
            data=csv_data,
            file_name=f"benchmark_{bench['paper'].replace('.pdf','')}_hit@{bench['k']}.csv",
            mime="text/csv",
            use_container_width=True,
        )
