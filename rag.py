import streamlit as st
import os
import tempfile
import json
import csv
import io
import time
import re
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
    /* ── Chat bubbles ── */
    .user-bubble {
        background:#1e3a5f;color:#e8f0fe;padding:12px 16px;
        border-radius:18px 18px 4px 18px;margin:6px 0 6px 60px;
        font-size:15px;line-height:1.5;position:relative;
    }
    .assistant-bubble {
        background:#1a1a2e;color:#e0e0e0;padding:12px 16px;
        border-radius:18px 18px 18px 4px;margin:6px 60px 6px 0;
        font-size:15px;line-height:1.5;border-left:3px solid #4a9eff;position:relative;
    }
    .msg-time {
        font-size:10px;color:#6b7280;margin-top:4px;text-align:right;
    }
    .copy-btn {
        position:absolute;top:8px;right:10px;
        background:transparent;border:1px solid #2d3748;border-radius:5px;
        color:#6b7280;font-size:10px;padding:2px 7px;cursor:pointer;
    }
    .copy-btn:hover { background:#1e293b;color:#94a3b8; }
    /* ── Citation refs ── */
    .cite-ref {
        display:inline-block;background:#1e3a5f;color:#93c5fd;
        border-radius:4px;font-size:10px;font-weight:700;
        padding:0px 5px;margin:0 2px;vertical-align:super;cursor:pointer;
    }
    /* ── Source cards ── */
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
    /* ── Layout ── */
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
    /* ── Benchmark ── */
    .bench-card { background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px;margin-bottom:10px; }
    .bench-metric { text-align:center;padding:16px 10px;background:#111827;border:1px solid #2d3748;border-radius:10px; }
    .bench-metric .val { font-size:28px;font-weight:600;margin-bottom:4px; }
    .bench-metric .lbl { font-size:12px;color:#6b7280; }
    .pass-row { border-left:3px solid #4ade80;padding:8px 12px;margin:4px 0;background:#0f1f0f;border-radius:0 6px 6px 0;font-size:13px; }
    .fail-row { border-left:3px solid #f87171;padding:8px 12px;margin:4px 0;background:#1f0f0f;border-radius:0 6px 6px 0;font-size:13px; }
    /* ── Diagram tab ── */
    .diagram-card { background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px;margin-bottom:12px; }
    .diagram-section { font-size:13px;font-weight:600;color:#93c5fd;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em; }
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
    "diagram_data": {},
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
# PDF PROCESSING
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
    docs = PyPDFLoader(tmp_path).load()
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80, separators=["\n\n", "\n", ".", " "])
    chunks   = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["source_name"] = uploaded_file.name
    vs = FAISS.from_documents(chunks, get_embeddings())
    vs.save_local(index_path)
    return vs, len(chunks), False

# ─────────────────────────────────────────────
# RERANKING
# ─────────────────────────────────────────────
def retrieve_and_rerank(query, paper_names):
    all_docs = []
    for name in paper_names:
        vs   = st.session_state.papers[name]["vectorstore"]
        docs = vs.similarity_search(query, k=TOP_K_RETRIEVE)
        for d in docs:
            d.metadata["source_name"] = name
        all_docs.extend(docs)
    if not all_docs:
        return []
    pairs  = [(query, d.page_content) for d in all_docs]
    scores = get_reranker().predict(pairs)
    for d, s in zip(all_docs, scores):
        d.metadata["rerank_score"] = float(s)
    all_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    top  = all_docs[:TOP_K_RERANK]
    vals = [d.metadata["rerank_score"] for d in top]
    lo, hi = min(vals), max(vals)
    norm = [round((s - lo) / (hi - lo) * 100, 1) if hi != lo else 75.0 for s in vals]
    for d, pct in zip(top, norm):
        d.metadata["confidence"] = pct
    return top

def score_color(pct):
    if pct >= 75:  return "#4ade80", "#0f2a0f"
    if pct >= 50:  return "#facc15", "#2a1f0f"
    return "#f87171", "#2a0f0f"

# ─────────────────────────────────────────────
# ANSWER WITH CITATIONS
# ─────────────────────────────────────────────
def get_answer(question, history, paper_names):
    top_docs = retrieve_and_rerank(question, paper_names)
    context_parts = []
    for i, doc in enumerate(top_docs):
        label = doc.metadata.get("source_name", "Unknown")
        conf  = doc.metadata.get("confidence", 0)
        context_parts.append(f"[SOURCE {i+1} | {conf}% | {label}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    history_text = "".join([
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}\n"
        for m in history[-(MAX_HISTORY * 2):]
    ])

    system_prompt = f"""You are a precise research assistant. Answer based strictly on the provided context.
When you use information from a source, cite it inline using [1], [2], [3], [4] corresponding to SOURCE 1, 2, 3, 4.
Example: "The system uses transformer models [1] which were validated experimentally [2]."
Be concise. Never fabricate. Always cite."""

    user_prompt = f"""Context:
{context}

Prior conversation:
{history_text if history_text else "(none)"}

Question: {question}

Answer with inline citations like [1], [2] etc."""

    response = get_llm().invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    sources  = [{
        "text":       doc.page_content[:300],
        "paper":      doc.metadata.get("source_name", "Unknown"),
        "confidence": doc.metadata.get("confidence", 0),
    } for doc in top_docs]
    return response.content, sources

def get_comparison(question, paper_names):
    results = {}
    for name in paper_names:
        docs     = retrieve_and_rerank(question, [name])
        context  = "\n\n---\n\n".join([d.page_content for d in docs])
        response = get_llm().invoke([
            SystemMessage(content=f"Analyze only: {name}. Be concise, 3-5 sentences."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ])
        results[name] = {
            "answer":  response.content,
            "sources": [{"text": d.page_content[:200], "confidence": d.metadata.get("confidence", 0)} for d in docs],
        }
    return results

# ─────────────────────────────────────────────
# RENDER CITATIONS  [1] → styled superscript
# ─────────────────────────────────────────────
def render_citations(text, sources):
    """Replace [N] in text with styled citation badges."""
    def replace(m):
        n = int(m.group(1))
        if 1 <= n <= len(sources):
            paper = sources[n-1].get("paper", "")[:20]
            return f'<span class="cite-ref" title="{paper}">[{n}]</span>'
        return m.group(0)
    return re.sub(r'\[(\d+)\]', replace, text)

# ─────────────────────────────────────────────
# PAPER ARCHITECTURE EXTRACTION (Phase 5)
# ─────────────────────────────────────────────
def extract_paper_architecture(paper_name):
    """Use Groq to extract architecture components and generate Mermaid diagram."""
    vs   = st.session_state.papers[paper_name]["vectorstore"]
    docs = vs.similarity_search(
        "system architecture pipeline methodology components modules flow diagram", k=8
    )
    text = "\n\n".join([d.page_content for d in docs])[:4000]

    prompt = f"""You are extracting the system architecture from a research paper to create a Mermaid flowchart.

Paper content:
{text}

Tasks:
1. Identify the main system components / modules / pipeline stages (4-10 nodes)
2. Identify the data flow between them
3. Identify inputs (what goes in) and outputs (what comes out)

Return ONLY valid JSON — no markdown, no explanation:
{{
  "title": "Short paper title",
  "input": "What the system takes as input",
  "output": "What the system produces",
  "components": [
    {{"id": "A", "label": "Component Name", "description": "1 sentence what it does", "type": "process"}},
    ...
  ],
  "edges": [
    {{"from": "A", "to": "B", "label": "optional short label"}},
    ...
  ],
  "key_tech": ["tech1", "tech2", "tech3"],
  "problem": "One sentence: what problem does this paper solve?",
  "contribution": "One sentence: what is the main contribution?"
}}

node types: input, process, model, output, decision
Keep labels short (2-4 words max). Keep descriptions under 15 words."""

    response = get_llm().invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

def build_mermaid(arch):
    """Convert extracted architecture JSON to Mermaid flowchart syntax."""
    lines = ["flowchart TD"]

    # Style map for node types
    shape = {
        "input":    lambda id, lbl: f'    {id}(["{lbl}"])',
        "output":   lambda id, lbl: f'    {id}(["{lbl}"])',
        "model":    lambda id, lbl: f'    {id}[/"{lbl}"/]',
        "decision": lambda id, lbl: f'    {id}{{"{lbl}"}}',
        "process":  lambda id, lbl: f'    {id}["{lbl}"]',
    }

    for comp in arch.get("components", []):
        fn  = shape.get(comp.get("type", "process"), shape["process"])
        lines.append(fn(comp["id"], comp["label"]))

    lines.append("")
    for edge in arch.get("edges", []):
        lbl = f'|"{edge["label"]}"|' if edge.get("label") else ""
        lines.append(f'    {edge["from"]} -->{lbl} {edge["to"]}')

    lines.append("")
    lines.append("    classDef inputNode  fill:#1e3a5f,stroke:#93c5fd,color:#e8f0fe")
    lines.append("    classDef outputNode fill:#1a2e1a,stroke:#6ee7b7,color:#d1fae5")
    lines.append("    classDef modelNode  fill:#2e1a2e,stroke:#c4b5fd,color:#f3e8ff")
    lines.append("    classDef procNode   fill:#1a1a2e,stroke:#4a9eff,color:#e0e0e0")

    for comp in arch.get("components", []):
        t = comp.get("type", "process")
        if t == "input":
            lines.append(f'    class {comp["id"]} inputNode')
        elif t == "output":
            lines.append(f'    class {comp["id"]} outputNode')
        elif t == "model":
            lines.append(f'    class {comp["id"]} modelNode')
        else:
            lines.append(f'    class {comp["id"]} procNode')

    return "\n".join(lines)

# ─────────────────────────────────────────────
# BENCHMARK ENGINE (Phase 4)
# ─────────────────────────────────────────────
def hit_at_k(query, keywords, paper_name, k=3):
    vs   = st.session_state.papers[paper_name]["vectorstore"]
    docs = vs.similarity_search(query, k=k)
    combined = " ".join([d.page_content.lower() for d in docs])
    for kw in keywords:
        if kw.lower() in combined:
            return True, docs, kw
    return False, docs, None

def run_benchmark(paper_name, test_cases, k=3):
    results, hits, lats = [], 0, []
    for tc in test_cases:
        t0 = time.time()
        hit, docs, matched = hit_at_k(tc["question"], tc["keywords"], paper_name, k=k)
        lat = round((time.time() - t0) * 1000, 1)
        lats.append(lat)
        if hit:
            hits += 1
        results.append({
            "question": tc["question"], "keywords": tc["keywords"],
            "difficulty": tc.get("difficulty", "medium"),
            "hit": hit, "matched_kw": matched, "latency_ms": lat,
            "top_chunks": [d.page_content[:200] for d in docs],
        })
    n   = len(test_cases)
    acc = round(hits / n * 100, 1) if n > 0 else 0
    avg = round(sum(lats) / len(lats), 1) if lats else 0
    by_diff = {}
    for r in results:
        d = r["difficulty"]
        if d not in by_diff:
            by_diff[d] = {"total": 0, "hits": 0}
        by_diff[d]["total"] += 1
        if r["hit"]:
            by_diff[d]["hits"] += 1
    return {"paper": paper_name, "k": k, "total": n, "hits": hits,
            "accuracy": acc, "avg_latency": avg, "by_difficulty": by_diff, "results": results}

def auto_generate_test_cases(paper_name):
    vs   = st.session_state.papers[paper_name]["vectorstore"]
    docs = vs.similarity_search("main contribution methodology results", k=6)
    text = "\n\n".join([d.page_content for d in docs])[:3000]
    prompt = f"""Generate exactly 10 retrieval test questions for this paper excerpt.
For each, provide 2-3 keywords that MUST appear in the relevant passage.

Paper:
{text}

Return ONLY valid JSON array:
[
  {{"question": "...", "keywords": ["kw1", "kw2"], "difficulty": "easy"}},
  ...
]
difficulty: easy | medium | hard. Return exactly 10 items."""
    response = get_llm().invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())[:10]

def results_to_csv(bench):
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["Question", "Keywords", "Difficulty", "Hit", "Matched KW", "Latency (ms)"])
    for r in bench["results"]:
        w.writerow([r["question"], "; ".join(r["keywords"]), r["difficulty"],
                    "PASS" if r["hit"] else "FAIL", r.get("matched_kw") or "", r["latency_ms"]])
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
        "Upload research papers (up to 5)", type="pdf",
        accept_multiple_files=True, label_visibility="collapsed",
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
        mode = st.radio("mode", ["single", "multi", "compare"],
            format_func=lambda x: {"single": "📄 Single paper", "multi": "📚 All selected", "compare": "⚖️ Compare"}[x],
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
            <div style="font-size:12px;color:#9ca3af;margin-bottom:4px">📄 Single · 📚 Multi · ⚖️ Compare</div>
            <div style="font-size:12px;color:#9ca3af;margin-bottom:4px">📊 Benchmark · 🏗 Architecture</div>
        </div>""", unsafe_allow_html=True)

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
    </small>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab_qa, tab_arch, tab_bench = st.tabs(["💬 Q&A", "🏗 Architecture", "📊 Benchmark"])

# ══════════════════════════════════════════════
# TAB 1 — Q&A
# ══════════════════════════════════════════════
with tab_qa:
    mode   = st.session_state.mode
    active = st.session_state.active_papers

    mode_labels = {
        "single":  ("📄 Single Paper Q&A",    "mode-single"),
        "multi":   ("📚 Multi-Paper Q&A",     "mode-multi"),
        "compare": ("⚖️ Side-by-Side Compare", "mode-compare"),
    }
    label, pill_class = mode_labels[mode]
    pill_text = ("Single paper" if mode == "single" else f"{len(active)} papers merged" if mode == "multi" else f"Comparing {len(active)}")
    st.markdown(f"""
    <div class="app-header">
        <h2 style="margin:0 0 6px">{label}
            <span class="rerank-badge" style="font-size:12px;vertical-align:middle">✦ CrossEncoder reranking</span>
        </h2>
        <span class="mode-pill {pill_class}">{pill_text}</span>
        <p style="margin:4px 0 0;color:#9ca3af;font-size:14px">Upload papers · select · pick mode · ask anything.</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.papers:
        col1, col2, col3, col4, col5 = st.columns(5)
        cards = [
            ("📄","#93c5fd","Single Q&A","One paper, grounded answers"),
            ("📚","#c4b5fd","Multi-paper","Query across papers, cited"),
            ("⚖️","#6ee7b7","Compare","Side-by-side per paper"),
            ("📊","#fcd34d","Benchmark","Hit@K accuracy metric"),
            ("🏗","#fb923c","Architecture","Auto diagram generation"),
        ]
        for col,(icon,color,title,desc) in zip([col1,col2,col3,col4,col5], cards):
            with col:
                st.markdown(f"""
                <div style="background:#111827;border:1px solid #2d3748;border-radius:10px;padding:14px;text-align:center">
                    <div style="font-size:22px;margin-bottom:6px">{icon}</div>
                    <div style="font-size:12px;font-weight:600;color:{color};margin-bottom:4px">{title}</div>
                    <div style="font-size:11px;color:#6b7280">{desc}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;padding:20px 0 0;color:#6b7280'><p>Upload PDFs in the sidebar to get started.</p></div>", unsafe_allow_html=True)
        st.stop()

    if not active:
        st.warning("No papers selected — check at least one in the sidebar.")
        st.stop()
    if mode == "compare" and len(active) < 2:
        st.warning("Compare mode needs at least 2 papers.")
        st.stop()

    if mode == "compare":
        st.markdown("#### Ask a question to compare across papers")
        compare_q = st.text_input("Question", placeholder="e.g. What methodology does each paper use?", label_visibility="collapsed")
        if st.button("⚖️ Compare", type="primary", disabled=not compare_q):
            with st.spinner("Reranking per paper..."):
                results = get_comparison(compare_q, active)
            st.markdown(f"**Q: {compare_q}**")
            st.markdown("---")
            cols = st.columns(len(active))
            for i, name in enumerate(active):
                with cols[i]:
                    color = paper_color(name)
                    short = (name[:30]+"…") if len(name)>30 else name
                    st.markdown(f'<div class="compare-col"><div style="{color};padding:4px 8px;border-radius:6px;margin-bottom:8px;font-size:13px;font-weight:600">📄 {short}</div>', unsafe_allow_html=True)
                    st.markdown(results[name]["answer"])
                    with st.expander("Source chunks + confidence"):
                        for j, src in enumerate(results[name]["sources"]):
                            pct = src["confidence"]; fill,_ = score_color(pct)
                            st.markdown(f"""<div class="source-card">
                                <span class="rank-badge">{j+1}</span>
                                <span style="font-size:12px;font-weight:600;color:{fill}">{pct}%</span>
                                <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{pct}%;background:{fill}"></div></div>
                                {src['text']}…</div>""", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        query_papers = [active[0]] if mode == "single" else active
        tag_html = "".join([f'<span class="pdf-chip" style="{paper_color(n)}">{(n[:18]+"…") if len(n)>18 else n}</span>' for n in query_papers])
        st.markdown(f"<div style='margin-bottom:12px'>{tag_html}</div>", unsafe_allow_html=True)

        with st.container():
            if not st.session_state.chat_history:
                st.markdown("""<div style="text-align:center;padding:40px 0;color:#6b7280">
                    <div style="font-size:32px">💬</div>
                    <p>Papers ready — ask your first question below.</p>
                    <p style="font-size:12px">Answers include inline citations [1][2] linked to source chunks.</p>
                </div>""", unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    ts = msg.get("timestamp", "")
                    if msg["role"] == "user":
                        st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}<div class="msg-time">{ts}</div></div>', unsafe_allow_html=True)
                    else:
                        rendered = render_citations(msg["content"], msg.get("sources", []))
                        # Copy button
                        st.markdown(f'<div class="assistant-bubble">🤖 {rendered}<div class="msg-time">{ts}</div></div>', unsafe_allow_html=True)
                        if msg.get("sources"):
                            with st.expander(f"📎 {len(msg['sources'])} reranked chunks"):
                                for j, src in enumerate(msg["sources"]):
                                    pct = src.get("confidence",0); paper = src.get("paper","")
                                    fill,_ = score_color(pct); color = paper_color(paper)
                                    short_p = (paper[:25]+"…") if len(paper)>25 else paper
                                    st.markdown(f"""<div class="source-card">
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
            ts = time.strftime("%H:%M")
            st.session_state.chat_history.append({"role": "user", "content": question, "sources": [], "timestamp": ts})
            with st.spinner("Retrieving → reranking → generating..."):
                answer, sources = get_answer(question, st.session_state.chat_history[:-1], query_papers)
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources, "timestamp": time.strftime("%H:%M")})
            st.rerun()

# ══════════════════════════════════════════════
# TAB 2 — ARCHITECTURE DIAGRAM (Phase 5)
# ══════════════════════════════════════════════
with tab_arch:
    st.markdown("""
    <div class="app-header">
        <h2 style="margin:0 0 6px">🏗 Paper Architecture</h2>
        <p style="margin:4px 0 0;color:#9ca3af;font-size:14px">
            Auto-extract system architecture from any paper and visualise it as an interactive diagram.
        </p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.papers:
        st.info("Upload a PDF in the sidebar to generate its architecture diagram.")
        st.stop()

    paper_options = list(st.session_state.papers.keys())
    sel_paper = st.selectbox("Select paper", paper_options, label_visibility="collapsed")

    col_btn, col_cache = st.columns([2, 3])
    with col_btn:
        gen_btn = st.button("🏗 Generate Architecture", type="primary", use_container_width=True)
    with col_cache:
        if sel_paper in st.session_state.diagram_data:
            st.caption("✓ Cached — click to regenerate")

    if gen_btn:
        with st.spinner("Extracting architecture with Groq..."):
            try:
                arch = extract_paper_architecture(sel_paper)
                st.session_state.diagram_data[sel_paper] = arch
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                st.stop()

    if sel_paper in st.session_state.diagram_data:
        arch    = st.session_state.diagram_data[sel_paper]
        mermaid = build_mermaid(arch)

        # ── Summary cards ──
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="diagram-card">
                <div class="diagram-section">Problem</div>
                <div style="font-size:13px;color:#e2e8f0">{arch.get('problem','—')}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="diagram-card">
                <div class="diagram-section">Contribution</div>
                <div style="font-size:13px;color:#e2e8f0">{arch.get('contribution','—')}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            tech_pills = "".join([f'<span style="display:inline-block;background:#1e3a5f;color:#93c5fd;border-radius:8px;padding:2px 8px;font-size:11px;margin:2px">{t}</span>' for t in arch.get('key_tech', [])])
            st.markdown(f"""<div class="diagram-card">
                <div class="diagram-section">Key Technologies</div>
                <div>{tech_pills}</div>
            </div>""", unsafe_allow_html=True)

        # ── Pipeline flow ──
        st.markdown("#### System pipeline")
        io_col1, io_col2 = st.columns(2)
        with io_col1:
            st.markdown(f"""<div style="background:#1e3a5f;border-radius:8px;padding:10px 14px;font-size:13px;color:#93c5fd">
                <b>Input:</b> {arch.get('input','—')}</div>""", unsafe_allow_html=True)
        with io_col2:
            st.markdown(f"""<div style="background:#1a2e1a;border-radius:8px;padding:10px 14px;font-size:13px;color:#6ee7b7">
                <b>Output:</b> {arch.get('output','—')}</div>""", unsafe_allow_html=True)

        # ── Component table ──
        st.markdown("#### Components")
        for comp in arch.get("components", []):
            type_colors = {"input":"#93c5fd","output":"#6ee7b7","model":"#c4b5fd","decision":"#fcd34d","process":"#94a3b8"}
            c = type_colors.get(comp.get("type","process"),"#94a3b8")
            st.markdown(f"""<div class="source-card" style="display:flex;align-items:flex-start;gap:12px">
                <span style="background:#1e293b;color:{c};border-radius:6px;padding:2px 8px;font-size:12px;font-weight:700;flex-shrink:0">{comp['id']}</span>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#e2e8f0">{comp['label']}</div>
                    <div style="font-size:12px;color:#6b7280;margin-top:2px">{comp.get('description','')}</div>
                </div>
                <span style="margin-left:auto;font-size:11px;color:{c};border:1px solid {c};border-radius:4px;padding:1px 6px;flex-shrink:0">{comp.get('type','process')}</span>
            </div>""", unsafe_allow_html=True)

        # ── Mermaid diagram ──
        st.markdown("#### Flow diagram")
        st.code(mermaid, language="text")
        st.caption("Copy the code above → paste at mermaid.live for an interactive, downloadable diagram.")

        # Download Mermaid source
        st.download_button(
            "⬇️ Download Mermaid (.mmd)",
            data=mermaid,
            file_name=f"{sel_paper.replace('.pdf','')}_architecture.mmd",
            mime="text/plain",
            use_container_width=True,
        )

        # ── Q&A about architecture ──
        st.markdown("---")
        st.markdown("#### Ask about this paper's architecture")
        arch_q = st.text_input("Architecture question", placeholder="How does the data flow through the system?", label_visibility="collapsed")
        if arch_q:
            with st.spinner("Answering..."):
                docs    = st.session_state.papers[sel_paper]["vectorstore"].similarity_search(arch_q, k=4)
                context = "\n\n".join([d.page_content for d in docs])
                resp    = get_llm().invoke([
                    SystemMessage(content="Answer based strictly on this paper's architecture and content."),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {arch_q}"),
                ])
            st.markdown(f'<div class="assistant-bubble">🤖 {resp.content}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — BENCHMARK (Phase 4)
# ══════════════════════════════════════════════
with tab_bench:
    st.markdown("""
    <div class="app-header">
        <h2 style="margin:0 0 6px">📊 Retrieval Benchmark</h2>
        <p style="margin:4px 0 0;color:#9ca3af;font-size:14px">Measure retrieval accuracy using the Hit@K metric.</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.papers:
        st.info("Upload at least one PDF in the sidebar.")
        st.stop()

    with st.expander("ℹ️ How Hit@K works", expanded=False):
        st.markdown("""**Hit@K** checks if the correct answer chunk appears in the top K retrieved results.
- For each question, we check if expected keywords appear in the top K chunks
- **Hit@3 = 80%** → answer found in top 3 chunks for 80% of questions
- Industry standard metric used by LlamaIndex, LangChain eval, RAGAS""")

    st.markdown("---")
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("**Select paper**")
        bench_paper = st.selectbox("Paper", list(st.session_state.papers.keys()), label_visibility="collapsed")
        st.markdown("**K value**")
        k_val = st.slider("Top K chunks", 1, 10, 3)
        st.markdown("**Test questions**")
        test_mode = st.radio("Mode", ["🤖 Auto-generate 10 via Groq", "✏️ Enter manually"], label_visibility="collapsed")
    with col_r:
        st.markdown("**What you get**")
        st.markdown("""<div class="bench-card"><div style="font-size:13px;color:#94a3b8;line-height:2">
            ✦ Hit@K accuracy % for your resume<br>
            ✦ Pass/fail per question<br>
            ✦ Per-difficulty breakdown<br>
            ✦ Avg retrieval latency<br>
            ✦ CSV download
        </div></div>""", unsafe_allow_html=True)

    custom_cases = []
    if "✏️" in test_mode:
        n_manual = st.number_input("Number of questions", 1, 20, 5)
        for i in range(int(n_manual)):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                q = st.text_input(f"Q{i+1}", key=f"bq_{i}", placeholder="What is the main contribution?")
            with c2:
                kws = st.text_input(f"Keywords {i+1}", key=f"bk_{i}", placeholder="contribution, system")
            with c3:
                diff = st.selectbox("Diff", ["easy","medium","hard"], key=f"bd_{i}")
            if q and kws:
                custom_cases.append({"question": q, "keywords": [k.strip() for k in kws.split(",")], "difficulty": diff})

    if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):
        if "🤖" in test_mode:
            with st.spinner("Generating questions with Groq..."):
                try:
                    test_cases = auto_generate_test_cases(bench_paper)
                    st.success(f"Generated {len(test_cases)} questions.")
                except Exception as e:
                    st.error(f"Failed: {e}"); st.stop()
        else:
            if not custom_cases:
                st.warning("Enter at least one question."); st.stop()
            test_cases = custom_cases
        with st.spinner(f"Running Hit@{k_val} on {len(test_cases)} questions..."):
            bench = run_benchmark(bench_paper, test_cases, k=k_val)
            st.session_state.bench_results = bench

    if st.session_state.bench_results:
        bench = st.session_state.bench_results
        st.markdown("---")
        st.markdown("### Results")
        m1, m2, m3, m4 = st.columns(4)
        acc_c = "#4ade80" if bench["accuracy"] >= 70 else "#facc15" if bench["accuracy"] >= 50 else "#f87171"
        with m1:
            st.markdown(f'<div class="bench-metric"><div class="val" style="color:{acc_c}">{bench["accuracy"]}%</div><div class="lbl">Hit@{bench["k"]} Accuracy</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="bench-metric"><div class="val" style="color:#93c5fd">{bench["hits"]}/{bench["total"]}</div><div class="lbl">Passed</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="bench-metric"><div class="val" style="color:#c4b5fd">{bench["avg_latency"]}ms</div><div class="lbl">Avg Latency</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="bench-metric"><div class="val" style="color:#fcd34d">{bench["k"]}</div><div class="lbl">K</div></div>', unsafe_allow_html=True)

        st.markdown(f"""<div style="margin:16px 0;padding:14px 18px;background:#0f2a1a;border:1px solid #166534;border-radius:10px;font-size:14px;color:#4ade80">
            📋 <b>Resume bullet:</b> "Achieved <b>{bench['accuracy']}% Hit@{bench['k']}</b> retrieval accuracy on {bench['total']} test questions — avg {bench['avg_latency']}ms retrieval latency."
        </div>""", unsafe_allow_html=True)

        if bench["by_difficulty"]:
            st.markdown("#### By difficulty")
            diff_cols = st.columns(len(bench["by_difficulty"]))
            diff_c = {"easy":"#4ade80","medium":"#facc15","hard":"#f87171"}
            for col,(diff,stats) in zip(diff_cols, bench["by_difficulty"].items()):
                pct = round(stats["hits"]/stats["total"]*100,1) if stats["total"] else 0
                c   = diff_c.get(diff,"#94a3b8")
                with col:
                    st.markdown(f'<div class="bench-metric"><div class="val" style="color:{c}">{pct}%</div><div class="lbl">{diff.capitalize()} ({stats["hits"]}/{stats["total"]})</div></div>', unsafe_allow_html=True)

        st.markdown("#### Per-question results")
        for i, r in enumerate(bench["results"]):
            css  = "pass-row" if r["hit"] else "fail-row"
            icon = "✅" if r["hit"] else "❌"
            kw   = f" — matched: <b>{r['matched_kw']}</b>" if r.get("matched_kw") else ""
            st.markdown(f'<div class="{css}">{icon} <b>Q{i+1}</b> [{r["difficulty"]}] {r["question"]}{kw}<span style="float:right;font-size:11px;color:#6b7280">{r["latency_ms"]}ms</span></div>', unsafe_allow_html=True)

        st.markdown("")
        st.download_button("⬇️ Download CSV", data=results_to_csv(bench),
            file_name=f"benchmark_hit@{bench['k']}.csv", mime="text/csv", use_container_width=True)
