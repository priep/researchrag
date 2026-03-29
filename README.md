# ResearchRAG 📄

**Intelligent Research Paper Q&A using Retrieval-Augmented Generation**

> Upload any research paper (PDF) and ask natural language questions about it. Every answer is grounded in the actual document — no hallucinations.

🔗 **[Live Demo → researchragproject.streamlit.app](https://researchragproject.streamlit.app)**  
💻 **[GitHub → github.com/priep/researchrag](https://github.com/priep/researchrag)**

---

## What it does

ResearchRAG is a **production-grade multi-document RAG system** with 5 core features:

| Feature | Description |
|---|---|
| 💬 **Multi-PDF Q&A** | Upload up to 5 papers, query them in single / merged / compare mode |
| ⚖️ **Side-by-side compare** | Ask the same question to multiple papers, answers shown in columns |
| ✦ **CrossEncoder reranking** | FAISS retrieves top-10 → `ms-marco-MiniLM-L-6-v2` reranks → top-4 |
| 📊 **Hit@K benchmarking** | Auto-generate test suites, measure retrieval accuracy with the Hit@K metric |
| 🏗️ **Architecture extraction** | Auto-extract system components, pipeline, and generate Mermaid diagrams |

---

## Benchmark Results

| Metric | Value |
|---|---|
| Hit@3 Accuracy | **80%** |
| Questions Passed | **8/10** |
| Avg Retrieval Latency | **12.6ms** |
| Easy questions | 100% (3/3) |
| Medium questions | 60% (3/5) |
| Hard questions | 100% (2/2) |

---

## System Architecture

```
PDF Upload
    ↓
Text Extraction (PyPDFLoader)
    ↓
Semantic Chunking (RecursiveCharacterTextSplitter — 600 tokens, 80 overlap)
    ↓
Embedding Generation (sentence-transformers/all-MiniLM-L6-v2 — 384 dim)
    ↓
FAISS Vector Index (saved to disk — persists across sessions)
    ↓
User Query → FAISS similarity search (top-10 candidates)
    ↓
CrossEncoder Reranking (ms-marco-MiniLM-L-6-v2 → top-4)
    ↓
Confidence Score Normalisation (min-max → 0–100%)
    ↓
Groq LLM (llama-3.1-8b-instant) → Answer with inline citations [1][2]
    ↓
Streamlit UI — chat bubbles · source cards · confidence bars
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq — `llama-3.1-8b-instant` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector store | FAISS-cpu with `save_local` / `load_local` persistence |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (CrossEncoder) |
| Orchestration | LangChain |
| PDF parsing | PyPDFLoader |
| Language | Python 3.10+ |

---

## Key Technical Decisions

**Why CrossEncoder reranking?**  
Standard vector similarity (FAISS) retrieves based on embedding distance — fast but imprecise. CrossEncoder reads both the question and each chunk together, scoring true relevance. This is the approach used in production RAG systems. Retrieve 10 with FAISS, rerank all 10, keep the best 4.

**Why min-max normalisation for confidence scores?**  
`ms-marco-MiniLM-L-6-v2` outputs raw logits in the range −10 to +10. Sigmoid collapses these to near 0. Min-max scaling guarantees meaningful visual spread: best chunk = 100%, worst = 0%.

**Why FAISS persistence?**  
Embedding 50+ page papers takes ~10s on CPU. `save_local` / `load_local` caches the index so repeat uploads are instant. The sidebar shows "Loaded from cache" instead of reprocessing.

---

## Local Setup

```bash
git clone https://github.com/priep/researchrag
cd researchrag
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Run:
```bash
streamlit run rag.py
```

---

## Streamlit Cloud Deployment

1. Push repo to GitHub (secrets.toml is gitignored)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Repo: `priep/researchrag` · Branch: `main` · File: `rag.py`
4. Advanced settings → Secrets → paste `GROQ_API_KEY = "gsk_..."`
5. Deploy

---

## Evaluation — Hit@K Metric

Hit@K checks whether the correct answer chunk appears in the top K retrieved results.

- For each test question, expected keywords must appear in the top K chunks
- **Hit@3 = 80%** → correct chunk found in top 3 for 8/10 questions
- Industry-standard metric used by LlamaIndex, LangChain Eval, RAGAS

The benchmark tab auto-generates 10 test questions using Groq, runs Hit@K, and exports results as CSV.

---

## What's different from a standard RAG tutorial

| Standard tutorial | ResearchRAG |
|---|---|
| Single PDF | Up to 5 PDFs simultaneously |
| FAISS only | FAISS + CrossEncoder reranking |
| No confidence scores | Min-max normalised % per chunk |
| No evaluation | Hit@K benchmarking with CSV export |
| Basic UI | 3 query modes, citation badges, timestamps |
| No architecture insight | Auto Mermaid diagram generation |

---

Built by Priya Patidar
