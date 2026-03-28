# ResearchRAG 📄

**Intelligent Research Paper Q&A using Retrieval-Augmented Generation**

Upload any research paper (PDF) and ask natural language questions about it. Every answer is grounded in the actual document — no hallucinations.

🔗 **[Live Demo →](https://your-app-name.streamlit.app)**  ← update this after deploying

---

## Features

- **Semantic Q&A** — asks questions in plain English, retrieves relevant passages via FAISS
- **Multi-turn chat** — follow-up questions work because conversation history is injected into every prompt
- **Source transparency** — every answer shows exactly which chunks from the paper were used
- **Fast inference** — Groq's llama-3.1-8b-instant for sub-3s responses

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq — llama-3.1-8b-instant |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS-cpu |
| Orchestration | LangChain |

## Local Setup

```bash
git clone https://github.com/yourusername/researchrag
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

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (secrets.toml is gitignored — safe to push)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo, branch `main`, file `rag.py`
4. Click **Advanced settings** → add secret: `GROQ_API_KEY = "gsk_..."`
5. Deploy

---

Built by [Priya Patidar](https://linkedin.com/in/priya--patidar)
