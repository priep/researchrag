import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
import time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Key is loaded from Streamlit secrets — see get_api_key() function above
MODEL_NAME = "llama-3.1-8b-instant"
TOP_K = 4
MAX_HISTORY_TURNS = 6  # how many past Q&A pairs to include in prompt

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchRAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Chat bubbles */
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
    /* Source chunks */
    .source-card {
        background: #111827;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 13px;
        color: #9ca3af;
    }
    /* Header */
    .app-header {
        padding: 10px 0 20px 0;
        border-bottom: 1px solid #2d3748;
        margin-bottom: 16px;
    }
    /* Sidebar status badge */
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    .status-ready { background: #064e3b; color: #6ee7b7; }
    .status-waiting { background: #1f2937; color: #9ca3af; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # list of {"role": "user"|"assistant", "content": str, "sources": list}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=get_api_key(),
        model_name=MODEL_NAME
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# PDF PROCESSING
# ─────────────────────────────────────────────
def process_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)

# ─────────────────────────────────────────────
# ANSWER GENERATION (with history)
# ─────────────────────────────────────────────
def get_answer(vectorstore, question, history):
    docs = vectorstore.similarity_search(question, k=TOP_K)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # Build conversation history string for the prompt
    history_text = ""
    recent = history[-MAX_HISTORY_TURNS * 2:]  # last N turns (user+assistant pairs)
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    system_prompt = """You are a precise research assistant. Answer questions based strictly on the provided paper context.
- Be concise and factual
- If the answer isn't in the context, say so clearly
- Reference specific parts of the paper when relevant
- Never fabricate information"""

    user_prompt = f"""Paper context:
{context}

Conversation so far:
{history_text if history_text else "(No prior conversation)"}

Current question: {question}

Answer clearly and concisely based on the paper."""

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    sources = [doc.page_content[:280] for doc in docs]
    return response.content, sources

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 ResearchRAG")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload a research paper", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            # New file uploaded — reset everything
            with st.spinner("Processing PDF..."):
                vs, n_chunks = process_pdf(uploaded_file)
                st.session_state.vectorstore = vs
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
            st.success(f"Ready! {n_chunks} chunks indexed.")

    if st.session_state.pdf_name:
        st.markdown(f"""
        <span class="status-badge status-ready">✓ Loaded</span>
        <br><small style="color:#6b7280;margin-top:4px;display:block">{st.session_state.pdf_name}</small>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <span class="status-badge status-waiting">○ No PDF loaded</span>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Chat history list in sidebar
    if st.session_state.chat_history:
        st.markdown("**Conversation**")
        questions = [m["content"] for m in st.session_state.chat_history if m["role"] == "user"]
        for i, q in enumerate(questions):
            short = q[:45] + ("…" if len(q) > 45 else "")
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
# MAIN CHAT AREA
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h2 style="margin:0">📄 Research Paper Q&A</h2>
    <p style="margin:4px 0 0;color:#9ca3af;font-size:14px">
        Upload a paper, ask anything — answers are grounded in the document.
    </p>
</div>
""", unsafe_allow_html=True)

# Render existing messages
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        if st.session_state.vectorstore:
            st.markdown("""
            <div style="text-align:center;padding:40px 0;color:#6b7280">
                <div style="font-size:32px">💬</div>
                <p>Paper is ready! Ask your first question below.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 0;color:#6b7280">
                <div style="font-size:48px">📂</div>
                <p style="font-size:16px">Upload a PDF in the sidebar to get started.</p>
                <p style="font-size:13px">Supports research papers, reports, and any text-heavy PDF.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-bubble">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

                # Expandable source chunks
                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} source chunks used", expanded=False):
                        for i, src in enumerate(msg["sources"]):
                            st.markdown(f"""
                            <div class="source-card">
                                <b style="color:#60a5fa">Chunk {i+1}</b><br>
                                {src}{"…" if len(src) == 280 else ""}
                            </div>
                            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT BAR
# ─────────────────────────────────────────────
if st.session_state.vectorstore:
    col1, col2 = st.columns([8, 1])
    with col1:
        question = st.chat_input("Ask a question about the paper...")
    
    if question:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "sources": [],
        })

        # Generate answer
        with st.spinner("Searching and generating answer..."):
            answer, sources = get_answer(
                st.session_state.vectorstore,
                question,
                [m for m in st.session_state.chat_history[:-1]],  # history excluding current Q
            )

        # Add assistant message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

        st.rerun()
else:
    st.chat_input("Upload a PDF first to enable Q&A...", disabled=True)