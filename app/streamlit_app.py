"""Streamlit frontend for Document Q&A."""
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# `streamlit run app/streamlit_app.py` only puts app/ on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="📄 Document Q&A", layout="wide")
st.title("📄 Intelligent Document Q&A")
st.caption("Upload documents and ask questions — powered by RAG")

# Initialize pipeline in session state
if "pipeline" not in st.session_state:
    try:
        with st.spinner("Loading models..."):
            st.session_state.pipeline = RAGPipeline()
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        st.info("Check that ANTHROPIC_API_KEY is set in your .env file.")
        st.stop()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("🔄 Process Documents"):
        with st.spinner("Processing documents..."):
            # Keep original filenames so citations show e.g. "report.pdf"
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_paths = []
                for f in uploaded_files:
                    path = Path(tmp_dir) / f.name
                    path.write_bytes(f.getvalue())
                    temp_paths.append(str(path))

                n_chunks = st.session_state.pipeline.ingest(temp_paths)

        if n_chunks:
            st.success(f"✅ Processed into {n_chunks} chunks!")
        else:
            st.warning("No text could be extracted from the uploaded files.")

    if st.session_state.chat_history and st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- Main: Chat Interface ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = st.session_state.pipeline.query(question)
        except Exception as e:
            result = {"answer": f"⚠️ Error generating answer: {e}", "sources": []}

        st.markdown(result["answer"])

        # Show sources in expander
        if result["sources"]:
            with st.expander("📎 View Sources"):
                for src in result["sources"]:
                    st.markdown(f"**{src['source']}** (Chunk {src['chunk_index']})")
                    st.caption(src["content_preview"])
                    st.divider()

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
    })
