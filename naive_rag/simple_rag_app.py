# Main Streamlit application — handles upload, ingestion, and chat entirely from the UI.

import streamlit as st
import os
import tempfile

# import methods from ingest.py and rag_chain.py
from simple_ingest_document import ingest_document, list_ingested_documents, delete_document
from simple_rag_chain import ask_question


# Must be the first Streamlit call
st.set_page_config(
    page_title="Agentic-AI RAG Chatbot",
    page_icon="📚",
    layout="wide",
)


st.title("📚 RAG Chatbot")
st.caption("Upload a PDF or DOCX and chat with your document.")


# Sidebar
with st.sidebar:
    st.header("📂 Document Management")
    st.subheader("Upload New Document")


    # File uploader — only PDF and DOCX allowed
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=["pdf", "docx"],
    )


    if uploaded_file is not None:
        st.info(f"**{uploaded_file.name}** ({round(uploaded_file.size / 1024, 1)} KB)")


        if st.button("Process Document", type="primary", use_container_width=True):


            # LlamaIndex needs a real file path, not bytes in memory
            # So we write the uploaded bytes to a temp file on disk
            # We preserve the original filename inside a temp directory so file readers
            # still see the correct extension and filename format
            temp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(temp_dir, uploaded_file.name)

            with open(tmp_path, "wb") as tmp:
                tmp.write(uploaded_file.getvalue())


            try:
                st.markdown("**Processing...**")
                progress_bar = st.progress(0, text="Starting ingestion...")
                status_text = st.empty()


                # Callback passed to ingest_document to update progress bar live
                def update_progress(current, total):
                    pct = int((current / total) * 100)
                    progress_bar.progress(pct, text=f"Embedding chunk {current} of {total}...")
                    status_text.caption(f"{pct}% complete")


                total_chunks = ingest_document(
                    file_path=tmp_path,
                    original_filename=uploaded_file.name,
                    progress_callback=update_progress,
                )


                progress_bar.progress(100, text="Done!")
                status_text.empty()
                st.success(f"✅ Ingested **{uploaded_file.name}** → {total_chunks} chunks stored in PostgreSQL")
                st.rerun()  # Refresh sidebar to show new doc in the list


            except Exception as e:
                import traceback
                st.error(f"❌ Ingestion failed: {repr(e)}")
                st.code(traceback.format_exc())


            finally:
                # Always clean up temp file + temp directory
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)


    st.divider()


    st.subheader("📋 Processed Documents")
    ingested_docs = list_ingested_documents()


    if not ingested_docs:
        st.caption("No documents ingested yet. Upload one above.")
    else:
        for doc in ingested_docs:
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"📄 {doc}")
            if col2.button("🗑️", key=f"del_{doc}", help=f"Delete {doc}"):
                delete_document(doc)
                st.success(f"Deleted {doc}")
                st.rerun()


    st.divider()


    with st.expander("ℹ️ How it works"):
        st.markdown("""
1. Upload PDF or DOCX from the UI
2. Document is split into ~500 token chunks with overlap
3. Each chunk is converted to a 1024-dim vector via NVIDIA NIM embedding API
4. Chunk text + vector stored together in PostgreSQL (pgvector)
5. User question is embedded using the same NVIDIA model
6. PostgreSQL runs cosine similarity search → returns top matching chunks
7. Those chunks are passed as context to Llama 3.2 (running via Ollama locally)
8. Llama 3.2 generates a grounded answer using only the retrieved context
""")


# Guard: block chat if no documents ingested
ingested_docs = list_ingested_documents()


if not ingested_docs:
    st.info("⬅️ Upload and ingest a document from the sidebar to start chatting.")
    st.stop()  # Stop rendering — hides chat input until at least one doc exists


# Chat history
# session_state persists data across reruns within the same browser session
if "messages" not in st.session_state:
    st.session_state.messages = []


# Render all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "context" in message:
            with st.expander("📎 View retrieved chunks from vector DB"):
                for i, chunk in enumerate(message["context"], 1):
                    st.markdown(
                        f"**Chunk {i}:** Document: `{chunk['doc_name']}` | "
                        f"Chunk Index: `{chunk['chunk_index']}` | "
                        f"Similarity: `{chunk['similarity']:.4f}`"
                    )
                    preview = chunk["chunk_text"][:500] + "..." if len(chunk["chunk_text"]) > 500 else chunk["chunk_text"]
                    st.caption(preview)
                    st.divider()


# Chat input
if prompt := st.chat_input("Ask something about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})


    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching vector DB and generating answer..."):
            try:
                # Run the full LangGraph RAG pipeline
                result = ask_question(prompt)
                answer = result["answer"]
                chunks = result["retrieved_chunks"]


                st.markdown(answer)


                # Show retrieved chunks — proves answer is from DB, not hallucinated
                with st.expander("📎 View retrieved chunks from vector DB"):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(
                            f"**Chunk {i}:** Document: `{chunk['doc_name']}` | "
                            f"Chunk Index: `{chunk['chunk_index']}` | "
                            f"Similarity: `{chunk['similarity']:.4f}`"
                        )
                        preview = chunk["chunk_text"][:500] + "..." if len(chunk["chunk_text"]) > 500 else chunk["chunk_text"]
                        st.caption(preview)
                        st.divider()


                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "context": chunks,
                })


            except Exception as e:
                import traceback
                st.error(f"Error: {repr(e)}")
                st.code(traceback.format_exc())