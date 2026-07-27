# ============================================================
# app.py
#
# WHAT THIS FILE DOES — UI SUMMARY
# ============================================================
#
# This is the main Streamlit UI for the Multi-Agent RAG Chatbot.
# It is the only file the user directly interacts with.
#
# RESPONSIBILITIES:
#
# 1. PAGE SETUP
#    - Configures Streamlit page title, icon, and layout
#    - Renders the app title and subtitle caption
#    - Shows the LangGraph Architecture Diagram once at the top
#
# 2. HELPER FUNCTIONS
#    - render_trace_in_container() → renders execution trace steps in UI
#    - deduplicate_trace()         → removes duplicate trace entries by trace_id
#    - _render_assistant_details() → renders trace, evidence, SQL, retry badge, evaluator score, agent label
#    - _handle_final_result()      → safely stores completed assistant result in chat history
#
# 3. SESSION STATE MANAGEMENT
#    - st.session_state.messages                    → full chat history across reruns
#    - st.session_state.thread_id                   → unique ID per conversation, required
#                                                     by LangGraph checkpointer for HITL resume
#    - st.session_state.hitl_pending_state          → stores the paused graph result when
#                                                     SQL agent needs human approval before running
#    - st.session_state.evaluator_hitl_pending_state → stores a low-confidence answer that
#                                                     needs human review after evaluator retries fail
#    - st.session_state.show_sql_editor             → toggles the editable SQL text area
#
# 4. SIDEBAR — DOCUMENT MANAGEMENT
#    - File uploader: accepts PDF and DOCX files
#    - Calls ingest_document() with a live progress bar callback
#    - Lists all ingested documents fetched from PostgreSQL
#    - Allows deletion of individual documents
#    - Shows Agent Legend (RAG / Web Search / SQL)
#    - Shows "How it works" explainer in an expander
#    - Warns if no documents are ingested (web search still works regardless)
#
# 5. HITL / HOTL UI
#    - SQL approval UI = Human-on-the-Loop
#      Triggered when SQL Agent generates a query and pauses before execution
#    - Evaluator review UI = Human-in-the-Loop
#      Triggered only when evaluator score stays below threshold after retries
#
# 6. CHAT HISTORY RENDERING
#    - Iterates st.session_state.messages and renders each message
#    - For each assistant message renders:
#        🧠 Reasoning + Action Trace expander
#        📎 Retrieved chunks panel
#        🌐 Web search results panel
#        🗃️ Generated SQL query panel
#        🔄 Evaluator retry badge
#        🧪 Evaluator score
#        Agent badge label
#
# 7. CHAT INPUT + NEW QUERY FLOW
#    - Chat input is rendered only when no SQL/HITL review is pending
#    - Each new question generates a fresh thread_id (uuid4) for LangGraph
#    - Calls ask_question(question, thread_id) from agentic_workflow.py
#    - Checks result for awaiting_human_approval=True → triggers SQL review flow
#    - Checks result for hitl_needed=True → triggers evaluator Human-in-the-Loop
#    - Otherwise renders answer, trace, evidence panels, and agent badge
#    - Saves completed result into st.session_state.messages for history
# ============================================================


# ============================================================
# app.py
# ============================================================

import os
import uuid
import tempfile
import streamlit as st

from ingest import ingest_document, list_ingested_documents, delete_document
from agentic_workflow import ask_question
from graph_visualizer import render_architecture_diagram

import mlflow
from mlflow import langchain as mlflow_langchain

# Configure MLflow to log all runs and traces to the local tracking server.
# This experiment name must match what we see in the MLflow UI.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Default")

# Enable MLflow's LangChain integration:
# - Automatically creates a trace for each LangChain chain/LLM call
# - Captures inputs, outputs, token usage, and timings
# - Populates the "Traces" tab in the MLflow GenAI UI
mlflow_langchain.autolog()


# ── 1. PAGE SETUP ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Agent RAG Chatbot")
st.caption("Ask Anything - The Router Agent decides the right agent for you based on your query")

render_architecture_diagram()


# ── 2. HELPER FUNCTIONS ────────────────────────────────────────────────────────

def render_trace_in_container(trace_steps: list, container):
    """Render each execution trace step as bold heading + caption detail."""
    with container:
        for item in trace_steps or []:
            st.markdown(f"**{item.get('icon', '')} {item.get('step', '')}**")
            if item.get("detail"):
                st.caption(item["detail"])


def deduplicate_trace(trace_steps: list) -> list:
    """
    Deduplicate trace entries while preserving order.
    Uses trace_id as the unique key if present.
    Falls back to (icon, step, detail) tuple if trace_id is missing.
    """
    deduped = []
    seen = set()

    for item in trace_steps or []:
        trace_id = item.get("trace_id")
        if trace_id:
            key = ("trace_id", trace_id)
        else:
            key = (
                "fallback",
                item.get("icon", ""),
                item.get("step", ""),
                item.get("detail", "")
            )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(item)

    return deduped


def _safe_answer_text(result: dict) -> str:
    """
    Return a non-empty assistant message.
    Prevents blank assistant bubbles if answer is missing or whitespace.
    """
    answer = result.get("answer", "")
    if answer is None:
        answer = ""
    answer = str(answer).strip()

    if not answer:
        if result.get("agent_used") == "sql":
            return "SQL query completed, but no final natural-language answer was returned."
        return "No final answer was returned."
    return answer


def _render_assistant_details(
    agent_used: str,
    chunks: list,
    web_results: list,
    generated_sql: str,
    execution_trace: list,
    retry_count: int,
    evaluator_reason: str = "",
):
    """Render trace, evidence panels, retry badge, evaluator note, and agent label."""
    if execution_trace:
        with st.expander("🧠 View reasoning + action trace", expanded=False):
            trace_container = st.container()
            render_trace_in_container(execution_trace, trace_container)

    if agent_used == "rag" and chunks:
        with st.expander("📎 View retrieved chunks from vector DB"):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(
                    f"**Chunk {i}:** Document: `{chunk['doc_name']}` | "
                    f"Chunk Index: `{chunk['chunk_index']}` | "
                    f"Similarity: `{chunk['similarity']:.4f}`"
                )
                preview = (
                    chunk["chunk_text"][:500] + "..."
                    if len(chunk["chunk_text"]) > 500
                    else chunk["chunk_text"]
                )
                st.caption(preview)
                st.divider()

    if agent_used == "web_search" and web_results:
        with st.expander("🌐 View web search results used"):
            for i, result_item in enumerate(web_results, 1):
                st.markdown(
                    f"**Result {i}:** "
                    f"[{result_item.get('title', 'No title')}]"
                    f"({result_item.get('url', '')})"
                )
                st.caption(result_item.get("content", "")[:400] + "...")
                st.divider()

    if agent_used == "sql" and generated_sql:
        with st.expander("🗃️ View generated SQL query"):
            st.code(generated_sql, language="sql")

    if retry_count > 0:
        st.markdown(
            f"🔄 Evaluator retried this answer "
            f"{retry_count} time(s) before accepting it."
        )

    if evaluator_reason:
        st.caption(f"Evaluator note: {evaluator_reason}")

    if agent_used == "web_search":
        st.markdown("🌐 **Answered by: Web Search Agent (Tavily)**")
    elif agent_used == "sql":
        st.markdown("🗃️ **Answered by: SQL Agent**")
    else:
        st.markdown("📄 **Answered by: PDF RAG Agent**")


def _handle_final_result(result: dict):
    """
    Save a completed assistant result into message history.
    This is used by BOTH the normal chat path and the resume/review paths.
    """
    answer = _safe_answer_text(result)
    chunks = result.get("retrieved_chunks", []) or []
    agent_used = result.get("agent_used", "rag")
    web_results = result.get("web_results", []) or []
    generated_sql = result.get("generated_sql", "") or ""
    retry_count = result.get("retry_count", 0) or 0
    evaluator_reason = result.get("evaluator_reason", "") or ""
    execution_trace = deduplicate_trace(result.get("execution_trace", []) or [])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context": chunks,
        "agent_used": agent_used,
        "web_results": web_results,
        "generated_sql": generated_sql,
        "execution_trace": execution_trace,
        "retry_count": retry_count,
        "evaluator_reason": evaluator_reason,
    })


def build_chat_history_for_workflow(messages: list[dict]) -> list[dict]:
    """
    Keep only lightweight history needed for follow-up rewriting.

    We do not pass retrieved chunks or web results into memory,
    only user/assistant text and agent label where available.
    """
    history = []

    for msg in messages[-6:]:  # last 6 messages
        history.append({
            "role": msg.get("role", ""),
            "content": msg.get("content", ""),
            "agent_used": msg.get("agent_used", ""),
        })

    return history


# ── 3. SESSION STATE MANAGEMENT ────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "hitl_pending_state" not in st.session_state:
    st.session_state.hitl_pending_state = None

if "show_sql_editor" not in st.session_state:
    st.session_state.show_sql_editor = False

if "evaluator_hitl_pending_state" not in st.session_state:
    st.session_state.evaluator_hitl_pending_state = None


# ── 4. SIDEBAR — DOCUMENT MANAGEMENT ──────────────────────────────────────────

with st.sidebar:
    st.header("📂 Document Management")
    st.subheader("Upload New Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=["pdf", "docx"],
    )

    if uploaded_file is not None:
        st.info(f"**{uploaded_file.name}** ({round(uploaded_file.size / 1024, 1)} KB)")

        if st.button("Process Document", type="primary", use_container_width=True):
            temp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(temp_dir, uploaded_file.name)

            with open(tmp_path, "wb") as tmp:
                tmp.write(uploaded_file.getvalue())

            try:
                st.markdown("**Processing...**")
                progress_bar = st.progress(0, text="Starting ingestion...")
                status_text = st.empty()

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
                st.success(
                    f"✅ Ingested **{uploaded_file.name}** → "
                    f"{total_chunks} chunks stored in PostgreSQL"
                )
                st.rerun()

            except Exception as e:
                import traceback
                st.error(f"❌ Ingestion failed: {repr(e)}")
                st.code(traceback.format_exc())

            finally:
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
            if col2.button(":material/delete:", key=f"del_{doc}", help=f"Delete {doc}"):
                delete_document(doc)
                st.success(f"Deleted {doc}")
                st.rerun()

    st.divider()
    st.markdown("**🧭 Agent Legend**")
    st.markdown("📄 **PDF RAG Agent** - answers from your uploaded documents")
    st.markdown("🌐 **Web Search Agent** - answers from live internet via Tavily")
    st.markdown("🗃️ **SQL Agent** - converts your question into SQL, queries PostgreSQL, and answers from table data")

    st.divider()

    with st.expander("ℹ️ How it works"):
        st.markdown("""
**Multi-Agent System with Router + Evaluator + Human Intervention**

The chatbot uses a multi-step workflow powered by **NVIDIA NIM**:

- **Routing + Evaluation LLM:** Meta Llama 3.1 8B Instruct (`meta/llama-3.1-8b-instruct`)
- **Generation LLM:** OpenAI GPT-OSS 20B (`openai/gpt-oss-20b`)
- **Embeddings:** NVIDIA NIM Retrieval QA E5 v5 (`nvidia/nv-embedqa-e5-v5`)


**Flow**

1. Your question first goes to the :orange[**Router Agent**]
2. The router decides whether the question should be handled by:
- the **PDF Agent**
- the **Web Search Agent**
- the **SQL Agent**

3. Based on that decision:

- **PDF Agent:** embeds the question, retrieves the most relevant document chunks from PostgreSQL using cosine similarity, and generates an answer grounded in the uploaded documents.
- **Web Search Agent:** searches the live internet through Tavily, gathers the top results, and generates an answer grounded in those results.
- **SQL Agent:** generates a SQL `SELECT` query, pauses for your approval before execution, and then generates an answer from the approved query results.

4. The final answer is then sent to the :orange[**Evaluator Agent**]

- The answer is generated by the OpenAI GPT OSS 20B model
- The Evaluator Agent checks it, acting as an :green[LLM-as-a-judge] using a separate, lightweight model (Meta Llama 3.1 8B) — so the model judging the answer is not the same one that generated it
- It checks whether the answer is relevant, grounded, and acceptable
- It can trigger a retry up to **2 times**
- If the answer still does not pass, **Human-in-the-Loop (HITL)** is invoked
                    
:blue[**Human-in-the-Loop (HITL)**]

This applies to **final answer quality control**.

- After an answer is generated, the Evaluator Agent checks it
- If the evaluator is not satisfied, it asks the system to retry
- The system can retry at most **2 times**
- Only if the answer still does not pass after those retries is HITL triggered
- At that stage, you can rephrase the question or accept the answer


:blue[**Human-on-the-Loop (HOTL)**]

This applies to the **SQL Agent**.

- Every generated SQL query requires human review before execution
- You can **Approve**, **Edit**, or **Reject** the query
- The SQL is executed only after approval
- This means human intervention is required **every time** before any SQL query runs

:blue[**Conversation Memory**] 
                    
This applies to follow-up question handling.

- The History Rewrite node reads the latest question plus recent turns
- If the question is a follow-up, it rewrites pronouns and implicit references into a standalone question
- The rewritten question is then used for routing and evaluation
- This improves answers for follow-ups like “how does it work?” or “explain more about that”


:blue[**Routing Logic**]

- **PDF Agent** → questions about uploaded files, PDFs, reports, or document content
- **Web Search Agent** → current, recent, live, or external information
- **SQL Agent** → structured questions about database tables, rows, counts, or aggregates


:blue[**Reasoning + Action Trace**]

The UI shows the workflow step by step, including:
- which agent was selected
- what action it performed
- why the evaluator accepted or retried the response

:blue[**Observability with MlFlow**]

MlFlow's Langchain autologging is used to store observability metrics
- Auto logging metrics such as - Request, Response, Tokens, Latency
- Custom metrics are logged using Mlflow runs
""")
        
    with st.expander("🛠️ Tech Stack Used", expanded=False):
        st.markdown("""
**Core Frameworks**
- Python (3.12.10)
- Streamlit
- LangChain
- LangGraph

**LLM / AI Layer**
- NVIDIA NIM
- ChatOpenAI wrapper
- Meta Llama 3.1 8B Instruct (`meta/llama-3.1-8b-instruct`) for routing + evaluation
- OpenAI GPT-OSS 20B (`openai/gpt-oss-20b`) for generation
- NVIDIA embedding model (`nvidia/nv-embedqa-e5-v5`)

**Retrieval / RAG**
- PostgreSQL
- pgvector
- Cosine similarity search
- Document chunking and embeddings

**Web Search**
- Tavily API

**SQL / Database**
- PostgreSQL
- SQL generation with human approval
- Human-on-the-Loop before query execution

**Workflow / Orchestration**
- Multi-agent routing
- Evaluator node
- Retry logic
- Human-in-the-Loop fallback
- Action trace logging

**Infrastructure / Dev Tools**
- NVIDIA NIM endpoints
- HTTPX
- dotenv
- Requests
- Git / VS Code / Python virtual environment

**Observability Tool**
- MlFlow
""")

    if not ingested_docs:
        st.info(
            "⚠️ No documents ingested yet. "
            "The **Web Search Agent & SQL Agent** are still available for questions. "
            "Upload a PDF from the sidebar to also enable the **PDF RAG Agent**."
        )


# ── 5. HITL — SQL APPROVAL UI ──────────────────────────────────────────────────

if st.session_state.hitl_pending_state is not None:
    pending = st.session_state.hitl_pending_state
    generated_sql = pending.get("generated_sql", "")

    st.markdown("⏸️ **Agent paused — SQL query needs your review before running**")

    with st.container(border=True):
        st.markdown("### 🗃️ SQL Query Ready for Review")
        st.caption(
            "The SQL Agent generated the query below based on your question. "
            "Please review it carefully before it runs on your PostgreSQL database."
        )

        st.code(generated_sql, language="sql")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Approve & Run", use_container_width=True, type="primary"):
                with st.spinner("Running approved SQL and generating answer..."):
                    result = ask_question(
                        question=pending["question"],
                        thread_id=st.session_state.thread_id,
                        human_approved_sql=generated_sql
                    )

                st.session_state.hitl_pending_state = None
                st.session_state.show_sql_editor = False

                if result.get("hitl_needed"):
                    st.session_state.evaluator_hitl_pending_state = result
                else:
                    _handle_final_result(result)

                st.rerun()

        with col2:
            if st.button("✏️ Edit SQL", use_container_width=True):
                st.session_state.show_sql_editor = True

        with col3:
            if st.button("❌ Reject", use_container_width=True):
                st.session_state.hitl_pending_state = None
                st.session_state.show_sql_editor = False
                st.info("❌ SQL rejected. Please rephrase your question below.")
                st.rerun()

        if st.session_state.show_sql_editor:
            st.markdown("**✏️ Modify the SQL below then click Run:**")
            edited_sql = st.text_area(
                "Edit SQL Query",
                value=generated_sql,
                height=150,
                key="sql_editor_input",
                label_visibility="collapsed"
            )

            if st.button("▶️ Run Edited SQL", type="primary"):
                with st.spinner("Running edited SQL and generating answer..."):
                    result = ask_question(
                        question=pending["question"],
                        thread_id=st.session_state.thread_id,
                        human_approved_sql=edited_sql
                    )

                st.session_state.hitl_pending_state = None
                st.session_state.show_sql_editor = False

                if result.get("hitl_needed"):
                    st.session_state.evaluator_hitl_pending_state = result
                else:
                    _handle_final_result(result)

                st.rerun()


# ── 6. HITL — EVALUATOR REVIEW UI ─────────────────────────────────────────────

if st.session_state.evaluator_hitl_pending_state is not None:
    pending = st.session_state.evaluator_hitl_pending_state

    st.markdown("🧑‍⚖️ **Human-in-the-Loop required — evaluator could not approve this answer after retries**")

    with st.container(border=True):
        st.markdown("### Review Answer")
        st.caption(
            "The evaluator could not approve this answer after retries. "
            "You can accept the current answer anyway or ask the system again with a better question."
        )

        st.markdown("**Current answer:**")
        st.markdown(_safe_answer_text(pending))

        evaluator_reason = pending.get("evaluator_reason", "")
        if evaluator_reason:
            st.caption(f"Evaluator reason: {evaluator_reason}")

        hitl_reason = pending.get("hitl_reason", "")
        if hitl_reason:
            st.caption(hitl_reason)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Accept anyway", type="primary", use_container_width=True):
                _handle_final_result(pending)
                st.session_state.evaluator_hitl_pending_state = None
                st.rerun()

        with col2:
            if st.button("🔁 Ask again", use_container_width=True):
                st.session_state.evaluator_hitl_pending_state = None
                st.info("Please rephrase or ask your question again below.")
                st.rerun()


# ── 7. CHAT HISTORY RENDERING ──────────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message.get("content", ""))

        if message["role"] == "assistant":
            _render_assistant_details(
                agent_used=message.get("agent_used", "rag"),
                chunks=message.get("context", []),
                web_results=message.get("web_results", []),
                generated_sql=message.get("generated_sql", ""),
                execution_trace=message.get("execution_trace", []),
                retry_count=message.get("retry_count", 0),
                evaluator_reason=message.get("evaluator_reason", ""),
            )


# ── 8. CHAT INPUT + NEW QUERY FLOW ─────────────────────────────────────────────

chat_blocked = (
    st.session_state.hitl_pending_state is not None
    or st.session_state.evaluator_hitl_pending_state is not None
)

if not chat_blocked:
    if prompt := st.chat_input("Ask anything..."):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            spinner_placeholder = st.empty()
            answer_placeholder = st.empty()

            try:
                with spinner_placeholder.container():
                    with st.spinner("🤖 Agent is working..."):
                        chat_history = build_chat_history_for_workflow(st.session_state.messages)

                        result = ask_question(
                            question=prompt,
                            thread_id=st.session_state.thread_id,
                            chat_history=chat_history,
                        )

                spinner_placeholder.empty()

                if result.get("awaiting_human_approval"):
                    st.session_state.hitl_pending_state = result
                    st.rerun()

                if result.get("hitl_needed"):
                    st.session_state.evaluator_hitl_pending_state = result
                    st.rerun()

                answer = _safe_answer_text(result)
                chunks = result.get("retrieved_chunks", []) or []
                agent_used = result.get("agent_used", "rag")
                web_results = result.get("web_results", []) or []
                generated_sql = result.get("generated_sql", "") or ""
                retry_count = result.get("retry_count", 0) or 0
                evaluator_reason = result.get("evaluator_reason", "") or ""
                execution_trace = deduplicate_trace(result.get("execution_trace", []) or [])

                answer_placeholder.markdown(answer)

                _render_assistant_details(
                    agent_used=agent_used,
                    chunks=chunks,
                    web_results=web_results,
                    generated_sql=generated_sql,
                    execution_trace=execution_trace,
                    retry_count=retry_count,
                    evaluator_reason=evaluator_reason,
                )

                _handle_final_result(result)

            except Exception as e:
                import traceback
                spinner_placeholder.empty()
                st.error(f"Error: {repr(e)}")
                st.code(traceback.format_exc())

else:
    if st.session_state.hitl_pending_state is not None:
        st.info("💬 Chat input is paused while the SQL review above is pending.")
    elif st.session_state.evaluator_hitl_pending_state is not None:
        st.info("💬 Chat input is paused while the Human-in-the-Loop review above is pending.")