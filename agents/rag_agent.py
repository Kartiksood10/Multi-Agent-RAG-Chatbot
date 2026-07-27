# ============================================================
# agents/rag_agent.py
# ============================================================
# PDF RAG Agent — answers questions from uploaded PDF/DOCX files.
#
# This agent has TWO nodes:
#
# Node 1: retrieve_node
# - converts the user's question to an embedding vector
# - runs cosine similarity search in PostgreSQL (pgvector)
# - returns the top matching text chunks
#
# Node 2: generate_node
# - takes those retrieved chunks
# - builds a strict prompt
# - asks the shared generation LLM to answer using ONLY the retrieved context
#
# HOW IT CONNECTS TO THE WORKFLOW:
# agentic_workflow.py imports retrieve_node and generate_node
# and registers them as "retrieve" and "generate" in the graph.
#
# TO MODIFY THIS AGENT:
# - Change the cosine similarity threshold (currently 0.45)
# - Change the number of chunks returned (currently LIMIT 8, top 3 trimmed chunks used for prompt generation)
# - Change the generation prompt
# ============================================================

from agents.shared import AgentState, generation_llm, get_db_connection, get_query_embedding
from langchain_core.prompts import PromptTemplate

# Load Prompt template from prompts directory
RAG_PROMPT = PromptTemplate.from_file("prompts/rag_prompt.txt")

def retrieve_node(state: AgentState) -> AgentState:
    question = state.get("standalonequestion") or state["question"]
    retry_count = state.get("retry_count", 0)

    # Each node emits ONLY its own steps.
    node_steps = [
        {
            "trace_id": f"rag-retrieve-start-{retry_count}",
            "icon": "📄",
            "step": "RAG Agent → Retrieving chunks",
            "detail": "Embedding the question and searching PostgreSQL vector store."
        }
    ]

    # Convert user question into vector embeddings
    query_embedding = get_query_embedding(question)

    conn = get_db_connection()
    cur = conn.cursor()

    # Find Top 8 Chunks through cosine similarity
    # <=> indicates cosine distance between the embedding in the DB and the user's question embedding
    # We convert cosine distance to similarity by doing subtraction of 1 - cosine distance
    # If cosine distance = 0.1 (close and relevant) then cosine similarity is 0.90
    cur.execute(
        """
        SELECT
            doc_name,
            chunk_index,
            chunk_text,
            chunk_length,
            1 - (embedding <=> %s::vector) AS similarity
        FROM document_chunks
        WHERE chunk_length IS NOT NULL
          AND chunk_length >= 80
        ORDER BY embedding <=> %s::vector
        LIMIT 8
        """,
        (query_embedding, query_embedding),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    retrieved_chunks = []
    for row in rows:
        doc_name, chunk_index, chunk_text, chunk_length, similarity = row
        if similarity is None:
            continue
        if similarity < 0.45:
            continue
        retrieved_chunks.append({
            "doc_name": doc_name,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "chunk_length": chunk_length,
            "similarity": float(similarity),
        })

    node_steps.append({
        "trace_id": f"rag-retrieve-done-{retry_count}",
        "icon": "📎",
        "step": f"RAG Agent → Retrieved {len(retrieved_chunks)} relevant chunks",
        "detail": "Relevant chunks were selected and passed to answer generation."
    })

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "answer": "",
        "agent_used": state.get("agent_used", "rag"),
        "web_results": state.get("web_results", []),
        "sql_result": state.get("sql_result", []),
        "generated_sql": state.get("generated_sql", ""),
        "retry_count": retry_count,
        "evaluator_feedback": state.get("evaluator_feedback", ""),
        "evaluation_result": state.get("evaluation_result", ""),
        "execution_trace": list(state.get("execution_trace", [])) + node_steps,
        "new_trace_steps": node_steps,
        "router_reason": state.get("router_reason", ""),
        "evaluator_reason": state.get("evaluator_reason", ""),
        "retry_instruction": state.get("retry_instruction", "")
    }

def generate_node(state: AgentState) -> AgentState:
    question = state.get("standalonequestion") or state["question"]
    chunks = state["retrieved_chunks"]
    retry_count = state.get("retry_count", 0)

    node_steps = []

    if not chunks:
        node_steps.append({
            "trace_id": f"rag-generate-empty-{retry_count}",
            "icon": "⚠️",
            "step": "RAG Agent → No relevant chunks found",
            "detail": "The vector search did not find strong matching document content."
        })
        return {
            **state,
            "answer": "I could not find relevant information in the uploaded documents.",
            "execution_trace": list(state.get("execution_trace", [])) + node_steps,
            "new_trace_steps": node_steps
        }

    node_steps.append({
        "trace_id": f"rag-generate-start-{retry_count}",
        "icon": "✍️",
        "step": "RAG Agent → Generating answer",
        "detail": "Using retrieved document chunks to produce a grounded answer."
    })

    context_parts = []

    # Use only the top 3 chunks for prompt construction and trim each chunk
    # to 800 characters. This reduces prompt size and improves happy path
    # speed while keeping the same RAG retrieval and answering logic.
    for item in chunks[:3]:
        trimmed_chunk_text = item["chunk_text"][:800]
        context_parts.append(
            f"[Document: {item['doc_name']} | Chunk: {item['chunk_index']} | Similarity: {item['similarity']:.4f}]\n"
            f"{trimmed_chunk_text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    retry_feedback_block = ""
    if state.get("retry_instruction") and state.get("retry_instruction", "").lower() != "none":
        retry_feedback_block = f"""
The evaluator asked for an improved retry.
Improve the answer using this instruction:
{state["retry_instruction"]}
"""

    # Inject actual values into Prompt template
    prompt = RAG_PROMPT.format(
        retry_feedback_block=retry_feedback_block,
        context=context,
        question=question
    )

    answer = generation_llm.invoke(prompt)

    return {
        **state,
        "answer": answer,
        "execution_trace": list(state.get("execution_trace", [])) + node_steps,
        "new_trace_steps": node_steps
    }