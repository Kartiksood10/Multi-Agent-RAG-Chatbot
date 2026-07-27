# This file defines the RAG pipeline using LangGraph.
# It contains two graph nodes:
#   1. retrieve_node  → embeds the query and fetches similar chunks from PostgreSQL
#   2. generate_node  → sends retrieved chunks as context to Llama 3.2 via Ollama
# LangGraph wires these two nodes into a stateful graph:
# retrieve → generate → END

# Langchain is used to call local LLM from Ollama

import os
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any  # For defining the shape of our graph state


# LangGraph — used to define the RAG workflow as a stateful directed graph
from langgraph.graph import StateGraph, END


# LangChain's Ollama integration — allows us to call a locally running Ollama model
from langchain_ollama import OllamaLLM


# OpenAI-compatible client for calling NVIDIA NIM embedding API
from openai import OpenAI


# psycopg2 — PostgreSQL driver for Python
import psycopg2

# To allow access to Nvidia embedding API
import httpx


# register_vector — teaches psycopg2 how to handle pgvector's vector type
from pgvector.psycopg2 import register_vector


# Load environment variables from .env
load_dotenv()


# NVIDIA NIM client
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    http_client=httpx.Client(verify=False, timeout=60.0),
)


# Local LLM via Ollama
# temperature=0.1 keeps the model focused and factual (low creativity)
llm = OllamaLLM(
    model="llama3.2:3b",
    temperature=0.1
)


# RAG Graph State
# Shared dictionary passed between nodes — each node reads and writes to this
class RAGState(TypedDict):
    question: str
    retrieved_chunks: List[Dict[str, Any]]
    answer: str


# Database connection helper
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    register_vector(conn)
    return conn


# Query embedding helper
def get_query_embedding(question: str) -> list[float]:
    """
    Converts the user's question into an embedding vector using the same
    NVIDIA embedding model used during ingestion.

    input_type="query" tells the model this is a search query,
    not a stored passage chunk.
    """
    response = nvidia_client.embeddings.create(
        input=[question],
        model="nvidia/nv-embedqa-e5-v5",
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "END"},
    )
    return response.data[0].embedding


# Node 1: Retrieve
def retrieve_node(state: RAGState) -> RAGState:
    """
    Embeds the user's question using NVIDIA NIM (input_type="query")
    then runs cosine similarity search in PostgreSQL to find top matching chunks.
    """
    question = state["question"]
    query_embedding = get_query_embedding(question)

    conn = get_db_connection()
    cur = conn.cursor()

    # <=> is pgvector's cosine distance operator
    # 1 - distance = cosine similarity
    # We fetch top 8 first, then drop weak matches using a similarity threshold
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

        # Debug logging in terminal so we can inspect what retrieval is actually returning
        print(f"[RETRIEVE] doc={doc_name}, chunk={chunk_index}, similarity={similarity:.4f}")
        print(chunk_text[:300])
        print("-" * 100)

        if similarity is None:
            continue

        # Filter out weak matches
        # This prevents irrelevant chunks from being passed to the LLM as context
        if similarity < 0.45:
            continue

        retrieved_chunks.append({
            "doc_name": doc_name,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "chunk_length": chunk_length,
            "similarity": float(similarity),
        })

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "answer": ""
    }


# Node 2: Generate
def generate_node(state: RAGState) -> RAGState:
    """
    Builds a prompt using retrieved chunks as context, then asks
    Llama 3.2 via Ollama to generate a grounded answer.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]

    if not chunks:
        return {
            **state,
            "answer": "I could not find relevant information in the uploaded documents."
        }

    # Join top chunks into a structured context block
    # Including document name, chunk index, and similarity is useful for transparency/debugging
    context_parts = []
    for item in chunks[:5]:
        context_parts.append(
            f"[Document: {item['doc_name']} | Chunk: {item['chunk_index']} | Similarity: {item['similarity']:.4f}]\n"
            f"{item['chunk_text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Prompt strictly instructs model to use ONLY the provided context
    # Also allows the model to answer partially if the document contains partial information
    prompt = f"""
You are a helpful assistant for question answering over uploaded documents.

Answer the user's question using ONLY the context provided below.

Rules:
1. If the answer is clearly present in the context, answer directly and simply.
2. If the answer is partially present, answer using only what is available in the context.
3. If the answer is not present in the context, say:
"I don't have enough information in the document to answer this."

Do not use outside knowledge.
Do not invent facts.
Be concise, accurate, and grounded in the retrieved text.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    answer = llm.invoke(prompt)

    return {
        **state,
        "answer": answer
    }


# Build the LangGraph
def build_rag_graph():
    """
    Builds the graph: [START] → retrieve → generate → [END]
    """
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")          # retrieve runs first
    graph.add_edge("retrieve", "generate")     # then generate
    graph.add_edge("generate", END)            # then end

    return graph.compile()


# Compiled graph — imported and called by app.py
rag_app = build_rag_graph()


# Public interface
def ask_question(question: str) -> dict:
    """Entry point for app.py — runs the full RAG pipeline and returns result."""
    return rag_app.invoke({
        "question": question,
        "retrieved_chunks": [],
        "answer": ""
    })