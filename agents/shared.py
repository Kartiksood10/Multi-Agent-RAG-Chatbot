# ============================================================
# agents/shared.py
# ============================================================
# Shared infrastructure used by ALL agents in the workflow.
#
# What lives here:
# - AgentState → the shared LangGraph state schema
# - routing_evaluation_llm → NVIDIA NIM hosted open-source LLM used for router and evaluator
# - generation_llm → NVIDIA NIM hosted open-source LLM used for answer generation
# - nvidia_client → NVIDIA NIM embedding API client
# - get_db_connection → PostgreSQL connection helper
# - get_query_embedding → converts question to vector
# - has_ingested_documents → checks if any PDFs are in DB
#
# WHY SHARED:
# Without this file, every agent would need to re-initialise
# the LLMs, DB connection, and embedding client separately.
# Centralising them here avoids duplication and makes it
# easy to swap models or credentials in one place.
#
# HOW TO ADD A NEW AGENT:
# Any new agent file only needs to:
# from agents.shared import AgentState, generation_llm, get_db_connection, ...
# ============================================================

import os
import psycopg2
import httpx
import urllib3

from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pgvector.psycopg2 import register_vector

load_dotenv()

# Disable insecure HTTPS warnings caused by verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================
# SHARED STATE SCHEMA
# ============================================================
class AgentState(TypedDict):
    question: str
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    agent_used: str
    web_results: List[Dict[str, Any]]
    sql_result: List[Dict[str, Any]]
    generated_sql: str

    retry_count: int
    evaluator_feedback: str
    evaluation_result: str

    # full accumulated trace for final state
    execution_trace: List[Dict[str, str]]

    # ONLY steps produced by the current node
    new_trace_steps: List[Dict[str, str]]

    # UI rendering of router reasoning ReAct part
    router_reason: str
    evaluator_reason: str
    retry_instruction: str

    # Human-on-the-loop for SQL
    awaiting_human_approval: bool
    human_approved_sql: str

    # evaluator scoring
    evaluator_score: int

    # Human-in-the-Loop after retries fail
    hitl_needed: bool
    hitl_reason: str

    # raw recent conversation messages from Streamlit UI
    chat_history: List[Dict[str, str]]

    # question after follow-up resolution
    standalone_question: str

    # whether rewrite was applied
    question_rewritten: bool

    # short reason for rewrite / no rewrite
    rewriter_reason: str

    # agent used in prev query
    last_agent_used: str

# ============================================================
# HOW THIS NVIDIA NIM WRAPPER WORKS
# ============================================================
# We use LangChain's ChatOpenAI client because NVIDIA NIM exposes
# an OpenAI-compatible chat API.
#
# Official docs:
# - ChatOpenAI: https://docs.langchain.com/oss/python/integrations/chat/openai
# - LangChain messages: https://docs.langchain.com/oss/python/langchain/messages
# - NVIDIA NIM LLM API: https://docs.nvidia.com/nim/large-language-models/latest/api-reference.html
#
# INPUT FLOW
# ----------
# The rest of our app uses a simple pattern:
#
#     result = some_llm.invoke(prompt_string)
#
# But ChatOpenAI is a chat model, so internally it expects messages,
# not just a raw string.
#
# In our custom invoke():
#
#     response = self._client.invoke([HumanMessage(content=prompt)])
#
# this means:
# - prompt is a plain Python string
# - HumanMessage(content=prompt) marks that string as a user message
# - [ ... ] makes it a message list for the chat API
#
# What gets sent to NVIDIA NIM is conceptually like:
#
# {
#   "model": "<model_name>",
#   "messages": [
#     {"role": "user", "content": "<prompt text>"}
#   ],
#   "temperature": 0.1
# }
#
# OUTPUT FLOW
# -----------
# ChatOpenAI returns an AIMessage-style response object, not a plain string.
# The generated text is available in:
#
#     response.content
#
# So our wrapper returns only response.content to keep all existing
# agent code unchanged.
#
# Example:
#
#     raw_response = self._client.invoke([HumanMessage(content=prompt)])
#     text_only = raw_response.content
#
# WHY THIS WRAPPER EXISTS
# -----------------------
# This wrapper hides chat-message formatting from the rest of the code.
# It converts:
# - plain string prompt -> HumanMessage list
# - AIMessage response  -> plain string content
#
# That lets router, evaluator, RAG, web, and SQL agents all keep using:
#
#     llm.invoke(prompt)
#
# instead of dealing with message objects or response parsing manually.
# ============================================================
class NIMChatLLM:
    def __init__(self, model: str, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

        # Create one reusable ChatOpenAI client per model.
        # NVIDIA NIM exposes an OpenAI-compatible chat API,
        # so ChatOpenAI can be pointed to the NVIDIA base URL.
        self._client = ChatOpenAI(
            model=model,
            openai_api_key=os.getenv("NVIDIA_API_KEY"),
            openai_api_base="https://integrate.api.nvidia.com/v1",
            temperature=temperature,
            http_client=httpx.Client(verify=False, timeout=60.0),
        )

    def invoke(self, prompt: str) -> str:
        # Convert the plain prompt string into a chat message list
        # because NVIDIA NIM chat models accept messages, not a raw string.
        response = self._client.invoke([HumanMessage(content=prompt)])

        # Return only the text content so all existing agent code
        # continues to receive a normal string response.
        return response.content

# ============================================================
# LLMs — shared across all agents
# ============================================================
# routing_evaluation_llm:
# Fast open-source model used for lightweight classification tasks
# such as routing and evaluation, where latency matters most.
routing_evaluation_llm = NIMChatLLM(
    model="meta/llama-3.1-8b-instruct",
    temperature=0.1
)

# generation_llm:
# Stronger open-source model used for answer generation across
# RAG, web search, and SQL answer synthesis.
generation_llm = NIMChatLLM(
    model="openai/gpt-oss-20b",
    temperature=0.1
)

# LLM example without wrapper
# llm = ChatOpenAI(
#     model="meta/llama-3.1-8b-instruct",
#     openai_api_key=os.getenv("NVIDIA_API_KEY"),
#     openai_api_base="https://integrate.api.nvidia.com/v1",
#     temperature=0.1,
#     http_client=httpx.Client(verify=False, timeout=60.0),
# )

# response = llm.invoke([HumanMessage(content="Explain vector databases simply.")])
# print(response.content)

# ============================================================

# LLM invocation using ollama
# from langchain_ollama import OllamaLLM

# llm = OllamaLLM(model="llama3.2:3b")

# response = llm.invoke("Explain vector databases.")

# print(response)

# ============================================================
# NVIDIA NIM embedding client
# ============================================================
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    http_client=httpx.Client(verify=False, timeout=60.0),
)

# ============================================================
# DATABASE CONNECTION HELPER
# ============================================================
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

# ============================================================
# DOCUMENT PRESENCE CHECK
# ============================================================
# Cache the existence check so the router does not open a fresh
# PostgreSQL connection on every single question and retry.
# This improves the happy path speed while keeping the same logic.
_docs_cache = None

def has_ingested_documents() -> bool:
    global _docs_cache

    # If the value was already checked once in this process,
    # return the cached result immediately.
    if _docs_cache is not None:
        return _docs_cache

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT EXISTS (SELECT 1 FROM document_chunks LIMIT 1)")
        result = cur.fetchone()
        _docs_cache = bool(result[0]) if result else False
        return _docs_cache
    finally:
        cur.close()
        conn.close()

def invalidate_docs_cache():
    # Reset the cache when a new document is ingested so future
    # router checks see the latest DB state.
    global _docs_cache
    _docs_cache = None

# ============================================================
# QUERY EMBEDDING HELPER
# ============================================================
def get_query_embedding(question: str) -> list[float]:
    response = nvidia_client.embeddings.create(
        input=[question],
        model="nvidia/nv-embedqa-e5-v5",
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "END"},
    )
    return response.data[0].embedding