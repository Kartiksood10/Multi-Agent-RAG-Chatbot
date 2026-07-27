# 🤖 Multi-Agent RAG Chatbot

A Multi-agent RAG chatbot built with LangChain, LangGraph, and NVIDIA NIM, featuring router-based orchestration, LLM-as-a-judge evaluation, and human oversight for safe and grounded responses. 

## 🧠 Architecture

- **Router Agent**
  - Sends each question to:
    - **PDF Agent** (uploaded docs / PDFs)
    - **Web Search Agent** (live external info)
    - **SQL Agent** (structured DB questions)

- **Specialist Agents**
  - **PDF Agent**: Embeds queries, retrieves chunks from PostgreSQL via `pgvector` + cosine similarity, and answers grounded in documents.
  - **Web Search Agent**: Uses Tavily API for live search and generates answers grounded in web results.
  - **SQL Agent**: Generates `SELECT` queries, requires human approval (HOTL), executes on PostgreSQL, and answers from query results.

- **Evaluator Agent (LLM-as-a-Judge)**
  - Uses Meta Llama 3.1 8B to evaluate answers produced by OpenAI GPT-OSS 20B.
  - Checks relevance, grounding, and quality.
  - Can trigger up to **2 retries**; otherwise escalates to Human-in-the-Loop (HITL).

- **Human Oversight**
  - **HITL**: Final gate when the Evaluator still rejects answers; user can rephrase or accept.
  - **HOTL**: Every SQL query must be Approved / Edited / Rejected before execution.

- **Conversation Memory**
  - History Rewrite node turns follow-ups into standalone questions for better routing and evaluation.

- **Reasoning Trace**
  - UI shows which agent ran, what actions were taken, and why the Evaluator accepted or retried.
 
- **MLflow Visibility**
  - MLflow LangChain autologging captures requests, responses, tokens, latency, and traces for each run, with additional custom metrics logged per workflow for observability and debugging.

## 🛠️ Tech Stack

- **Core**
  - Python 3.12.10  
  - Streamlit  
  - LangChain  
  - LangGraph 

- **LLM / Embeddings**
  - NVIDIA NIM (LLM + embeddings)  
  - Meta Llama 3.1 8B Instruct (`meta/llama-3.1-8b-instruct`) for routing + evaluation  
  - OpenAI GPT-OSS 20B (`openai/gpt-oss-20b`) for generation  
  - NVIDIA embedding model `nvidia/nv-embedqa-e5-v5`

- **RAG / Retrieval**
  - PostgreSQL  
  - `pgvector`  
  - Cosine similarity over document chunks

- **Web / SQL**
  - Tavily API for web search  
  - PostgreSQL for SQL answers with human-approved queries

- **Infrastructure / Tools**
  - NVIDIA NIM endpoints  
  - `httpx`, `requests`, `python-dotenv`  
  - Git, VS Code, Python virtual envs

- **Observability**
  - MLflow with LangChain autologging and custom metrics (requests, responses, tokens, latency, retries)

## 🗺️ LangGraph Diagram

The UI includes a LangGraph Mermaid diagram to visualize the end-to-end workflow (Router → Agents → Evaluator → HITL/HOTL) and action trace.
