# ============================================================
# agents/web_search_agent.py
# ============================================================
# Web Search Agent — answers questions using live internet search.
#
# This agent has TWO nodes:
#
# Node 1: tavily_node
# - calls the Tavily Search API to get live web results
# - stores those results in state["web_results"]
#
# Node 2: tavily_generate_node
# - takes those web results
# - builds a prompt with the web content as context
# - asks the shared generation LLM to produce a grounded answer
#
# HOW IT CONNECTS TO THE WORKFLOW:
# agentic_workflow.py imports tavily_node and tavily_generate_node
# and registers them as "web_search" and "tavily_generate" in the graph.
#
# SSL NOTE:
# Tavily API is called with verify=False because the current environment
# is behind a corporate proxy with a self-signed certificate chain.
# This is a dev workaround only.
#
# TO MODIFY THIS AGENT:
# - Change max_results (currently 5)
# - Change search_depth ("basic" for faster, "advanced" for better results)
# - Change the generation prompt style
#
# TO ADD A FUTURE AGENT (e.g. SQL, Wikipedia):
# - Create agents/sql_agent.py with search_node + generate_node
# - Import those nodes in agentic_workflow.py
# - Register them as new nodes in the graph
# - Add a new conditional edge mapping entry
# - Update the router prompt to mention the new tool
# ============================================================

import os
import requests
from agents.shared import AgentState, generation_llm
from langchain_core.prompts import PromptTemplate

WEB_SEARCH_PROMPT = PromptTemplate.from_file("prompts/web_search_prompt.txt")

def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": os.getenv("TAVILY_API_KEY"),
        "query": query,
        "search_depth": "basic",
        "max_results": max_results
    }
    response = requests.post(url, json=payload, timeout=60, verify=False)
    response.raise_for_status()
    return response.json().get("results", [])

def tavily_node(state: AgentState) -> AgentState:
    question = state.get("standalonequestion") or state["question"]
    retry_count = state.get("retry_count", 0)

    node_steps = [
        {
            "trace_id": f"web-search-start-{retry_count}",
            "icon": "🌐",
            "step": "Web Search Agent → Searching Tavily",
            "detail": f"Searching the live internet for: {question}"
        }
    ]

    web_results = tavily_search(question, max_results=5)

    node_steps.append({
        "trace_id": f"web-search-done-{retry_count}",
        "icon": "📰",
        "step": f"Web Search Agent → Retrieved {len(web_results)} results",
        "detail": "Live web results were collected and passed to answer generation."
    })

    return {
        **state,
        "web_results": web_results,
        "execution_trace": list(state.get("execution_trace", [])) + node_steps,
        "new_trace_steps": node_steps
    }

def tavily_generate_node(state: AgentState) -> AgentState:
    question = state.get("standalonequestion") or state["question"]
    web_results = state.get("web_results", [])
    retry_count = state.get("retry_count", 0)

    node_steps = []

    if not web_results:
        node_steps.append({
            "trace_id": f"web-generate-empty-{retry_count}",
            "icon": "⚠️",
            "step": "Web Search Agent → No web results found",
            "detail": "Tavily did not return enough relevant results."
        })
        return {
            **state,
            "answer": "I could not find relevant information from web search.",
            "execution_trace": list(state.get("execution_trace", [])) + node_steps,
            "new_trace_steps": node_steps
        }

    node_steps.append({
        "trace_id": f"web-generate-start-{retry_count}",
        "icon": "✍️",
        "step": "Web Search Agent → Generating answer",
        "detail": "Using live web results to generate a grounded response."
    })

    context_parts = []
    for i, result in enumerate(web_results[:5], 1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        content = result.get("content", "")
        context_parts.append(f"[Source {i}: {title}]\nURL: {url}\n{content}")
    context = "\n\n---\n\n".join(context_parts)

    retry_feedback_block = ""
    if state.get("retry_instruction") and state.get("retry_instruction", "").lower() != "none":
        retry_feedback_block = f"""
The evaluator asked for an improved retry.
Improve the answer using this instruction:
{state["retry_instruction"]}
"""

    prompt = WEB_SEARCH_PROMPT.format(
        retry_feedback_block=retry_feedback_block,
        context=context,
        question=question
    )

    # Use the shared generation LLM here because this node is responsible
    # for synthesizing retrieved web context into the final grounded answer.
    answer = generation_llm.invoke(prompt)

    return {
        **state,
        "answer": answer,
        "execution_trace": list(state.get("execution_trace", [])) + node_steps,
        "new_trace_steps": node_steps
    }