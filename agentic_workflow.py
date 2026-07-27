# ============================================================
# agentic_workflow.py
# ============================================================
# This is the CENTRAL ORCHESTRATION file for a multi-agent
# Agentic AI workflow built on LangGraph.
#
# WHY THIS IS "AGENTIC AI":
# ─────────────────────────────────────────────────────────────
# Unlike a single LLM call that just answers a prompt, this
# system behaves like a team of autonomous specialist agents
# that reason, decide, act, and self-correct:
#
#   • Router Agent      ->  Perceives the user's intent and
#                           autonomously DECIDES which tool/
#                           specialist agent (RAG, SQL, or Web
#                           Search) is best suited to answer.
#   • RAG Agent          -> Retrieves relevant document chunks
#                           from a vector store and generates
#                           a grounded answer.
#   • Web Search Agent   -> Uses the Tavily search tool to
#                           fetch live external information and
#                           generates an answer from it.
#   • SQL Agent          -> Generates a SQL query from natural
#                           language, executes it against a
#                           database, and produces a
#                           natural-language answer.
#   • Evaluator Agent     -> Acts as a self-reflection/critique
#                           loop: it JUDGES the quality of the
#                           generated answer and decides whether
#                           to PASS (end) or RETRY (send the
#                           question back to the Router with
#                           feedback for another attempt).
#   • History Rewrite     -> A lightweight reasoning step that
#     Agent                 resolves ambiguous follow-up
#                           questions ("explain more about
#                           that") into standalone questions
#                           using conversation memory, before
#                           routing.
#
# This creates a closed feedback loop (Router -> Agent ->
# Evaluator -> Router...) where agents plan, act, observe
# results, and self-correct autonomously without hardcoded
# control flow -- the hallmark of agentic behavior.
#
# HUMAN-ON-THE-LOOP (HOTL):
# ─────────────────────────────────────────────────────────────
# The SQL branch pauses execution BEFORE executing a generated
# SQL query (via LangGraph's `interrupt_before`), requiring
# explicit human approval before the query runs against the
# database. State is persisted via a MemorySaver checkpointer
# so the paused graph can be safely resumed later with the
# approved query.
#
# WHAT THIS FILE DOES (MECHANICS):
# ─────────────────────────────────────────────────────────────
# 1. Defines the Router Agent (router_node + route_decision)
#    and the History Rewrite Agent (history_rewrite_node)
# 2. Imports specialist agent nodes from their individual
#    agent files (rag_agent, web_search_agent, sql_agent,
#    evaluator_agent)
# 3. Wires everything into a single LangGraph StateGraph,
#    including conditional edges for routing and evaluator
#    retry decisions
# 4. Compiles the graph with a checkpointer + HOTL interrupt,
#    and exposes ask_question()/stream_question() to app.py
#
# HOW TO ADD A NEW AGENT:
# ─────────────────────────────────────────────────────────────
# Step 1: Create agents/your_agent.py with your node functions
# Step 2: Add your output fields to AgentState in agents/shared.py
# Step 3: Import your node functions below
# Step 4: Register them with graph.add_node(...)
# Step 5: Add fixed edges: graph.add_edge(...)
# Step 6: Add your agent's key to the conditional edge mapping
# Step 7: Update the router prompt to mention your new tool
# ============================================================


import urllib3
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END


from langgraph.checkpoint.memory import MemorySaver


from agents.shared import AgentState, routing_evaluation_llm, has_ingested_documents
from agents.rag_agent import retrieve_node, generate_node
from agents.web_search_agent import tavily_node, tavily_generate_node
from agents.sql_agent import (
    sql_generate_query_node,
    sql_execute_query_node,
    sql_generate_answer_node
)
from agents.evaluator_agent import evaluator_node


from langchain_core.prompts import PromptTemplate

import time
import mlflow

ROUTER_PROMPT = PromptTemplate.from_file("prompts/router_prompt.txt")
HISTORY_REWRITE_PROMPT = PromptTemplate.from_file("prompts/history_rewrite_prompt.txt")


load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Extract the decision and reason string values from the output of the LLM
def _parse_router_output(raw_text: str) -> tuple[str, str]:
    text = str(raw_text).strip()
    text_lower = text.lower()


    decision = "web_search"
    reason = "Defaulted to web search."


    for line in text.splitlines():
        lower_line = line.lower().strip()
        if lower_line.startswith("decision:"):
            value = line.split(":", 1)[1].strip().lower()
            if value in ("rag", "web_search", "sql"):
                decision = value
        elif lower_line.startswith("reason:"):
            extracted = line.split(":", 1)[1].strip()
            if extracted:
                reason = extracted


    if "decision:" not in text_lower:
        if "sql" in text_lower or "database" in text_lower or "query" in text_lower:
            decision = "sql"
        elif "web_search" in text_lower or "web" in text_lower or "search" in text_lower:
            decision = "web_search"
        elif "rag" in text_lower or "document" in text_lower or "pdf" in text_lower:
            decision = "rag"


    return decision, reason


# Method to extract rewrite reason, rewrite needed and new question from LLM
# For follow up questions cases such as "explain more about that"
def parse_history_rewrite_output(raw_text: str) -> tuple[bool, str, str]:
    """
    Parse the history-aware rewriter output.

    Returns:
        (rewrite_needed, standalone_question, reason)
    """
    rewrite_needed = False
    standalone_question = ""
    reason = "No reason provided."

    for line in str(raw_text).strip().splitlines():
        lower_line = line.lower().strip()

        if lower_line.startswith("rewrite_needed:"):
            value = line.split(":", 1)[1].strip().lower()
            rewrite_needed = value == "yes"

        elif lower_line.startswith("standalone_question:"):
            standalone_question = line.split(":", 1)[1].strip()

        elif lower_line.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    if not standalone_question:
        standalone_question = str(raw_text).strip()

    return rewrite_needed, standalone_question, reason


# UI chat to plain text to rewrite new question to LLM
def format_chat_history(chat_history: list[dict]) -> str:
    """
    Convert UI chat history into plain text for the rewrite prompt.

    Only recent turns are needed.
    """
    if not chat_history:
        return "No prior conversation."

    recent_messages = chat_history[-6:]  # last 6 messages = 3 turns approx
    lines = []

    for msg in recent_messages:
        role = msg.get("role", "user").capitalize()
        content = str(msg.get("content", "")).strip()
        agent_used = msg.get("agent_used", "")

        if role.lower() == "assistant" and agent_used:
            lines.append(f"{role} ({agent_used}): {content}")
        else:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


# Before routing, check if the question is a follow up or a new question
# If it is a follow question, pass the follow up question and chat history to LLM
# LLM then reframes the question based on chat history and sends it to router to answer
def history_rewrite_node(state: AgentState) -> AgentState:
    """
    Rewrites follow-up questions into standalone questions before routing.

    Why:
    A question like "give an example for that" is too ambiguous on its own.
    We resolve it using recent conversation so the router gets a better query.
    """
    question = state["question"]
    chat_history = state.get("chat_history", []) or []
    existing_trace = list(state.get("execution_trace", []))
    new_trace_steps = []

    last_agent_used = ""
    for msg in reversed(chat_history):
        if msg.get("role", "").lower() == "assistant" and msg.get("agent_used"):
            last_agent_used = msg["agent_used"]
            break

    # If there is no history, skip rewriting
    if not chat_history:
        new_trace_steps.append({
            "trace_id": "history-rewrite-skip-no-history",
            "icon": "🧠",
            "step": "History Rewrite Skipped",
            "detail": "No prior chat history was available, so the original question was used.",
        })
        return {
            **state,
            "standalone_question": question,
            "question_rewritten": False,
            "rewriter_reason": "No prior chat history available.",
             "last_agent_used": last_agent_used, 
            "execution_trace": existing_trace + new_trace_steps,
            "new_trace_steps": new_trace_steps,
        }

    prompt = HISTORY_REWRITE_PROMPT.format(
        chat_history=format_chat_history(chat_history),
        question=question,
    )

    raw_output = routing_evaluation_llm.invoke(prompt)
    rewrite_needed, standalone_question, reason = parse_history_rewrite_output(str(raw_output))

    if not standalone_question.strip():
        standalone_question = question
        rewrite_needed = False
        reason = "Rewrite returned empty output; original question kept."

    if rewrite_needed:
        step = "History Rewrite Applied"
        detail = f"Standalone question: {standalone_question}"
    else:
        step = "History Rewrite Not Needed"
        detail = reason

    new_trace_steps.append({
        "trace_id": f"history-rewrite-{rewrite_needed}",
        "icon": "🧠",
        "step": step,
        "detail": detail,
    })

    return {
        **state,
        "standalone_question": standalone_question,
        "question_rewritten": rewrite_needed,
        "rewriter_reason": reason,
        "execution_trace": existing_trace + new_trace_steps,
        "new_trace_steps": new_trace_steps,
    }


# Routing agent that routes to the correct agent based on user query
# If retry attempt, the prompt is added with retry_count and retry_instructions for better output
# Pass the agent_used, router_reason and trace to LangGraph AgentState
def router_node(state: AgentState) -> AgentState:
    question = state.get("standalone_question") or state["question"]
    retry_count = state.get("retry_count", 0)
    retry_instruction = state.get("retry_instruction", "")
    existing_trace = list(state.get("execution_trace", []))
    node_steps = []

    documents_available = has_ingested_documents()
    if not documents_available:
        print("ROUTER: No uploaded documents. Forcing web_search.")
        node_steps.append({
            "trace_id": f"router-web-search-no-docs-{retry_count}",
            "icon": "🧭",
            "step": "Router → Selected web_search",
            "detail": "No uploaded documents are available, so web search was forced."
        })
        return {
            **state,
            "agent_used": "web_search",
            "router_reason": "No uploaded documents are available, so RAG cannot be used.",
            "retrieved_chunks": [],
            "web_results": [],
            "sql_result": [],
            "generated_sql": "",
            "answer": "",
            "evaluation_result": "",
            "evaluator_reason": "",
            "execution_trace": existing_trace + node_steps,
            "new_trace_steps": node_steps
        }

    retry_context = ""
    if retry_count > 0 and retry_instruction and retry_instruction.lower() != "none":
        retry_context = f"""
        This is retry attempt {retry_count}.
        Evaluator improvement instruction:
        {retry_instruction}
        Use this instruction while choosing the best agent.
        """

    routing_prompt = ROUTER_PROMPT.format(
        retry_context=retry_context,
        question=question
    )

    # Use the fast routing/evaluation model here because routing is a
    # lightweight classification task and does not require the heavier
    # generation model.
    raw_decision = routing_evaluation_llm.invoke(routing_prompt)
    print(f"ROUTER RAW OUTPUT: {raw_decision}")

    raw_text = str(raw_decision)
    parsed_decision, parsed_reason = _parse_router_output(raw_text)

    if parsed_decision in ("rag", "web_search", "sql"):
        final_decision = parsed_decision
    else:
        lowered = raw_text.strip().lower()
        if "sql" in lowered or "database" in lowered or "query" in lowered:
            final_decision = "sql"
        elif "web_search" in lowered or "web" in lowered or "search" in lowered:
            final_decision = "web_search"
        elif "rag" in lowered or "document" in lowered or "pdf" in lowered:
            final_decision = "rag"
        else:
            final_decision = "web_search"

    print(f"ROUTER FINAL DECISION: {final_decision} | REASON: {parsed_reason}")

    if retry_count > 0:
        node_steps.append({
            "trace_id": f"router-retry-{retry_count}-{final_decision}",
            "icon": "🔁",
            "step": f"Router → Retry {retry_count}: selected {final_decision}",
            "detail": parsed_reason
        })
    else:
        node_steps.append({
            "trace_id": f"router-select-{retry_count}-{final_decision}",
            "icon": "🧭",
            "step": f"Router → Selected {final_decision}",
            "detail": parsed_reason
        })

    return {
        **state,
        "agent_used": final_decision,
        "router_reason": parsed_reason,
        "retrieved_chunks": [],
        "web_results": [],
        "sql_result": [],
        "generated_sql": "",
        "answer": "",
        "evaluation_result": "",
        "evaluator_reason": "",
        "execution_trace": existing_trace + node_steps,
        "new_trace_steps": node_steps
    }


# route_decision used in the conditional edge to decide which agent to go to from router_node
# Fallback to web search if llm gives a hallucinated decision
def route_decision(state: AgentState) -> str:
    decision = state.get("agent_used", "web_search")
    if decision not in ("rag", "web_search", "sql"):
        decision = "web_search"
    print(f"ROUTE_DECISION -> {decision}")
    return decision


# evaluator_decision used in the conditional_edge to decide if we should go to END state or back to router_node
# If answer is deemed good, decision is pass else retry
def evaluator_decision(state: AgentState) -> str:
    decision = str(state.get("evaluation_result", "pass")).strip().lower()
    if decision not in ("pass", "retry"):
        decision = "pass"
    print(f"EVALUATOR_DECISION -> {decision}")
    return decision


# Main method to build the Entire LangGraph graph with the AgentState
# Steps:
# 1. Add all nodes into the StateGraph (nodes are the python methods that actually do the task )
# 2. Set entry point from where the graph should begin, i.e router_node
# 3. Set a conditional_edge based on router_decision taken by the router_node, choose the appropriate sub-agent (rag, sql, web_search)
# 4. Connect each agent's retrieve node to its generate node
# 5. Each agent's retrieve extracts the relevant info (vectors/ tavily API/ SQL query) and generate sends the retrieved content back to the generation LLM
# 6. Set a conditional_edge based on the evaluator_node's response of either pass or retry
# 7. Move back to router_node if failed, and to the END if passed
# 8. Compile the entire graph, and store in a variable agentic_app
def build_agentic_workflow():
    graph = StateGraph(AgentState)


    graph.add_node("history_rewrite", history_rewrite_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("web_search", tavily_node)
    graph.add_node("tavily_generate", tavily_generate_node)
    graph.add_node("sql_generate_query", sql_generate_query_node)
    graph.add_node("sql_execute_query", sql_execute_query_node)
    graph.add_node("sql_generate_answer", sql_generate_answer_node)
    graph.add_node("evaluator", evaluator_node)


    graph.set_entry_point("history_rewrite")


    graph.add_edge("history_rewrite", "router")


    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "retrieve",
            "web_search": "web_search",
            "sql": "sql_generate_query",
        }
    )


    # Rag edges
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluator")


    # Web search edges
    graph.add_edge("web_search", "tavily_generate")
    graph.add_edge("tavily_generate", "evaluator")


    # ── HOTL: SQL branch now has two steps ──
    # sql_generate_query runs, then graph PAUSES before sql_execute_query
    # After human approval, sql_execute_query runs, then sql_generate_answer
    graph.add_edge("sql_generate_query", "sql_execute_query")
    graph.add_edge("sql_execute_query", "sql_generate_answer")
    graph.add_edge("sql_generate_answer", "evaluator")


    graph.add_conditional_edges(
        "evaluator",
        evaluator_decision,
        {
            "pass": END,
            "retry": "router"
        }
    )


    # ── HOTL: MemorySaver checkpointer + interrupt_before ──
    # checkpointer=memory → saves graph state at the pause point
    # interrupt_before=["sql_execute_query"] → pause the graph
    # BEFORE sql_execute_query runs and wait for human input


    memory = MemorySaver()


    return graph.compile(
        checkpointer=memory,
        interrupt_before=["sql_execute_query"] # ← HOTL: the actual pause trigger
    )


agentic_app = build_agentic_workflow()


def _initial_state(question: str, chat_history: list[dict] | None = None) -> dict:
    return {
        "question": question,
        "chat_history": chat_history or [],
        "standalone_question": question,
        "question_rewritten": False,
        "rewriter_reason": "",
        "retrieved_chunks": [],
        "answer": "",
        "agent_used": "",
        "web_results": [],
        "sql_result": [],
        "generated_sql": "",
        "retry_count": 0,
        "evaluator_feedback": "",
        "evaluation_result": "",
        "execution_trace": [],
        "new_trace_steps": [],
        "router_reason": "",
        "evaluator_reason": "",
        "retry_instruction": "",
        "awaiting_human_approval": False,
        "human_approved_sql": "",
        "evaluator_score": 0,
        "hitl_needed": False,
        "hitl_reason": "",
    }

# Method called from the Streamlit UI whenever the user sends a message.
# Responsibilities:
# 1) Take the user's question + thread_id + compact chat_history from app.py.
# 2) Start an MLflow run to log per-question telemetry (thread_id, question_length, final_agent, latency).
# 3) Either:
#    - Start a fresh LangGraph run from an initial AgentState, or
#    - Resume a paused SQL branch after human approval (HOTL) using the saved state.
#    by calling agentic_app.invoke(...).
# 4) Normalize the resulting AgentState so all UI-expected fields are present and non-null
#    (answer, execution_trace, retrieved_chunks, web_results, SQL fields, evaluator/HITL flags).
# 5) Return the normalized AgentState to app.py, which uses it to render:
#    - the final answer,
#    - reasoning + action trace,
#    - retrieved chunks / web results / SQL query,
#    - and any SQL approval or evaluator HITL review screens.
def ask_question(
    question: str,
    thread_id: str,
    chat_history: list[dict] | None = None,
    human_approved_sql: str = None
) -> dict:
    """
    Entry point invoked from app.py for every user message.

    Two modes:
    1. Fresh start:
       - Called from the main Streamlit chat flow in app.py
       - Builds a fresh AgentState from the user's question + chat history
       - Runs the full LangGraph workflow from the beginning
    2. Resume:
       - Called from the SQL approval / HITL UI in app.py
       - Resumes a paused LangGraph run after the user approves or edits SQL
       - Continues execution from the SQL branch instead of starting over
    """
    # LangGraph uses this config object to identify the per-thread graph state.
    # thread_id comes from app.py (st.session_state.thread_id).
    config = {"configurable": {"thread_id": thread_id}}

    # Measure total latency for this question so we can log it as a metric.
    start_time = time.time()

    # Start one MLflow run per user question to capture high-level telemetry:
    # - thread_id: which chat session this message belongs to
    # - question_length: size of the user's question (for basic analytics)
    # - final_agent: which agent (RAG / Web / SQL) ultimately answered
    # - total_latency_ms: end-to-end latency for the entire LangGraph workflow
    # These appear in the MLflow "Model training → Runs" UI under the Default experiment.
    # Traces and spans (per-LLM call) are handled separately by mlflow.langchain.autolog().
    with mlflow.start_run(run_name=f"thread-{thread_id}"):
        # Log basic request-level parameters for later analysis in the Runs UI.
        mlflow.log_param("thread_id", thread_id)
        mlflow.log_param("question_length", len(question or ""))

        if human_approved_sql is not None:
            # RESUME MODE:
            # This path is triggered when the SQL agent paused for approval and
            # app.py calls ask_question again with human_approved_sql set.
            print(f"ASK_QUESTION: Resuming paused graph for thread {thread_id}")

            # Inject the approved SQL back into the LangGraph state and mark
            # that we are no longer awaiting human approval.
            agentic_app.update_state(
                config,
                {
                    "human_approved_sql": human_approved_sql,
                    "awaiting_human_approval": False,
                },
            )

            # Resume the existing LangGraph session identified by config.
            # We pass None as input because the graph already has state.
            result = agentic_app.invoke(None, config)

        else:
            # FRESH START MODE:
            # This is the normal path when the user asks a new question from the chat UI.
            print(f"ASK_QUESTION: Starting fresh graph for thread {thread_id}")

            # Build the initial AgentState from the question + compact chat history
            # (history is prepared in app.py by build_chat_history_for_workflow()).
            initial_state = _initial_state(question, chat_history or [])

            # Start a brand-new LangGraph run for this question.
            result = agentic_app.invoke(initial_state, config)

        # Defensive guard: if the graph returned None, log latency and fall back
        # to a default initial state so the UI does not break.
        if result is None:
            total_latency_ms = int((time.time() - start_time) * 1000)
            mlflow.log_metric("total_latency_ms", total_latency_ms)
            return _initial_state(question, chat_history or [])

        # ── RESULT NORMALIZATION ───────────────────────────────────────────────
        # Ensure all fields the Streamlit UI expects are present and non-null.
        # This prevents KeyError / None issues when rendering trace panels,
        # evidence panels, SQL panels, and evaluator/HITL badges.

        if "execution_trace" in result and result["execution_trace"] is None:
            result["execution_trace"] = []
        if "retrieved_chunks" in result and result["retrieved_chunks"] is None:
            result["retrieved_chunks"] = []
        if "web_results" in result and result["web_results"] is None:
            result["web_results"] = []
        if "sql_result" in result and result["sql_result"] is None:
            result["sql_result"] = []
        if "generated_sql" not in result or result["generated_sql"] is None:
            result["generated_sql"] = ""
        if "answer" not in result or result["answer"] is None:
            result["answer"] = ""
        if "agent_used" not in result or result["agent_used"] is None:
            # Default to web_search so the UI has a label even if router failed.
            result["agent_used"] = "web_search"
        if "retry_count" not in result or result["retry_count"] is None:
            result["retry_count"] = 0
        if "awaiting_human_approval" not in result or result["awaiting_human_approval"] is None:
            result["awaiting_human_approval"] = False
        if "human_approved_sql" not in result or result["human_approved_sql"] is None:
            result["human_approved_sql"] = ""

        # Evaluator + HITL-related fields used by the UI to show badges and
        # to decide whether a Human-in-the-Loop review screen should appear.
        if "evaluator_score" not in result or result["evaluator_score"] is None:
            result["evaluator_score"] = 0
        if "evaluator_reason" not in result or result["evaluator_reason"] is None:
            result["evaluator_reason"] = ""
        if "hitl_needed" not in result or result["hitl_needed"] is None:
            result["hitl_needed"] = False
        if "hitl_reason" not in result or result["hitl_reason"] is None:
            result["hitl_reason"] = ""

        # History rewrite fields used for showing whether the question was
        # rewritten into a standalone form before routing.
        if "standalone_question" not in result or result["standalone_question"] is None:
            result["standalone_question"] = result.get("question", "")
        if "question_rewritten" not in result or result["question_rewritten"] is None:
            result["question_rewritten"] = False
        if "rewriter_reason" not in result or result["rewriter_reason"] is None:
            result["rewriter_reason"] = ""

        # ── MLflow metrics / params for this run ───────────────────────────────
        # Log which agent ultimately answered and the total end-to-end latency.
        # These are visible in the MLflow "Runs" UI and useful for aggregate
        # analysis (e.g., average latency per agent, routing behavior over time).
        final_agent = result.get("agent_used", "web_search")
        mlflow.log_param("final_agent", final_agent)

        total_latency_ms = int((time.time() - start_time) * 1000)
        mlflow.log_metric("total_latency_ms", total_latency_ms)

        # The returned dict is the full AgentState, which app.py uses to:
        # - Render the answer bubble
        # - Render execution trace steps and evidence panels
        # - Decide whether to show SQL approval or evaluator HITL UI
        return result


def stream_question(question: str, chat_history: list[dict] | None = None):
    for event in agentic_app.stream(_initial_state(question, chat_history or []), stream_mode="updates"):
        yield event


# Export the compiled graph so graph_visualizer.py and app.py can access it.
# graph_visualizer.py calls get_graph().draw_mermaid() on this object.
def get_compiled_graph():
    """
    Returns the compiled LangGraph object.
    Used by graph_visualizer.py to generate the architecture diagram.
    The graph is already built at module load time as agentic_app,
    so this just returns the existing instance.
    """
    return agentic_app