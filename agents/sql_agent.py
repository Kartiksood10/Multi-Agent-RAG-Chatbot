# ============================================================
# agents/sql_agent.py
# ============================================================
# SQL Agent — Human-in-the-Loop (HITL) version
#
# PURPOSE
# -------
# This agent answers questions by:
# 1. converting natural language into SQL,
# 2. pausing for human review before execution,
# 3. executing the approved SQL on PostgreSQL,
# 4. generating a natural-language answer from the SQL result.
#
# WHY THIS FILE EXISTS
# --------------------
# The SQL branch is different from RAG and Web Search because it can
# execute queries directly on the database. That makes it the right
# place to introduce Human-in-the-Loop approval before execution.
#
# NODE FLOW
# ---------
# This file now contains THREE nodes:
#
# 1. sql_generate_query_node
# - reads the user's question
# - fetches the schema from PostgreSQL
# - asks the shared generation LLM to generate a SELECT query
# - cleans the generated SQL
# - blocks unsafe / non-SELECT SQL
# - stores generated_sql in state
# - sets awaiting_human_approval = True
# - DOES NOT execute SQL yet
#
# 2. sql_execute_query_node
# - runs only after the human approves or edits the SQL in app.py
# - reads human_approved_sql from state
# - executes only that approved SQL
# - stores rows in state["sql_result"]
# - resets awaiting_human_approval = False
#
# 3. sql_generate_answer_node
# - reads SQL rows from state["sql_result"]
# - builds a natural language prompt for the generation LLM
# - returns a readable answer in state["answer"]
#
# IMPORTANT LLM NOTE
# ------------------
# The shared generation LLM is wrapped so invoke(...) returns plain text,
# not an object with .content.
# Therefore this file uses:
#
# str(generation_llm.invoke(prompt)).strip()
#
# and never:
#
# response.content
#
# SAFETY
# ------
# - LLM is instructed to generate ONLY SELECT queries
# - clean_sql() strips markdown/code fences
# - hard safety check blocks anything not starting with SELECT
# - HITL pause ensures a human approves the SQL before execution
#
# TABLE
# -----
# Default target table is document_chunks.
# Change TABLE_NAME below if you want to query another table.
# ============================================================

import re
from agents.shared import AgentState, generation_llm, get_db_connection
from langchain_core.prompts import PromptTemplate

SQL_GENERATE_PROMPT = PromptTemplate.from_file("prompts/sql_generate_prompt.txt")
SQL_LLM_PROMPT = PromptTemplate.from_file("prompts/sql_llm_prompt.txt")

TABLE_NAME = "document_chunks"

# ============================================================
# SCHEMA HELPER
# ============================================================
def get_table_schema(table_name: str) -> str:
    """
    Fetch the PostgreSQL schema for the given table and format it
    as readable text for the LLM prompt.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
        """, (table_name,))
        rows = cur.fetchall()

        if not rows:
            return f"Table '{table_name}' not found or has no columns."

        schema_lines = [f" - {col} ({dtype})" for col, dtype in rows]
        return f"Table: {table_name}\nColumns:\n" + "\n".join(schema_lines)

    finally:
        cur.close()
        conn.close()

# ============================================================
# SQL CLEANUP HELPER
# ============================================================
def clean_sql(raw_sql: str) -> str:
    """
    Clean LLM-generated SQL.

    Handles cases where the model returns:
    - ```sql ... ```
    - ``` ... ```
    - multiple statements separated by semicolons

    Returns only the first cleaned statement.
    """
    cleaned = re.sub(r"```(?:sql)?", "", raw_sql, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    return statements[0] if statements else cleaned

# ============================================================
# NODE 1 — GENERATE SQL AND PAUSE FOR HUMAN APPROVAL
# ============================================================
def sql_generate_query_node(state: AgentState) -> AgentState:
    """
    Generate a SQL query from the user's question.

    HITL behavior:
    - Generates SQL only
    - Does NOT execute it
    - Sets awaiting_human_approval = True so the workflow can pause
    """
    question = state.get("standalonequestion") or state["question"]
    retry_count = state.get("retry_count", 0)

    node_steps = [
        {
            "trace_id": f"sql-schema-start-{retry_count}",
            "icon": "🗃️",
            "step": "SQL Agent → Reading schema",
            "detail": f"Fetching schema for table: {TABLE_NAME}"
        },
        {
            "trace_id": f"sql-generate-query-{retry_count}",
            "icon": "🤖",
            "step": "SQL Agent → Generating SQL query",
            "detail": "Converting the question into a safe SQL SELECT query."
        }
    ]

    schema = get_table_schema(TABLE_NAME)

    sql_prompt = SQL_GENERATE_PROMPT.format(
        schema=schema,
        question=question
    )

    # IMPORTANT:
    # generation_llm.invoke(...) returns plain text because the shared
    # NVIDIA NIM wrapper normalizes the response into a string.
    raw_sql = generation_llm.invoke(sql_prompt)
    sql_query = clean_sql(str(raw_sql))

    if not sql_query.strip().upper().startswith("SELECT"):
        node_steps.append({
            "trace_id": f"sql-unsafe-{retry_count}",
            "icon": "🚫",
            "step": "SQL Agent → Unsafe SQL blocked",
            "detail": "Generated query was not a SELECT statement, so execution was stopped."
        })
        return {
            **state,
            "generated_sql": sql_query,
            "sql_result": [],
            "answer": "I was unable to generate a safe query for this question.",
            "awaiting_human_approval": False,
            "human_approved_sql": "",
            "execution_trace": list(state.get("execution_trace", [])) + node_steps,
            "new_trace_steps": node_steps
        }

    node_steps.append({
        "trace_id": f"sql-awaiting-approval-{retry_count}",
        "icon": "⏸️",
        "step": "SQL Agent → Waiting for human approval",
        "detail": f"Generated SQL is ready for review: {sql_query}"
    })

    return {
        **state,
        "agent_used": "sql",
        "generated_sql": sql_query,
        "sql_result": [],
        "awaiting_human_approval": True,
        "human_approved_sql": "",
        "execution_trace": list(state.get("execution_trace", [])) + node_steps,
        "new_trace_steps": node_steps
    }

# ============================================================
# NODE 2 — EXECUTE HUMAN-APPROVED SQL
# ============================================================
def sql_execute_query_node(state: AgentState) -> AgentState:
    """
    Execute the SQL that was explicitly approved or edited by the human.

    This node should run only after the workflow resumes from the HITL pause.
    """
    retry_count = state.get("retry_count", 0)
    approved_sql = state.get("human_approved_sql", "").strip()

    node_steps = [
        {
            "trace_id": f"sql-approved-received-{retry_count}",
            "icon": "✅",
            "step": "SQL Agent → Human-approved SQL received",
            "detail": f"Approved SQL ready for execution: {approved_sql}"
        }
    ]

    if not approved_sql:
        node_steps.append({
            "trace_id": f"sql-no-approved-sql-{retry_count}",
            "icon": "❌",
            "step": "SQL Agent → No approved SQL found",
            "detail": "Execution was aborted because no approved SQL was provided."
        })
        return {
            **state,
            "sql_result": [],
            "answer": "No approved SQL was provided for execution.",
            "awaiting_human_approval": False,
            "execution_trace": list(state.get("execution_trace", [])) + node_steps,
            "new_trace_steps": node_steps
        }

    if not approved_sql.upper().startswith("SELECT"):
        node_steps.append({
            "trace_id": f"sql-approved-unsafe-{retry_count}",
            "icon": "🚫",
            "step": "SQL Agent → Unsafe approved SQL blocked",
            "detail": "Only SELECT queries are allowed for execution."
        })
        return {
            **state,
            "sql_result": [],
            "answer": "Only SELECT queries are allowed.",
            "awaiting_human_approval": False,
            "execution_trace": list(state.get("execution_trace", [])) + node_steps,
            "new_trace_steps": node_steps
        }

    node_steps.append({
        "trace_id": f"sql-execute-{retry_count}",
        "icon": "⚡",
        "step": "SQL Agent → Executing approved query",
        "detail": f"Running SQL: {approved_sql}"
    })

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(approved_sql)
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]
        sql_result = [dict(zip(col_names, row)) for row in rows]

        node_steps.append({
            "trace_id": f"sql-result-{retry_count}",
            "icon": "📊",
            "step": f"SQL Agent → Query returned {len(sql_result)} rows",
            "detail": "SQL results were collected and passed to answer generation."
        })

        return {
            **state,
            "agent_used": "sql",
            "generated_sql": approved_sql,
            "sql_result": sql_result,
            "awaiting_human_approval": False,
            "execution_trace": list(state.get("execution_trace", [])) + node_steps,
            "new_trace_steps": node_steps
        }

    except Exception as e:
        node_steps.append({
            "trace_id": f"sql-error-{retry_count}",
            "icon": "❌",
            "step": "SQL Agent → Query execution failed",
            "detail": str(e)
        })
        return {
            **state,
            "generated_sql": approved_sql,
            "sql_result": [],
            "answer": f"SQL query failed to execute: {str(e)}",
            "awaiting_human_approval": False,
            "execution_trace": list(state.get("execution_trace", [])) + node_steps,
            "new_trace_steps": node_steps
        }

    finally:
        cur.close()
        conn.close()

# ============================================================
# NODE 3 — GENERATE NATURAL LANGUAGE ANSWER FROM SQL RESULT
# ============================================================
def sql_generate_answer_node(state: AgentState) -> AgentState:
    """
    Generate a human-readable answer from SQL result rows.
    """
    question = state.get("standalonequestion") or state["question"]
    sql_result = state.get("sql_result", [])
    retry_count = state.get("retry_count", 0)

    node_steps = []

    if not sql_result:
        node_steps.append({
            "trace_id": f"sql-generate-empty-{retry_count}",
            "icon": "⚠️",
            "step": "SQL Agent → No rows returned",
            "detail": "The SQL query completed but returned no data."
        })
        return {
            **state,
            "answer": "The SQL query returned no results for your question.",
            "execution_trace": list(state.get("execution_trace", [])) + node_steps,
            "new_trace_steps": node_steps
        }

    node_steps.append({
        "trace_id": f"sql-generate-start-{retry_count}",
        "icon": "✍️",
        "step": "SQL Agent → Generating answer",
        "detail": "Converting SQL result rows into a clear natural language answer."
    })

    display_rows = sql_result[:20]
    rows_text = "\n".join([str(row) for row in display_rows])
    if len(sql_result) > 20:
        rows_text += f"\n... and {len(sql_result) - 20} more rows."

    retry_feedback_block = ""
    if state.get("retry_instruction") and str(state.get("retry_instruction")).lower() != "none":
        retry_feedback_block = f"""
The evaluator asked for an improved retry.
Improve the answer using this instruction:
{state["retry_instruction"]}
"""

    prompt = SQL_LLM_PROMPT.format(
     question=question,
     rows_text=rows_text,
     retry_feedback_block=retry_feedback_block   
    )

    # IMPORTANT:
    # generation_llm.invoke(...) returns plain text because the shared
    # NVIDIA NIM wrapper normalizes the response into a string.
    answer = str(generation_llm.invoke(prompt)).strip()

    node_steps.append({
        "trace_id": f"sql-answer-finished-{retry_count}",
        "icon": "💬",
        "step": "SQL Agent → Answer generated",
        "detail": answer[:200] + ("..." if len(answer) > 200 else "")
    })

    return {
        **state,
        "agent_used": "sql",
        "answer": answer,
        "execution_trace": list(state.get("execution_trace", [])) + node_steps,
        "new_trace_steps": node_steps
    }