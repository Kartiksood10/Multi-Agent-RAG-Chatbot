# ============================================================
# graph_visualizer.py
#
# Handles LangGraph architecture diagram generation without any
# extra third-party Mermaid rendering dependency.
#
# Why this file exists:
# - Keeps diagram logic separate from app.py and workflow code
# - Generates Mermaid syntax directly from the compiled LangGraph
# - Caches the Mermaid output because graph topology is static
#   unless the code itself changes
# - Shows the architecture once in a dedicated Streamlit expander
# ============================================================

import streamlit as st


@st.cache_resource
def get_cached_mermaid_code() -> str:
    """
    Generate Mermaid syntax once from the compiled LangGraph object.

    Why cache this?
    - The graph topology does not change per user prompt
    - It only changes if your LangGraph code changes
    - So generating it once is enough and more efficient

    Returns:
        Mermaid diagram code as a string.
    """
    try:
        from agentic_workflow import get_compiled_graph
        compiled_graph = get_compiled_graph()
        return compiled_graph.get_graph().draw_mermaid()
    except Exception as e:
        print(f"GRAPH_VISUALIZER: Failed to generate Mermaid code - {e}")
        return ""


def render_architecture_diagram() -> None:
    """
    Render the LangGraph architecture once in a single Streamlit expander.

    """
    with st.expander("🗺️ LangGraph Mermaid Diagram", expanded=False):
        st.caption(
            "This diagram is auto-generated from the live LangGraph definition. "
            "It reflects the actual router, agent branches, evaluator, and retry loop."
        )

        mermaid_code = get_cached_mermaid_code()

        if mermaid_code:
            st.link_button(
                "Open in Mermaid Live",
                "https://mermaid.live",
                use_container_width=False
            )

            st.markdown("**Mermaid Code:**")
            st.code(mermaid_code, language="markdown")
        else:
            st.warning("Could not generate Mermaid diagram from the compiled graph.")