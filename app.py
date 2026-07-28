"""
app.py
------
Streamlit dashboard: end-to-end UI for the Agentic Codebase & Knowledge QA
System. Combines the LanceDB RAG engine and the 2-agent LangGraph workflow
(Groq-powered).

Run with:
    streamlit run app.py
"""

import os
import traceback

import streamlit as st
from dotenv import load_dotenv

from rag_engine import RAGEngine
from agent_graph import run_agent_workflow

load_dotenv()

st.set_page_config(page_title="Agentic Codebase QA System", layout="wide")


@st.cache_resource(show_spinner=False)
def get_rag_engine() -> RAGEngine:
    return RAGEngine()


def main():
    st.title("🧠 Agentic Codebase & Knowledge QA System")
    st.caption(
        "Custom MCP context tools • LanceDB RAG • LangGraph 2-Agent Workflow • "
        "Groq (llama-3.3-70b-versatile)"
    )

    if not os.getenv("GROQ_API_KEY"):
        st.warning(
            "GROQ_API_KEY not found in environment. Add it to a `.env` file "
            "(GROQ_API_KEY=your_key_here) before running the agent workflow. "
            "Get a free key at https://console.groq.com/keys"
        )

    try:
        rag_engine = get_rag_engine()
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {e}")
        st.stop()

    tab_ingest, tab_search, tab_agents = st.tabs(
        ["📥 Ingest Codebase", "🔍 Semantic Search", "🤖 Review & Test Generation"]
    )

    # ------------------------------------------------------------------ #
    # Tab 1: Ingestion
    # ------------------------------------------------------------------ #
    with tab_ingest:
        st.subheader("Ingest a local codebase directory into LanceDB")
        dir_path = st.text_input("Directory path to ingest", value=".")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Ingest Directory", type="primary"):
                with st.spinner("Chunking, embedding, and indexing files..."):
                    try:
                        result = rag_engine.ingest_directory(dir_path)
                        if result["status"] == "success":
                            st.success(result["message"])
                        else:
                            st.error(result["message"])
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

        with col2:
            if st.button("Clear Vector Index"):
                try:
                    rag_engine.clear()
                    st.info("Vector index cleared.")
                except Exception as e:
                    st.error(f"Failed to clear index: {e}")

        st.write(
            "**Index status:**",
            "Ready ✅" if rag_engine.is_ready() else "Empty ⚠️ (ingest a directory first)",
        )

    # ------------------------------------------------------------------ #
    # Tab 2: Semantic search
    # ------------------------------------------------------------------ #
    with tab_search:
        st.subheader("Semantic search over the ingested codebase")
        query = st.text_input("Search query", placeholder="e.g. how is authentication handled?")
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

        if st.button("Search", type="primary"):
            if not rag_engine.is_ready():
                st.warning("No data indexed yet. Please ingest a directory first.")
            else:
                try:
                    hits = rag_engine.search(query, k=top_k)
                    if not hits:
                        st.info("No results found.")
                    for i, hit in enumerate(hits, 1):
                        label = f"Result {i}: {hit['path']} (chunk {hit['chunk_id']})"
                        if hit["distance"] is not None:
                            label += f" — distance {hit['distance']:.4f}"
                        with st.expander(label):
                            st.code(hit["text"])
                except Exception as e:
                    st.error(f"Search failed: {e}")

    # ------------------------------------------------------------------ #
    # Tab 3: Agent workflow
    # ------------------------------------------------------------------ #
    with tab_agents:
        st.subheader("Run the 2-Agent workflow: Code Reviewer → Unit Test Generator")

        use_rag_context = st.checkbox("Augment agents with RAG context from ingested codebase", value=False)
        rag_query = ""
        if use_rag_context:
            rag_query = st.text_input("RAG context query (used to fetch related snippets)", value="")

        language = st.selectbox("Language", ["python", "javascript", "typescript", "java", "go"], index=0)

        code_input = st.text_area(
            "Paste code to review and generate tests for",
            height=300,
            placeholder="def divide(a, b):\n    return a / b",
        )

        if st.button("Run Agent Workflow", type="primary"):
            if not code_input or not code_input.strip():
                st.warning("Please paste some code first.")
            else:
                context_text = ""
                if use_rag_context and rag_query.strip():
                    if not rag_engine.is_ready():
                        st.warning("RAG index is empty — proceeding without context.")
                    else:
                        try:
                            hits = rag_engine.search(rag_query, k=5)
                            context_text = "\n\n".join(f"[{h['path']}]\n{h['text']}" for h in hits)
                        except Exception as e:
                            st.warning(f"RAG context retrieval failed, proceeding without it: {e}")

                with st.spinner("Running Code Reviewer and Test Generator agents via Groq..."):
                    result = None
                    try:
                        result = run_agent_workflow(code_input, context=context_text, language=language)
                    except Exception as e:
                        st.error(f"Agent workflow crashed: {e}")
                        st.text(traceback.format_exc())

                if result:
                    if result.get("error"):
                        st.error(result["error"])
                    if result.get("review"):
                        st.markdown("### 🔎 Code Review Findings")
                        st.markdown(result["review"])
                    if result.get("tests"):
                        st.markdown("### 🧪 Generated Unit Tests")
                        st.code(result["tests"], language=language)


if __name__ == "__main__":
    main()
