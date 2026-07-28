"""
agent_graph.py
--------------
LangGraph orchestration of two specialized agents powered by ChatGroq
(llama-3.3-70b-versatile):

    Agent 1: Code Reviewer & Bug Finder
    Agent 2: Unit Test Generator

Standalone test:
    python agent_graph.py
"""

import os
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


class AgentState(TypedDict):
    code: str
    context: str
    language: str
    review: str
    tests: str
    error: Optional[str]


def _get_llm(temperature: float = 0.2) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file in this directory "
            "(GROQ_API_KEY=your_key_here). Get a free key at https://console.groq.com/keys"
        )
    return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=api_key)


def code_reviewer_node(state: AgentState) -> AgentState:
    """Agent 1: Reviews code for bugs, code smells, and improvement opportunities."""
    try:
        llm = _get_llm(temperature=0.2)
        context_block = (
            f"\n\nRelevant codebase context (retrieved via RAG):\n{state.get('context', '')}"
            if state.get("context")
            else ""
        )

        system_prompt = (
            "You are a Principal Software Engineer acting as a rigorous Code Reviewer and Bug Finder. "
            "Analyze the given code for: (1) logical bugs, (2) unhandled edge cases, (3) security issues, "
            "(4) performance concerns, (5) style/readability improvements. "
            "Be specific, reference the exact lines or functions where possible, and prioritize findings "
            "by severity (Critical / Major / Minor). "
            "End with a one-line summary naming the single most critical issue to fix first."
        )

        human_prompt = (
            f"Language: {state.get('language', 'python')}\n\n"
            f"Code to review:\n```{state.get('language', 'python')}\n{state['code']}\n```"
            f"{context_block}"
        )

        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        state["review"] = response.content
        state["error"] = None
    except Exception as e:
        state["review"] = ""
        state["error"] = f"Code Reviewer Agent failed: {str(e)}"
    return state


def test_generator_node(state: AgentState) -> AgentState:
    """Agent 2: Generates unit tests informed by the reviewer's findings."""
    if state.get("error"):
        state["tests"] = ""
        return state

    try:
        llm = _get_llm(temperature=0.3)

        system_prompt = (
            "You are a Senior QA Engineer specializing in writing thorough, idiomatic unit tests. "
            "Given source code and a prior code review, generate a complete unit test suite using "
            "the appropriate testing framework for the language (e.g. pytest for Python, Jest for "
            "JavaScript/TypeScript, JUnit for Java, the standard 'testing' package for Go). "
            "Cover normal cases, boundary/edge cases, and any risks or bugs the reviewer flagged. "
            "Output only runnable test code, with brief inline comments explaining intent."
        )

        human_prompt = (
            f"Language: {state.get('language', 'python')}\n\n"
            f"Original code:\n```{state.get('language', 'python')}\n{state['code']}\n```\n\n"
            f"Code review findings:\n{state['review']}\n\n"
            "Generate a complete unit test file now."
        )

        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        state["tests"] = response.content
    except Exception as e:
        state["tests"] = ""
        state["error"] = f"Test Generator Agent failed: {str(e)}"
    return state


def build_graph():
    """Builds and compiles the 2-agent LangGraph state machine."""
    graph = StateGraph(AgentState)
    graph.add_node("code_reviewer", code_reviewer_node)
    graph.add_node("test_generator", test_generator_node)

    graph.set_entry_point("code_reviewer")
    graph.add_edge("code_reviewer", "test_generator")
    graph.add_edge("test_generator", END)

    return graph.compile()


def run_agent_workflow(code: str, context: str = "", language: str = "python") -> AgentState:
    """Convenience wrapper to run the full 2-agent workflow on a code snippet."""
    if not code or not code.strip():
        return {
            "code": code,
            "context": context,
            "language": language,
            "review": "",
            "tests": "",
            "error": "No code provided for review.",
        }

    app = build_graph()
    initial_state: AgentState = {
        "code": code,
        "context": context,
        "language": language,
        "review": "",
        "tests": "",
        "error": None,
    }

    try:
        final_state = app.invoke(initial_state)
        return final_state
    except Exception as e:
        initial_state["error"] = f"Agent graph execution failed: {str(e)}"
        return initial_state


if __name__ == "__main__":
    sample_code = """
def divide(a, b):
    return a / b
"""
    result = run_agent_workflow(sample_code)
    print("--- REVIEW ---")
    print(result.get("review"))
    print("--- TESTS ---")
    print(result.get("tests"))
    if result.get("error"):
        print("--- ERROR ---")
        print(result["error"])
