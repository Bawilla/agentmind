"""
AgentMind — Day 1: Minimal LangGraph Pipeline

Three-node graph: receive_input → process → output
Demonstrates the basic LangGraph state machine pattern.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------------
# State schema — shared dict passed between nodes
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    answer: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------
def receive_input(state: AgentState) -> AgentState:
    question = state["question"]
    print(f"[Node 1 — receive_input] Received: {question}")
    return state


def process(state: AgentState) -> AgentState:
    print("[Node 2 — process] Processing...")
    return {**state, "answer": "This is a placeholder answer"}


def output(state: AgentState) -> AgentState:
    print(f"[Node 3 — output] Final answer: {state['answer']}")
    return state


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("receive_input", receive_input)
    graph.add_node("process", process)
    graph.add_node("output", output)

    graph.set_entry_point("receive_input")
    graph.add_edge("receive_input", "process")
    graph.add_edge("process", "output")
    graph.add_edge("output", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    app = build_graph()

    initial_state: AgentState = {
        "question": "What is agentic RAG?",
        "answer": "",
    }

    print("=" * 56)
    print("  AgentMind — Day 1: Minimal LangGraph Pipeline")
    print("=" * 56)
    print()

    final_state = app.invoke(initial_state)

    print()
    print("-" * 56)
    print("  Full graph state after execution:")
    print("-" * 56)
    for key, value in final_state.items():
        print(f"  {key}: {value!r}")
    print("-" * 56)


if __name__ == "__main__":
    main()
