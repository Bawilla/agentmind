"""
AgentMind — Day 3: Multi-Tool Agent

LangGraph agent with 3 tools it can choose from:
  Tool 1 — retrieve_papers  : ChromaDB semantic search over papers/ folder
  Tool 2 — web_search       : DuckDuckGo live web search
  Tool 3 — calculate        : Safe Python math expression evaluator

Graph structure:
  agent → (conditional) → retrieve_papers | web_search | calculate → answer

The LLM in "agent" decides which tool to call based on the question.
"""

import os
import glob
import math
import statistics
import re
import time
from typing import TypedDict, Literal

import groq as _groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from ddgs import DDGS

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAPERS_DIR = os.path.join(os.path.dirname(__file__), "papers")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db_main3")
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 5


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    tool_selected: str        # "retrieve_papers" | "web_search" | "calculate"
    tool_reason: str          # LLM's explanation for choosing the tool
    tool_result: str          # raw output from the chosen tool
    answer: str


# ---------------------------------------------------------------------------
# Build / load vector store once
# ---------------------------------------------------------------------------
def _build_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("[VectorStore] Loading existing ChromaDB from", CHROMA_DIR)
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    print("[VectorStore] Building ChromaDB — loading PDFs …")
    pdf_paths = glob.glob(os.path.join(PAPERS_DIR, "*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {PAPERS_DIR}")

    docs = []
    for path in pdf_paths:
        print(f"  Loading {os.path.basename(path)} …")
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"[VectorStore] {len(chunks)} chunks from {len(pdf_paths)} PDFs")

    store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print("[VectorStore] ChromaDB saved to", CHROMA_DIR)
    return store


VECTORSTORE: Chroma = _build_vectorstore()

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM = ChatGroq(model=GROQ_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Rate-limit-safe LLM helper
# ---------------------------------------------------------------------------
def _llm_invoke(messages: list):
    """Invoke LLM with exponential backoff on RateLimitError (max 3 retries)."""
    for attempt in range(3):
        try:
            time.sleep(5)
            return LLM.invoke(messages)
        except _groq.RateLimitError:
            if attempt == 2:
                raise
            wait = 10 * (2 ** attempt)
            print(f"  [RateLimit] Groq rate limit hit — waiting {wait}s …")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Node 1 — agent: decides which tool to use
# ---------------------------------------------------------------------------
TOOL_SELECTION_PROMPT = """You are a routing assistant. Given a question, decide which tool to use:

Tools available:
  1. retrieve_papers — Use when the question is about research topics: RAG, Self-RAG, Corrective RAG,
     ReAct, agentic AI, LLM agents, or any topic likely covered in academic AI papers.
  2. web_search — Use when the question asks about current events, recent versions/releases,
     real-world facts, news, sports, prizes, or anything requiring up-to-date web information.
  3. calculate — Use when the question involves arithmetic, percentages, powers, roots,
     or mathematical computations.

Reply in this exact format (two lines only):
TOOL: <tool_name>
REASON: <one sentence explanation>

Where <tool_name> is exactly one of: retrieve_papers, web_search, calculate"""


def agent(state: AgentState) -> AgentState:
    question = state["question"]

    messages = [
        SystemMessage(content=TOOL_SELECTION_PROMPT),
        HumanMessage(content=question),
    ]
    response = _llm_invoke(messages)
    text = response.content.strip()

    # Parse TOOL and REASON lines
    tool_selected = "web_search"  # safe default
    tool_reason = "No reason provided."
    for line in text.splitlines():
        if line.startswith("TOOL:"):
            raw = line.split(":", 1)[1].strip().lower()
            # normalise to one of the three valid names
            if "retrieve" in raw or "paper" in raw:
                tool_selected = "retrieve_papers"
            elif "calculat" in raw or "math" in raw:
                tool_selected = "calculate"
            else:
                tool_selected = "web_search"
        elif line.startswith("REASON:"):
            tool_reason = line.split(":", 1)[1].strip()

    print(f"\n[Node: agent]")
    print(f"  Question : {question!r}")
    print(f"  Selected : {tool_selected}")
    print(f"  Reason   : {tool_reason}")

    return {**state, "tool_selected": tool_selected, "tool_reason": tool_reason}


# ---------------------------------------------------------------------------
# Node 2 — retrieve_papers
# ---------------------------------------------------------------------------
def retrieve_papers(state: AgentState) -> AgentState:
    question = state["question"]
    print(f"[Node: retrieve_papers] Querying ChromaDB (top {TOP_K}) …")

    results = VECTORSTORE.similarity_search(question, k=TOP_K)
    parts = []
    for i, doc in enumerate(results, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        parts.append(f"[{i}] ({source})\n{doc.page_content.strip()}")

    tool_result = "\n\n".join(parts)
    print(f"[Node: retrieve_papers] Retrieved {len(results)} chunks.")
    return {**state, "tool_result": tool_result}


# ---------------------------------------------------------------------------
# Node 3 — web_search
# ---------------------------------------------------------------------------
def web_search(state: AgentState) -> AgentState:
    question = state["question"]
    print(f"[Node: web_search] Searching DuckDuckGo for: {question!r}")

    results = []
    ddgs = DDGS()
    for r in ddgs.text(question, max_results=3):
        results.append(r)

    if results:
        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            snippet = r.get("body", "No snippet")
            parts.append(f"[{i}] {title}\n{snippet}")
        tool_result = "\n\n".join(parts)
    else:
        tool_result = "No web results found."

    print(f"[Node: web_search] Got {len(results)} results.")
    return {**state, "tool_result": tool_result}


# ---------------------------------------------------------------------------
# Node 4 — calculate
# ---------------------------------------------------------------------------
# Allowed names for safe eval
_SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "divmod": divmod,
    "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "pi": math.pi, "e": math.e,
    "mean": statistics.mean, "median": statistics.median,
    "stdev": statistics.stdev,
}


def _extract_expression(question: str) -> str:
    """Best-effort extraction of a math expression from natural language."""
    # Handle 'X% of Y' → (X/100)*Y
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", question, re.IGNORECASE)
    if pct_match:
        pct, base = pct_match.group(1), pct_match.group(2)
        return f"({pct}/100)*{base}"

    # 'square root of X' → sqrt(X)
    sqrt_match = re.search(r"square\s+root\s+of\s+(\d+(?:\.\d+)?)", question, re.IGNORECASE)
    if sqrt_match:
        return f"sqrt({sqrt_match.group(1)})"

    # 'cube root of X' → X**(1/3)
    cbrt_match = re.search(r"cube\s+root\s+of\s+(\d+(?:\.\d+)?)", question, re.IGNORECASE)
    if cbrt_match:
        return f"({cbrt_match.group(1)})**(1/3)"

    # Keep only math-safe characters and try as-is
    expr = re.sub(r"[^0-9+\-*/().,^ %sqrtlogpiabsce\s]", "", question)
    expr = expr.replace("^", "**").strip()
    return expr if expr else question


def calculate(state: AgentState) -> AgentState:
    question = state["question"]
    expr = _extract_expression(question)
    print(f"[Node: calculate] Expression: {expr!r}")

    try:
        result = eval(expr, _SAFE_GLOBALS)  # noqa: S307 — restricted globals
        tool_result = f"Expression: {expr}\nResult: {result}"
    except Exception as exc:
        tool_result = f"Could not evaluate '{expr}': {exc}"

    print(f"[Node: calculate] {tool_result}")
    return {**state, "tool_result": tool_result}


# ---------------------------------------------------------------------------
# Node 5 — answer: generates final natural-language answer
# ---------------------------------------------------------------------------
def answer(state: AgentState) -> AgentState:
    question = state["question"]
    tool_used = state.get("tool_selected", "unknown")
    tool_result = state.get("tool_result", "")

    if tool_used == "retrieve_papers":
        system_prompt = (
            "You are a helpful research assistant. "
            "Answer the question using ONLY the provided context excerpts from research papers. "
            "Be concise and accurate. Cite source filenames when relevant."
        )
        user_content = f"Context from papers:\n{tool_result}\n\nQuestion: {question}"
    elif tool_used == "web_search":
        system_prompt = (
            "You are a helpful assistant. "
            "Answer the question using the web search results provided. "
            "Be concise and factual."
        )
        user_content = f"Web search results:\n{tool_result}\n\nQuestion: {question}"
    else:  # calculate
        system_prompt = (
            "You are a helpful math assistant. "
            "Present the calculation result clearly and in plain English."
        )
        user_content = f"Calculation:\n{tool_result}\n\nQuestion: {question}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = _llm_invoke(messages)
    ans = response.content.strip()
    print(f"[Node: answer] Answer generated (tool used: {tool_used}).")
    return {**state, "answer": ans}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_to_tool(state: AgentState) -> Literal["retrieve_papers", "web_search", "calculate"]:
    return state["tool_selected"]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent)
    graph.add_node("retrieve_papers", retrieve_papers)
    graph.add_node("web_search", web_search)
    graph.add_node("calculate", calculate)
    graph.add_node("answer", answer)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        route_to_tool,
        {
            "retrieve_papers": "retrieve_papers",
            "web_search": "web_search",
            "calculate": "calculate",
        },
    )

    graph.add_edge("retrieve_papers", "answer")
    graph.add_edge("web_search", "answer")
    graph.add_edge("calculate", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    app = build_graph()

    questions = [
        "What is 15% of 2847?",
        "How does corrective RAG handle irrelevant chunks?",
        "What is the latest version of LangGraph?",
        "What is the square root of 1764?",
        "How does ReAct combine reasoning and acting?",
        "Who won the 2024 Nobel Prize in Physics?",
    ]

    print("\n" + "=" * 65)
    print("  AgentMind — Day 3: Multi-Tool Agent")
    print("=" * 65)

    for i, q in enumerate(questions, 1):
        print(f"\n{'─' * 65}")
        print(f"  Question {i}: {q}")
        print(f"{'─' * 65}")

        initial_state: AgentState = {
            "question": q,
            "tool_selected": "",
            "tool_reason": "",
            "tool_result": "",
            "answer": "",
        }

        final = app.invoke(initial_state)

        print(f"\n  Tool used : {final['tool_selected']}")
        print(f"  Reason    : {final['tool_reason']}")
        print(f"  Answer    : {final['answer']}")

    print(f"\n{'=' * 65}")
    print("  All questions processed.")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
