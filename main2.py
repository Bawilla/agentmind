"""
AgentMind — Day 2: Basic Agentic RAG with Conditional Retrieval Routing

LangGraph agent with three nodes:
  decide   → LLM decides if retrieval is needed
  retrieve → ChromaDB lookup using HuggingFace embeddings
  answer   → Groq llama3-8b generates the final answer

Conditional edge: decide → retrieve → answer  (if needs_retrieval)
                  decide → answer             (if not needs_retrieval)
"""

import os
import glob
import time
from typing import TypedDict

import groq as _groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAPERS_DIR = os.path.join(os.path.dirname(__file__), "papers")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db_main2")
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 5


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    needs_retrieval: bool


# ---------------------------------------------------------------------------
# Build / load the vector store (runs once on import)
# ---------------------------------------------------------------------------
def _build_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # If the DB already exists, just load it
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
# LLM client
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
# Node 1 — decide
# ---------------------------------------------------------------------------
def decide(state: AgentState) -> AgentState:
    question = state["question"]

    system_prompt = (
        "You are a routing assistant. "
        "Decide whether the following question requires looking up specific details "
        "from research papers (RAG systems, agentic AI, Self-RAG, Corrective RAG, "
        "ReAct, LLM agents). "
        "Reply with exactly one word: YES if retrieval is needed, NO if the question "
        "can be answered from general knowledge alone."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]
    response = _llm_invoke(messages)
    verdict = response.content.strip().upper()
    needs = verdict.startswith("YES")

    path = "retrieve → answer" if needs else "answer (direct)"
    print(f"[Node 1 — decide]  Q: {question!r}")
    print(f"                   LLM verdict: {verdict!r}  →  path: {path}")

    return {**state, "needs_retrieval": needs}


# ---------------------------------------------------------------------------
# Node 2 — retrieve
# ---------------------------------------------------------------------------
def retrieve(state: AgentState) -> AgentState:
    question = state["question"]
    print(f"[Node 2 — retrieve] Querying ChromaDB for top {TOP_K} chunks …")

    results = VECTORSTORE.similarity_search(question, k=TOP_K)
    context_parts = []
    for i, doc in enumerate(results, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        context_parts.append(f"[{i}] ({source})\n{doc.page_content.strip()}")

    context = "\n\n".join(context_parts)
    print(f"[Node 2 — retrieve] Retrieved {len(results)} chunks.")
    return {**state, "context": context}


# ---------------------------------------------------------------------------
# Node 3 — answer
# ---------------------------------------------------------------------------
def answer(state: AgentState) -> AgentState:
    question = state["question"]
    context = state.get("context", "")

    if context:
        system_prompt = (
            "You are a helpful research assistant. "
            "Answer the question using ONLY the provided context excerpts. "
            "Be concise and accurate."
        )
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        system_prompt = (
            "You are a helpful assistant. "
            "Answer the question from your general knowledge. Be concise."
        )
        user_content = question

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = _llm_invoke(messages)
    ans = response.content.strip()
    mode = "with retrieved context" if context else "from general knowledge"
    print(f"[Node 3 — answer]   Answered {mode}.")
    return {**state, "answer": ans}


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------
def route_after_decide(state: AgentState) -> str:
    return "retrieve" if state["needs_retrieval"] else "answer"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("decide", decide)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)

    graph.set_entry_point("decide")
    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {"retrieve": "retrieve", "answer": "answer"},
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    app = build_graph()

    questions = [
        "What is 2 + 2?",
        "What is the main contribution of the RAG paper?",
        "Who is the president of France?",
        "How does Self-RAG reflect on its own answers?",
        "What year was Python created?",
    ]

    print("\n" + "=" * 60)
    print("  AgentMind — Day 2: Basic Agentic RAG")
    print("=" * 60)

    for i, q in enumerate(questions, 1):
        print(f"\n{'─' * 60}")
        print(f"  Question {i}: {q}")
        print(f"{'─' * 60}")

        initial_state: AgentState = {
            "question": q,
            "context": "",
            "answer": "",
            "needs_retrieval": False,
        }

        final = app.invoke(initial_state)
        path_taken = "retrieve → answer" if final["needs_retrieval"] else "answer (direct)"
        print(f"\n  Path taken  : {path_taken}")
        print(f"  Answer      : {final['answer']}")

    print(f"\n{'=' * 60}")
    print("  All questions processed.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
