"""
AgentMind — Day 4: Corrective RAG (CRAG)

Full CRAG loop implemented as a LangGraph graph:

  retrieve → grade_chunks → (conditional)
      ├─ all relevant       → answer
      ├─ any irrelevant     → web_search → answer
      └─ all ambiguous      → rewrite_query → retrieve  (max 2 rewrites)

Nodes:
  retrieve       — ChromaDB top-5 similarity search (chroma_db_main4/)
  grade_chunks   — LLM grades each chunk: relevant / irrelevant / ambiguous
  web_search     — DuckDuckGo supplement (triggered on any irrelevant/ambiguous)
  rewrite_query  — LLM rewrites question to be more specific (all-ambiguous path)
  answer         — Groq llama3-8b final answer from all kept context
"""

import os
import glob
import time
from typing import TypedDict, List

import groq as _groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from ddgs import DDGS

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAPERS_DIR  = os.path.join(os.path.dirname(__file__), "papers")
CHROMA_DIR  = os.path.join(os.path.dirname(__file__), "chroma_db_main4")
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.1-8b-instant"
TOP_K       = 5
MAX_REWRITES = 2


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question:        str            # current (possibly rewritten) question
    original_question: str          # never modified
    rewrite_count:   int            # how many times query has been rewritten
    chunks:          List[str]      # raw page_content of retrieved chunks
    chunk_sources:   List[str]      # filenames matching each chunk
    grades:          List[str]      # "relevant" | "irrelevant" | "ambiguous"
    relevant_context: str           # kept chunks (formatted)
    web_context:     str            # DuckDuckGo results (may be empty)
    answer:          str


# ---------------------------------------------------------------------------
# Build / load vector store
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
# Node 1 — retrieve
# ---------------------------------------------------------------------------
def retrieve(state: AgentState) -> AgentState:
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    label = f"(rewrite #{rewrite_count})" if rewrite_count > 0 else ""
    print(f"\n[Node: retrieve] {label}")
    print(f"  Query : {question!r}")
    print(f"  Fetching top {TOP_K} chunks from ChromaDB …")

    results = VECTORSTORE.similarity_search(question, k=TOP_K)

    chunks = [doc.page_content.strip() for doc in results]
    sources = [os.path.basename(doc.metadata.get("source", "unknown")) for doc in results]

    for i, (src, chunk) in enumerate(zip(sources, chunks), 1):
        preview = chunk[:120].replace("\n", " ")
        print(f"  Chunk {i} [{src}]: {preview} …")

    return {
        **state,
        "chunks": chunks,
        "chunk_sources": sources,
        "grades": [],
        "relevant_context": "",
        "web_context": "",
    }


# ---------------------------------------------------------------------------
# Node 2 — grade_chunks
# ---------------------------------------------------------------------------
GRADE_PROMPT = """You are a relevance grader. Given a question and a retrieved text chunk,
decide if the chunk is:
  relevant   — directly answers or strongly supports answering the question
  irrelevant — off-topic, unrelated to the question
  ambiguous  — partially related but not directly answering

Reply with exactly one word: relevant, irrelevant, or ambiguous."""


def grade_chunks(state: AgentState) -> AgentState:
    question = state["question"]
    chunks = state["chunks"]
    sources = state["chunk_sources"]

    print(f"\n[Node: grade_chunks]")
    print(f"  Grading {len(chunks)} chunks for: {question!r}")

    grades: List[str] = []
    for i, (chunk, src) in enumerate(zip(chunks, sources), 1):
        messages = [
            SystemMessage(content=GRADE_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nChunk:\n{chunk[:600]}"),
        ]
        response = _llm_invoke(messages)
        raw = response.content.strip().lower()

        # normalise to one of three valid labels
        if "irrelevant" in raw:
            grade = "irrelevant"
        elif "ambiguous" in raw:
            grade = "ambiguous"
        else:
            grade = "relevant"

        grades.append(grade)
        print(f"  Chunk {i} [{src}] → {grade.upper()}")

    return {**state, "grades": grades}


# ---------------------------------------------------------------------------
# Routing after grade_chunks
# ---------------------------------------------------------------------------
def route_after_grading(state: AgentState) -> str:
    grades = state["grades"]
    rewrite_count = state.get("rewrite_count", 0)

    has_relevant   = any(g == "relevant"   for g in grades)
    has_irrelevant = any(g == "irrelevant" for g in grades)
    all_ambiguous  = all(g == "ambiguous"  for g in grades)

    print(f"\n[Routing] grades={grades}")

    # All ambiguous + rewrites remaining → rewrite query
    if all_ambiguous and rewrite_count < MAX_REWRITES:
        print(f"  → rewrite_query  (all ambiguous, rewrite #{rewrite_count + 1})")
        return "rewrite_query"

    # Any irrelevant (or all-ambiguous but out of rewrites) → web_search
    if has_irrelevant or (all_ambiguous and rewrite_count >= MAX_REWRITES):
        print("  → web_search  (irrelevant/ambiguous chunks detected)")
        return "web_search"

    # All relevant → straight to answer
    print("  → answer  (all chunks relevant)")
    return "answer"


# ---------------------------------------------------------------------------
# Node 3 — web_search
# ---------------------------------------------------------------------------
def web_search(state: AgentState) -> AgentState:
    question = state["original_question"]
    print(f"\n[Node: web_search]")
    print(f"  Searching DuckDuckGo for: {question!r}")

    results = []
    try:
        ddgs = DDGS()
        for r in ddgs.text(question, max_results=3):
            results.append(r)
    except Exception as exc:
        print(f"  DuckDuckGo error: {exc}")

    if results:
        parts = []
        for i, r in enumerate(results, 1):
            title   = r.get("title", "No title")
            snippet = r.get("body", "No snippet")
            parts.append(f"[Web {i}] {title}\n{snippet}")
        web_context = "\n\n".join(parts)
        print(f"  Retrieved {len(results)} web results.")
    else:
        web_context = ""
        print("  No web results returned.")

    return {**state, "web_context": web_context}


# ---------------------------------------------------------------------------
# Node 4 — rewrite_query
# ---------------------------------------------------------------------------
REWRITE_PROMPT = """You are a query rewriter. The original question did not yield relevant
retrieval results. Rewrite the question to be more specific, use different terminology,
and focus on the core information need.

Reply with only the rewritten question — no explanations, no quotes."""


def rewrite_query(state: AgentState) -> AgentState:
    original  = state["original_question"]
    current   = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    print(f"\n[Node: rewrite_query]  (attempt {rewrite_count + 1}/{MAX_REWRITES})")
    print(f"  Current query: {current!r}")

    messages = [
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=f"Original question: {original}\nCurrent query: {current}"),
    ]
    response = _llm_invoke(messages)
    new_question = response.content.strip().strip('"').strip("'")

    print(f"  Rewritten  to: {new_question!r}")

    return {
        **state,
        "question": new_question,
        "rewrite_count": rewrite_count + 1,
    }


# ---------------------------------------------------------------------------
# Node 5 — answer
# ---------------------------------------------------------------------------
def answer(state: AgentState) -> AgentState:
    question = state["original_question"]
    grades   = state["grades"]
    chunks   = state["chunks"]
    sources  = state["chunk_sources"]
    web_ctx  = state.get("web_context", "")

    # Collect all non-irrelevant chunks
    kept_parts = []
    for i, (chunk, src, grade) in enumerate(zip(chunks, sources, grades), 1):
        if grade != "irrelevant":
            kept_parts.append(f"[Paper chunk {i} | {src}]\n{chunk.strip()}")

    rag_context = "\n\n".join(kept_parts)

    # Build combined context
    context_sections = []
    if rag_context:
        context_sections.append("=== Retrieved Paper Context ===\n" + rag_context)
    if web_ctx:
        context_sections.append("=== Web Search Results ===\n" + web_ctx)

    combined_context = "\n\n".join(context_sections)

    print(f"\n[Node: answer]")
    print(f"  Paper chunks kept : {sum(1 for g in grades if g != 'irrelevant')}/{len(grades)}")
    print(f"  Web context       : {'yes' if web_ctx else 'no'}")

    if combined_context:
        system_prompt = (
            "You are a helpful research assistant. "
            "Answer the question using the provided context. "
            "If web results are present, integrate them with the paper excerpts. "
            "Be concise and accurate."
        )
        user_content = f"Context:\n{combined_context}\n\nQuestion: {question}"
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

    return {**state, "answer": ans}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve",      retrieve)
    graph.add_node("grade_chunks",  grade_chunks)
    graph.add_node("web_search",    web_search)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("answer",        answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_chunks")

    graph.add_conditional_edges(
        "grade_chunks",
        route_after_grading,
        {
            "answer":        "answer",
            "web_search":    "web_search",
            "rewrite_query": "rewrite_query",
        },
    )

    graph.add_edge("web_search",    "answer")
    graph.add_edge("rewrite_query", "retrieve")   # loop back for re-retrieval
    graph.add_edge("answer",        END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    app = build_graph()

    questions = [
        "How does surface code correct quantum errors?",
        "What did Elon Musk say about AI in 2024?",
        "How does CRAG improve retrieval?",
        "What are the latest LangGraph features?",
    ]

    print("\n" + "=" * 65)
    print("  AgentMind — Day 4: Corrective RAG (CRAG)")
    print("=" * 65)

    for i, q in enumerate(questions, 1):
        print(f"\n{'━' * 65}")
        print(f"  Question {i}: {q}")
        print(f"{'━' * 65}")

        initial_state: AgentState = {
            "question":          q,
            "original_question": q,
            "rewrite_count":     0,
            "chunks":            [],
            "chunk_sources":     [],
            "grades":            [],
            "relevant_context":  "",
            "web_context":       "",
            "answer":            "",
        }

        final = app.invoke(initial_state)

        grades_summary = ", ".join(
            f"C{j+1}:{g[0].upper()}" for j, g in enumerate(final["grades"])
        )
        print(f"\n  ── Final Trace ──────────────────────────────────────────")
        print(f"  Grades          : {grades_summary}")
        print(f"  Query rewrites  : {final['rewrite_count']}")
        print(f"  Web search used : {'yes' if final['web_context'] else 'no'}")
        print(f"  Final question  : {final['question']!r}")
        print(f"\n  Answer:\n  {final['answer']}")

    print(f"\n{'=' * 65}")
    print("  All questions processed.")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
