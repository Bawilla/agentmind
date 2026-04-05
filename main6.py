"""
AgentMind — Day 6: CRAG + LangSmith Tracing

Rebuilds the Day 4 Corrective RAG pipeline with full LangSmith observability:
  - Every retrieve call logged: query, chunks, sources
  - Every LLM call logged: prompt, response, token estimate, latency
  - Every grading decision logged: chunk text, grade, reasoning
  - Every web search logged: query, result count, snippets
  - Run URL printed after each question

Graph (same as Day 4):
  retrieve → grade_chunks → (conditional)
      ├─ any irrelevant  → web_search → answer
      ├─ all ambiguous   → rewrite_query → retrieve  (max 2×)
      └─ has relevant    → answer
"""

import os
import glob
import time
from typing import TypedDict, List

# tracing.py must be imported first — it calls load_dotenv()
from tracing import init_tracing, trace_step, get_run_url

import groq as _groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from ddgs import DDGS
from langsmith import traceable

# Initialise LangSmith at startup
init_tracing()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAPERS_DIR   = os.path.join(os.path.dirname(__file__), "papers")
CHROMA_DIR   = os.path.join(os.path.dirname(__file__), "chroma_db_main6")
EMBED_MODEL  = "all-MiniLM-L6-v2"
GROQ_MODEL   = "llama-3.1-8b-instant"
TOP_K        = 5
MAX_REWRITES = 2


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question:         str
    original_question: str
    rewrite_count:    int
    chunks:           List[str]
    chunk_sources:    List[str]
    grades:           List[str]
    grade_reasons:    List[str]
    relevant_context: str
    web_context:      str
    answer:           str


# ---------------------------------------------------------------------------
# Vector store
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
LLM = ChatGroq(model=GROQ_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Traced LLM call helper
# ---------------------------------------------------------------------------
@traceable(name="llm_call", run_type="llm")
def _traced_llm_call(messages: list, node_name: str) -> str:
    """Wraps every LLM invoke so LangSmith captures prompt + response.
    Includes 5s pre-call sleep and exponential backoff on RateLimitError."""
    for attempt in range(3):
        try:
            t0 = time.perf_counter()
            time.sleep(5)
            response = LLM.invoke(messages)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            content = response.content.strip()
            token_est = int(sum(len(m.content.split()) for m in messages) * 1.3)
            print(f"  [LLM] {node_name} — {latency_ms}ms, ~{token_est} tokens in")
            return content
        except _groq.RateLimitError:
            if attempt == 2:
                raise
            wait = 10 * (2 ** attempt)
            print(f"  [RateLimit] Groq rate limit hit — waiting {wait}s …")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Node 1 — retrieve
# ---------------------------------------------------------------------------
@traceable(name="retrieve", run_type="retriever")
def retrieve(state: AgentState) -> AgentState:
    question      = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    label = f"(rewrite #{rewrite_count})" if rewrite_count > 0 else ""
    print(f"\n[Node: retrieve] {label}")
    print(f"  Query : {question!r}")

    t0 = time.perf_counter()
    results = VECTORSTORE.similarity_search(question, k=TOP_K)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    chunks  = [doc.page_content.strip() for doc in results]
    sources = [os.path.basename(doc.metadata.get("source", "unknown")) for doc in results]

    print(f"  Retrieved {len(chunks)} chunks in {latency_ms}ms:")
    for i, (src, chunk) in enumerate(zip(sources, chunks), 1):
        preview = chunk[:100].replace("\n", " ")
        print(f"    Chunk {i} [{src}]: {preview} …")

    return {
        **state,
        "chunks":         chunks,
        "chunk_sources":  sources,
        "grades":         [],
        "grade_reasons":  [],
        "relevant_context": "",
        "web_context":    "",
    }


# ---------------------------------------------------------------------------
# Node 2 — grade_chunks
# ---------------------------------------------------------------------------
GRADE_PROMPT = """You are a relevance grader. Given a question and a retrieved text chunk,
classify the chunk as:
  relevant   — directly answers or strongly supports the question
  irrelevant — off-topic, unrelated to the question
  ambiguous  — partially related but not directly answering

Reply in exactly this format (two lines):
GRADE: <relevant|irrelevant|ambiguous>
REASON: <one sentence>"""


@traceable(name="grade_chunks", run_type="chain")
def grade_chunks(state: AgentState) -> AgentState:
    question = state["question"]
    chunks   = state["chunks"]
    sources  = state["chunk_sources"]

    print(f"\n[Node: grade_chunks]")
    print(f"  Grading {len(chunks)} chunks for: {question!r}")

    grades:  List[str] = []
    reasons: List[str] = []

    for i, (chunk, src) in enumerate(zip(chunks, sources), 1):
        messages = [
            SystemMessage(content=GRADE_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nChunk:\n{chunk[:600]}"),
        ]
        text = _traced_llm_call(messages, f"grade_chunk_{i}")

        grade  = "ambiguous"
        reason = ""
        for line in text.splitlines():
            if line.upper().startswith("GRADE:"):
                raw = line.split(":", 1)[1].strip().lower()
                if "irrelevant" in raw:
                    grade = "irrelevant"
                elif "relevant" in raw and "ir" not in raw:
                    grade = "relevant"
                else:
                    grade = "ambiguous"
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        grades.append(grade)
        reasons.append(reason)
        print(f"    Chunk {i} [{src}] → {grade.upper()}  | {reason}")

    return {**state, "grades": grades, "grade_reasons": reasons}


# ---------------------------------------------------------------------------
# Routing after grade_chunks
# ---------------------------------------------------------------------------
def route_after_grading(state: AgentState) -> str:
    grades        = state["grades"]
    rewrite_count = state.get("rewrite_count", 0)

    has_relevant   = any(g == "relevant"   for g in grades)
    has_irrelevant = any(g == "irrelevant" for g in grades)
    all_ambiguous  = all(g == "ambiguous"  for g in grades)

    print(f"\n[Routing] grades={grades}")

    if all_ambiguous and rewrite_count < MAX_REWRITES:
        print(f"  → rewrite_query  (all ambiguous, attempt #{rewrite_count + 1})")
        return "rewrite_query"

    if has_irrelevant or (all_ambiguous and rewrite_count >= MAX_REWRITES):
        print("  → web_search  (irrelevant/ambiguous chunks present)")
        return "web_search"

    print("  → answer  (relevant chunks found)")
    return "answer"


# ---------------------------------------------------------------------------
# Node 3 — web_search
# ---------------------------------------------------------------------------
@traceable(name="web_search", run_type="tool")
def web_search(state: AgentState) -> AgentState:
    question = state["original_question"]
    print(f"\n[Node: web_search]")
    print(f"  Querying DuckDuckGo: {question!r}")

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
            snippet = r.get("body",  "No snippet")
            parts.append(f"[Web {i}] {title}\n{snippet}")
            print(f"    Result {i}: {title}")
        web_context = "\n\n".join(parts)
    else:
        web_context = ""
        print("  No results returned.")

    print(f"  Got {len(results)} web results.")
    return {**state, "web_context": web_context}


# ---------------------------------------------------------------------------
# Node 4 — rewrite_query
# ---------------------------------------------------------------------------
REWRITE_PROMPT = """You are a query rewriter. The original question did not yield relevant
retrieval results. Rewrite it to be more specific and use different terminology likely
to match academic AI paper content.

Reply with only the rewritten question — no explanations, no quotes."""


@traceable(name="rewrite_query", run_type="chain")
def rewrite_query(state: AgentState) -> AgentState:
    original      = state["original_question"]
    current       = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    print(f"\n[Node: rewrite_query]  (attempt {rewrite_count + 1}/{MAX_REWRITES})")
    print(f"  Current query: {current!r}")

    messages = [
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=f"Original: {original}\nCurrent: {current}"),
    ]
    new_question = _traced_llm_call(messages, "rewrite_query").strip('"').strip("'")
    print(f"  Rewritten to : {new_question!r}")

    return {
        **state,
        "question":     new_question,
        "rewrite_count": rewrite_count + 1,
    }


# ---------------------------------------------------------------------------
# Node 5 — answer
# ---------------------------------------------------------------------------
@traceable(name="answer", run_type="chain")
def answer(state: AgentState) -> AgentState:
    question = state["original_question"]
    grades   = state["grades"]
    chunks   = state["chunks"]
    sources  = state["chunk_sources"]
    web_ctx  = state.get("web_context", "")

    # Keep non-irrelevant paper chunks
    kept = [
        f"[Paper chunk {i} | {src}]\n{chunk.strip()}"
        for i, (chunk, src, grade) in enumerate(zip(chunks, sources, grades), 1)
        if grade != "irrelevant"
    ]
    rag_context = "\n\n".join(kept)

    sections = []
    if rag_context:
        sections.append("=== Retrieved Paper Context ===\n" + rag_context)
    if web_ctx:
        sections.append("=== Web Search Results ===\n" + web_ctx)
    combined = "\n\n".join(sections)

    print(f"\n[Node: answer]")
    print(f"  Paper chunks kept : {len(kept)}/{len(grades)}")
    print(f"  Web context used  : {'yes' if web_ctx else 'no'}")

    if combined:
        system_prompt = (
            "You are a helpful research assistant. "
            "Answer the question using the provided context. "
            "Integrate web results with paper excerpts where both are present. "
            "Be concise and accurate."
        )
        user_content = f"Context:\n{combined}\n\nQuestion: {question}"
    else:
        system_prompt = "You are a helpful assistant. Answer concisely from general knowledge."
        user_content  = question

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    ans = _traced_llm_call(messages, "answer_generation")

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
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("answer",        END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main — batch mode + interactive loop
# ---------------------------------------------------------------------------
def run_question(app, question: str) -> None:
    print(f"\n{'━' * 65}")
    print(f"  Question: {question}")
    print(f"{'━' * 65}")

    initial: AgentState = {
        "question":          question,
        "original_question": question,
        "rewrite_count":     0,
        "chunks":            [],
        "chunk_sources":     [],
        "grades":            [],
        "grade_reasons":     [],
        "relevant_context":  "",
        "web_context":       "",
        "answer":            "",
    }

    final = app.invoke(initial)

    grades_summary = ", ".join(
        f"C{j+1}:{g[0].upper()}" for j, g in enumerate(final["grades"])
    )

    print(f"\n  ── Trace Summary ────────────────────────────────────────")
    print(f"  Chunk grades   : {grades_summary}")
    print(f"  Query rewrites : {final['rewrite_count']}")
    print(f"  Web search     : {'yes' if final['web_context'] else 'no'}")
    print(f"\n  Answer:\n  {final['answer']}")
    print()
    get_run_url(question)


def main():
    app = build_graph()

    # --- Batch test questions ---
    test_questions = [
        "How does corrective RAG handle ambiguous chunks?",
        "What are the latest developments in LangGraph?",
        "Explain the ReAct reasoning framework",
    ]

    print("\n" + "=" * 65)
    print("  AgentMind — Day 6: CRAG + LangSmith Tracing")
    print("=" * 65)

    for q in test_questions:
        run_question(app, q)

    # --- Interactive loop ---
    print("\n" + "=" * 65)
    print("  Interactive mode — type a question or 'quit' to exit")
    print("=" * 65)

    while True:
        try:
            user_input = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input or user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        run_question(app, user_input)


if __name__ == "__main__":
    main()
