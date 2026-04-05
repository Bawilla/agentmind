"""
AgentMind — Day 5: Self-RAG

Full Self-RAG loop implemented as a LangGraph graph:

  retrieve → generate → reflect_retrieval → (conditional)
      └─ unsupported → retrieve again (rewritten query, max 2×)
      └─ supported / partial → reflect_answer → (conditional)
          └─ any score < 0.7 → regenerate → reflect_answer  (max 2×)
          └─ all scores >= 0.7 → answer

Nodes:
  retrieve            — ChromaDB top-5 similarity search (chroma_db_main5/)
  generate            — Groq llama3-8b initial answer from retrieved context
  reflect_retrieval   — LLM scores whether chunks were useful: supported / unsupported / partial
  reflect_answer      — LLM scores answer on groundedness, relevance, completeness (0.0-1.0 each)
  regenerate          — Rewrites answer targeting the weakest scoring dimension (max 2×)
  answer              — Final output node with all reflection scores
"""

import os
import glob
import re
from typing import TypedDict, List

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
PAPERS_DIR       = os.path.join(os.path.dirname(__file__), "papers")
CHROMA_DIR       = os.path.join(os.path.dirname(__file__), "chroma_db_main5")
EMBED_MODEL      = "all-MiniLM-L6-v2"
GROQ_MODEL       = "llama-3.1-8b-instant"
TOP_K            = 5
MAX_RETRIEVAL_RETRIES  = 2   # max times we re-retrieve on unsupported
MAX_REGEN_ATTEMPTS     = 2   # max regeneration passes
SCORE_THRESHOLD        = 0.7


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question:           str         # current (possibly rewritten) query
    original_question:  str         # never modified
    retrieval_retries:  int         # how many times we've re-retrieved
    regen_count:        int         # how many times we've regenerated
    chunks:             List[str]   # page_content of retrieved chunks
    chunk_sources:      List[str]   # filename per chunk
    context:            str         # formatted context passed to generator
    answer:             str         # current generated answer
    retrieval_score:    str         # "supported" | "unsupported" | "partial"
    retrieval_reason:   str         # LLM reasoning for retrieval score
    groundedness:       float       # 0.0 – 1.0
    relevance:          float       # 0.0 – 1.0
    completeness:       float       # 0.0 – 1.0
    weak_dimensions:    List[str]   # dimensions that scored below threshold


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
# Helpers
# ---------------------------------------------------------------------------
def _parse_float(text: str, key: str) -> float:
    """Extract a float value for a labelled key from LLM output."""
    pattern = rf"{re.escape(key)}\s*[:\-=]\s*([0-9]+(?:\.[0-9]+)?)"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        # Accept values in 0-1 or 0-10 range
        return round(val / 10.0, 2) if val > 1.0 else round(val, 2)
    return 0.5   # neutral default when parsing fails


# ---------------------------------------------------------------------------
# Node 1 — retrieve
# ---------------------------------------------------------------------------
def retrieve(state: AgentState) -> AgentState:
    question = state["question"]
    retry    = state.get("retrieval_retries", 0)

    label = f"(retry #{retry})" if retry > 0 else ""
    print(f"\n[Node: retrieve] {label}")
    print(f"  Query : {question!r}")
    print(f"  Fetching top {TOP_K} chunks from ChromaDB …")

    results = VECTORSTORE.similarity_search(question, k=TOP_K)
    chunks  = [doc.page_content.strip() for doc in results]
    sources = [os.path.basename(doc.metadata.get("source", "unknown")) for doc in results]

    context_parts = []
    for i, (src, chunk) in enumerate(zip(sources, chunks), 1):
        preview = chunk[:100].replace("\n", " ")
        print(f"  Chunk {i} [{src}]: {preview} …")
        context_parts.append(f"[{i}] ({src})\n{chunk}")

    context = "\n\n".join(context_parts)

    return {
        **state,
        "chunks":         chunks,
        "chunk_sources":  sources,
        "context":        context,
        # reset downstream scores on each fresh retrieval
        "answer":             "",
        "retrieval_score":    "",
        "retrieval_reason":   "",
        "groundedness":       0.0,
        "relevance":          0.0,
        "completeness":       0.0,
        "weak_dimensions":    [],
    }


# ---------------------------------------------------------------------------
# Node 2 — generate
# ---------------------------------------------------------------------------
def generate(state: AgentState) -> AgentState:
    question = state["original_question"]
    context  = state["context"]
    regen    = state.get("regen_count", 0)

    label = f"(regen #{regen})" if regen > 0 else "(initial)"
    print(f"\n[Node: generate] {label}")

    system_prompt = (
        "You are a helpful research assistant. "
        "Answer the question using ONLY the provided context excerpts. "
        "Be accurate, concise, and ground every claim in the context."
    )
    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    response = LLM.invoke(messages)
    ans = response.content.strip()

    preview = ans[:120].replace("\n", " ")
    print(f"  Generated: {preview} …")

    return {**state, "answer": ans}


# ---------------------------------------------------------------------------
# Node 3 — reflect_retrieval
# ---------------------------------------------------------------------------
REFLECT_RETRIEVAL_PROMPT = """You are a retrieval quality judge.

Given a question and a set of retrieved text chunks, decide whether the chunks
are useful for answering the question:

  supported   — the chunks directly and sufficiently support answering the question
  partial     — the chunks are somewhat relevant but incomplete or tangential
  unsupported — the chunks are off-topic and do not help answer the question

Reply in this exact format (two lines):
SCORE: <supported|partial|unsupported>
REASON: <one sentence explanation>"""


def reflect_retrieval(state: AgentState) -> AgentState:
    question = state["original_question"]
    context  = state["context"]

    print(f"\n[Node: reflect_retrieval]")

    messages = [
        SystemMessage(content=REFLECT_RETRIEVAL_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nRetrieved chunks:\n{context[:2000]}"),
    ]
    response = LLM.invoke(messages)
    text = response.content.strip()

    score  = "partial"   # safe default
    reason = ""
    for line in text.splitlines():
        if line.upper().startswith("SCORE:"):
            raw = line.split(":", 1)[1].strip().lower()
            if "unsupported" in raw:
                score = "unsupported"
            elif "supported" in raw and "un" not in raw:
                score = "supported"
            else:
                score = "partial"
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    print(f"  Retrieval score : {score.upper()}")
    print(f"  Reason          : {reason}")

    return {**state, "retrieval_score": score, "retrieval_reason": reason}


# ---------------------------------------------------------------------------
# Routing after reflect_retrieval
# ---------------------------------------------------------------------------
def route_after_retrieval_reflection(state: AgentState) -> str:
    score   = state["retrieval_score"]
    retries = state.get("retrieval_retries", 0)

    print(f"\n[Routing: retrieval] score={score!r}, retries={retries}")

    if score == "unsupported" and retries < MAX_RETRIEVAL_RETRIES:
        print("  → re_retrieve  (unsupported chunks, will rewrite query)")
        return "re_retrieve"

    print("  → reflect_answer  (proceeding to answer reflection)")
    return "reflect_answer"


# ---------------------------------------------------------------------------
# Node: re_retrieve  (rewrite query then retrieve again)
# ---------------------------------------------------------------------------
REWRITE_PROMPT = """You are a query rewriter. The original question did not yield useful
retrieval results. Rewrite the question to be more specific and use different terminology
that is more likely to match relevant academic paper content.

Reply with only the rewritten question — no explanations, no quotes."""


def re_retrieve(state: AgentState) -> AgentState:
    original  = state["original_question"]
    current   = state["question"]
    retries   = state.get("retrieval_retries", 0)

    print(f"\n[Node: re_retrieve]  (attempt {retries + 1}/{MAX_RETRIEVAL_RETRIES})")
    print(f"  Current query : {current!r}")

    messages = [
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=f"Original: {original}\nCurrent: {current}"),
    ]
    response = LLM.invoke(messages)
    new_q = response.content.strip().strip('"').strip("'")
    print(f"  Rewritten to  : {new_q!r}")

    # Update question and bump retry counter, then retrieve will run next
    new_state = {**state, "question": new_q, "retrieval_retries": retries + 1}

    # Inline retrieval so the graph edge goes back to reflect_retrieval
    results  = VECTORSTORE.similarity_search(new_q, k=TOP_K)
    chunks   = [doc.page_content.strip() for doc in results]
    sources  = [os.path.basename(doc.metadata.get("source", "unknown")) for doc in results]
    ctx_parts = [f"[{i}] ({src})\n{c}" for i, (src, c) in enumerate(zip(sources, chunks), 1)]
    context  = "\n\n".join(ctx_parts)

    for i, (src, chunk) in enumerate(zip(sources, chunks), 1):
        preview = chunk[:100].replace("\n", " ")
        print(f"  Chunk {i} [{src}]: {preview} …")

    return {
        **new_state,
        "chunks":         chunks,
        "chunk_sources":  sources,
        "context":        context,
        "answer":         "",
    }


# ---------------------------------------------------------------------------
# Node 4 — reflect_answer
# ---------------------------------------------------------------------------
REFLECT_ANSWER_PROMPT = """You are an answer quality evaluator. Score the generated answer
on three dimensions, each on a scale from 0.0 to 1.0:

  groundedness  — every claim in the answer is directly supported by the context (1.0 = fully grounded)
  relevance     — the answer actually addresses the question asked (1.0 = fully relevant)
  completeness  — the answer covers all important aspects of the question (1.0 = fully complete)

Reply in exactly this format (three lines, no extra text):
GROUNDEDNESS: <score>
RELEVANCE: <score>
COMPLETENESS: <score>"""


def reflect_answer(state: AgentState) -> AgentState:
    question = state["original_question"]
    context  = state["context"]
    answer   = state["answer"]
    regen    = state.get("regen_count", 0)

    print(f"\n[Node: reflect_answer]  (regen_count={regen})")

    messages = [
        SystemMessage(content=REFLECT_ANSWER_PROMPT),
        HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Context (excerpts):\n{context[:2000]}\n\n"
                f"Answer:\n{answer}"
            )
        ),
    ]
    response = LLM.invoke(messages)
    text = response.content.strip()

    g = _parse_float(text, "groundedness")
    r = _parse_float(text, "relevance")
    c = _parse_float(text, "completeness")

    weak = [dim for dim, score in [("groundedness", g), ("relevance", r), ("completeness", c)]
            if score < SCORE_THRESHOLD]

    print(f"  Groundedness : {g:.2f}  {'✓' if g >= SCORE_THRESHOLD else '✗'}")
    print(f"  Relevance    : {r:.2f}  {'✓' if r >= SCORE_THRESHOLD else '✗'}")
    print(f"  Completeness : {c:.2f}  {'✓' if c >= SCORE_THRESHOLD else '✗'}")
    if weak:
        print(f"  Weak dims    : {', '.join(weak)}")

    return {
        **state,
        "groundedness":    g,
        "relevance":       r,
        "completeness":    c,
        "weak_dimensions": weak,
    }


# ---------------------------------------------------------------------------
# Routing after reflect_answer
# ---------------------------------------------------------------------------
def route_after_answer_reflection(state: AgentState) -> str:
    weak  = state["weak_dimensions"]
    regen = state.get("regen_count", 0)

    print(f"\n[Routing: answer] weak={weak}, regen_count={regen}")

    if weak and regen < MAX_REGEN_ATTEMPTS:
        print(f"  → regenerate  (improving: {', '.join(weak)})")
        return "regenerate"

    print("  → answer  (scores acceptable or max regenerations reached)")
    return "answer"


# ---------------------------------------------------------------------------
# Node 5 — regenerate
# ---------------------------------------------------------------------------
_DIM_INSTRUCTIONS = {
    "groundedness":  "Ensure EVERY claim is directly backed by the provided context. Remove any unsupported statements.",
    "relevance":     "Make sure the answer directly and completely addresses the specific question asked.",
    "completeness":  "Cover ALL important aspects of the question. Do not omit key details present in the context.",
}


def regenerate(state: AgentState) -> AgentState:
    question   = state["original_question"]
    context    = state["context"]
    prev_ans   = state["answer"]
    weak       = state["weak_dimensions"]
    regen      = state.get("regen_count", 0)

    print(f"\n[Node: regenerate]  (attempt {regen + 1}/{MAX_REGEN_ATTEMPTS})")
    print(f"  Targeting weak dimensions: {', '.join(weak)}")

    focused_instructions = "\n".join(f"- {_DIM_INSTRUCTIONS[d]}" for d in weak if d in _DIM_INSTRUCTIONS)

    system_prompt = (
        "You are a helpful research assistant improving a prior answer. "
        "Use ONLY the provided context. Address the specific weaknesses listed.\n\n"
        f"Improvement instructions:\n{focused_instructions}"
    )
    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Previous answer (needs improvement):\n{prev_ans}\n\n"
        "Write an improved answer:"
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    response = LLM.invoke(messages)
    new_ans = response.content.strip()

    preview = new_ans[:120].replace("\n", " ")
    print(f"  Regenerated: {preview} …")

    return {**state, "answer": new_ans, "regen_count": regen + 1}


# ---------------------------------------------------------------------------
# Node 6 — answer (final output node)
# ---------------------------------------------------------------------------
def answer(state: AgentState) -> AgentState:
    # This node is a pass-through — all printing is done in main()
    print(f"\n[Node: answer]  Final answer ready.")
    return state


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve",           retrieve)
    graph.add_node("generate",           generate)
    graph.add_node("reflect_retrieval",  reflect_retrieval)
    graph.add_node("re_retrieve",        re_retrieve)
    graph.add_node("reflect_answer",     reflect_answer)
    graph.add_node("regenerate",         regenerate)
    graph.add_node("answer",             answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",          "generate")
    graph.add_edge("generate",          "reflect_retrieval")

    graph.add_conditional_edges(
        "reflect_retrieval",
        route_after_retrieval_reflection,
        {
            "re_retrieve":    "re_retrieve",
            "reflect_answer": "reflect_answer",
        },
    )

    # After re_retrieve, re-run generate then reflect_retrieval
    graph.add_edge("re_retrieve",       "generate")

    graph.add_conditional_edges(
        "reflect_answer",
        route_after_answer_reflection,
        {
            "regenerate": "regenerate",
            "answer":     "answer",
        },
    )

    graph.add_edge("regenerate",        "reflect_answer")
    graph.add_edge("answer",            END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    app = build_graph()

    questions = [
        "How does Self-RAG decide when to retrieve?",
        "What is the capital of Germany?",
        "Explain the mathematical formulation of RAG retrieval scoring",
        "How do LLM agents handle multi-step reasoning?",
    ]

    print("\n" + "=" * 65)
    print("  AgentMind — Day 5: Self-RAG")
    print("=" * 65)

    for i, q in enumerate(questions, 1):
        print(f"\n{'━' * 65}")
        print(f"  Question {i}: {q}")
        print(f"{'━' * 65}")

        initial_state: AgentState = {
            "question":          q,
            "original_question": q,
            "retrieval_retries": 0,
            "regen_count":       0,
            "chunks":            [],
            "chunk_sources":     [],
            "context":           "",
            "answer":            "",
            "retrieval_score":   "",
            "retrieval_reason":  "",
            "groundedness":      0.0,
            "relevance":         0.0,
            "completeness":      0.0,
            "weak_dimensions":   [],
        }

        final = app.invoke(initial_state)

        g = final["groundedness"]
        r = final["relevance"]
        c = final["completeness"]
        avg = round((g + r + c) / 3, 2)

        print(f"\n  ── Reflection Summary ───────────────────────────────────")
        print(f"  Retrieval score  : {final['retrieval_score'].upper()}")
        print(f"  Retrieval retries: {final['retrieval_retries']}")
        print(f"  Regen attempts   : {final['regen_count']}")
        print(f"  Groundedness     : {g:.2f}")
        print(f"  Relevance        : {r:.2f}")
        print(f"  Completeness     : {c:.2f}")
        print(f"  Avg quality score: {avg:.2f}")
        print(f"\n  Answer:\n  {final['answer']}")

    print(f"\n{'=' * 65}")
    print("  All questions processed.")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
