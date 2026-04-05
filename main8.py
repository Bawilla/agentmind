"""
AgentMind — Day 8: Streaming Responses

Demonstrates real-time token-by-token streaming from the CRAG pipeline:
  - Spinner during retrieval and chunk-grading steps
  - Answer tokens printed immediately as they arrive via LLM.stream()
  - Same CRAG routing logic as Day 4 (retrieve → grade → web_search/rewrite → answer)
"""

import os
import glob
import sys
import time
import threading
import itertools
from typing import TypedDict, List

import groq as _groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from ddgs import DDGS

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(__file__)
PAPERS_DIR   = os.path.join(BASE_DIR, "papers")
CHROMA_DIR   = os.path.join(BASE_DIR, "chroma_db_main4")   # reuse Day 4 store
EMBED_MODEL  = "all-MiniLM-L6-v2"
GROQ_MODEL   = "llama-3.1-8b-instant"
TOP_K        = 5
MAX_REWRITES = 2

GRADE_PROMPT = """You are a relevance grader. Given a question and a retrieved text chunk,
decide if the chunk is:
  relevant   — directly answers or strongly supports answering the question
  irrelevant — off-topic, unrelated to the question
  ambiguous  — partially related but not directly answering

Reply with exactly one word: relevant, irrelevant, or ambiguous."""

REWRITE_PROMPT = """You are a query rewriter. The original question did not yield relevant
retrieval results. Rewrite the question to be more specific, use different terminology,
and focus on the core information need.

Reply with only the rewritten question — no explanations, no quotes."""


# ---------------------------------------------------------------------------
# Spinner — shows an animated indicator during blocking steps
# ---------------------------------------------------------------------------
class Spinner:
    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str = ""):
        self.message  = message
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "Spinner":
        self._running = True
        self._thread  = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self):
        for frame in itertools.cycle(self._FRAMES):
            if not self._running:
                break
            sys.stdout.write(f"\r  {frame}  {self.message}   ")
            sys.stdout.flush()
            time.sleep(0.1)

    def stop(self, final: str = ""):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        label = final or self.message
        sys.stdout.write(f"\r  ✓  {label}\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
def _load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("[VectorStore] Loading existing ChromaDB …")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    print("[VectorStore] Building ChromaDB …")
    pdf_paths = glob.glob(os.path.join(PAPERS_DIR, "*.pdf"))
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    store  = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    return store


VECTORSTORE = _load_vectorstore()
LLM         = ChatGroq(model=GROQ_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Rate-limit-safe invoke (non-streaming path)
# ---------------------------------------------------------------------------
def _llm_invoke(messages: list):
    for attempt in range(3):
        try:
            time.sleep(5)
            return LLM.invoke(messages)
        except _groq.RateLimitError:
            if attempt == 2:
                raise
            wait = 10 * (2 ** attempt)
            print(f"\n  [RateLimit] waiting {wait}s …")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Streaming CRAG pipeline
# ---------------------------------------------------------------------------
def run_streaming(question: str) -> None:
    """
    Runs the full CRAG pipeline with:
      - Spinner output during blocking steps
      - Token-by-token streaming for the final answer
    """
    original_question = question
    current_question  = question
    rewrite_count     = 0

    print(f"\n  Question: {question!r}")

    while True:
        # ── Retrieve ──────────────────────────────────────────────────────
        spinner = Spinner(f"Retrieving top {TOP_K} chunks …").start()
        results = VECTORSTORE.similarity_search(current_question, k=TOP_K)
        chunks  = [doc.page_content.strip() for doc in results]
        sources = [os.path.basename(doc.metadata.get("source", "unknown")) for doc in results]
        pages   = [str(doc.metadata.get("page", "?")) for doc in results]
        spinner.stop(f"Retrieved {len(chunks)} chunks")

        # ── Grade chunks ──────────────────────────────────────────────────
        grades: List[str] = []
        for i, (chunk, src, pg) in enumerate(zip(chunks, sources, pages), 1):
            spinner = Spinner(f"Grading chunk {i}/{len(chunks)} [{src} p.{pg}] …").start()
            messages = [
                SystemMessage(content=GRADE_PROMPT),
                HumanMessage(content=f"Question: {current_question}\n\nChunk:\n{chunk[:600]}"),
            ]
            response = _llm_invoke(messages)
            raw   = response.content.strip().lower()
            grade = ("irrelevant" if "irrelevant" in raw
                     else "ambiguous" if "ambiguous" in raw
                     else "relevant")
            grades.append(grade)
            spinner.stop(f"Chunk {i} [{src}] → {grade.upper()}")

        # ── Route ─────────────────────────────────────────────────────────
        has_irrelevant = any(g == "irrelevant" for g in grades)
        all_ambiguous  = all(g == "ambiguous"  for g in grades)

        if all_ambiguous and rewrite_count < MAX_REWRITES:
            spinner = Spinner(f"All ambiguous — rewriting query (attempt {rewrite_count + 1}) …").start()
            messages = [
                SystemMessage(content=REWRITE_PROMPT),
                HumanMessage(content=f"Original: {original_question}\nCurrent: {current_question}"),
            ]
            response = _llm_invoke(messages)
            current_question = response.content.strip().strip('"').strip("'")
            rewrite_count += 1
            spinner.stop(f"Rewritten → {current_question!r}")
            continue   # loop back to retrieve

        web_context = ""
        if has_irrelevant or (all_ambiguous and rewrite_count >= MAX_REWRITES):
            spinner = Spinner("Searching web (DuckDuckGo) …").start()
            web_results = []
            try:
                ddgs = DDGS()
                for r in ddgs.text(original_question, max_results=3):
                    web_results.append(r)
            except Exception as exc:
                spinner.stop(f"Web search error: {exc}")
            if web_results:
                web_context = "\n\n".join(
                    f"[Web {i}] {r.get('title','')}\n{r.get('body','')}"
                    for i, r in enumerate(web_results, 1)
                )
                spinner.stop(f"Got {len(web_results)} web results")
            else:
                spinner.stop("No web results")

        break   # exit retrieval/grading loop

    # ── Build context ──────────────────────────────────────────────────────
    kept_parts = [
        f"[Paper chunk {i} | {src}]\n{chunk.strip()}"
        for i, (chunk, src, grade) in enumerate(zip(chunks, sources, grades), 1)
        if grade != "irrelevant"
    ]
    rag_context = "\n\n".join(kept_parts)
    sections = []
    if rag_context:
        sections.append("=== Retrieved Paper Context ===\n" + rag_context)
    if web_context:
        sections.append("=== Web Search Results ===\n" + web_context)
    combined = "\n\n".join(sections)

    # ── Stream answer ──────────────────────────────────────────────────────
    if combined:
        system_prompt = (
            "You are a helpful research assistant. "
            "Answer the question using the provided context. "
            "Integrate web results with paper excerpts where both are present. "
            "Be concise and accurate."
        )
        user_content = f"Context:\n{combined}\n\nQuestion: {original_question}"
    else:
        system_prompt = "You are a helpful assistant. Answer concisely from general knowledge."
        user_content  = original_question

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]

    print("\n  ── Streaming Answer ─────────────────────────────────────────")
    print("  ", end="", flush=True)

    full_answer = ""
    t0 = time.perf_counter()

    time.sleep(5)   # rate-limit pre-call delay
    try:
        for chunk in LLM.stream(messages):
            token = chunk.content
            if token:
                full_answer += token
                sys.stdout.write(token)
                sys.stdout.flush()
    except _groq.RateLimitError:
        print("\n  [RateLimit] falling back to invoke …")
        response = _llm_invoke(messages)
        full_answer = response.content.strip()
        sys.stdout.write(full_answer)
        sys.stdout.flush()

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # ── Summary ────────────────────────────────────────────────────────────
    grade_counts = {g: grades.count(g) for g in ("relevant", "irrelevant", "ambiguous")}
    has_rag = any(g != "irrelevant" for g in grades)
    has_web = bool(web_context)
    tool    = "both" if (has_rag and has_web) else "web_search" if has_web else "retrieval"

    print(f"\n\n  ── Summary ──────────────────────────────────────────────────")
    print(f"  Tool used    : {tool}")
    print(f"  Chunk grades : {grade_counts}")
    print(f"  Rewrites     : {rewrite_count}")
    print(f"  Stream time  : {latency_ms}ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    questions = [
        "How does corrective RAG handle irrelevant chunks?",
        "What is the ReAct framework for LLM reasoning?",
    ]

    print("\n" + "=" * 65)
    print("  AgentMind — Day 8: Streaming Responses")
    print("=" * 65)

    for i, q in enumerate(questions, 1):
        print(f"\n{'━' * 65}")
        print(f"  Question {i}/{len(questions)}")
        run_streaming(q)

    print(f"\n{'=' * 65}")
    print("  Done.")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
