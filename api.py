"""
AgentMind — api.py

FastAPI wrapper around the Day 4 Corrective RAG (CRAG) pipeline.

Endpoints:
  POST   /ask                    — run full CRAG pipeline (blocking)
  POST   /ask/stream             — run CRAG pipeline with SSE token streaming
  GET    /history/{session_id}   — retrieve session exchange history
  DELETE /history/{session_id}   — clear session history
  GET    /health                 — liveness check
  POST   /upload                 — add a PDF to ChromaDB at runtime
"""

import os
import glob
import time
import uuid
import json
import tempfile
from contextlib import asynccontextmanager
from typing import TypedDict, List, Dict, Optional, Generator

import groq as _groq
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import mlflow_tracker
import sagemaker_tracker
from monitoring import drift_monitor

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
BASE_DIR    = os.path.dirname(__file__)
PAPERS_DIR  = os.path.join(BASE_DIR, "papers")
CHROMA_DIR  = os.path.join(BASE_DIR, "chroma_db_main4")   # reuse Day 4 store
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.1-8b-instant"
TOP_K       = 5
MAX_REWRITES = 2

# ---------------------------------------------------------------------------
# Global singletons (initialised in lifespan)
# ---------------------------------------------------------------------------
VECTORSTORE: Optional[Chroma] = None
EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
LLM: Optional[ChatGroq] = None
CRAG_APP = None                          # compiled LangGraph

# In-memory session store: session_id → list of {"question": ..., "answer": ...}
SESSION_STORE: Dict[str, List[dict]] = {}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    tool_used: str
    chunk_grades: Dict[str, int]
    session_id: str
    latency_ms: float

class HistoryResponse(BaseModel):
    session_id: str
    exchanges: List[Dict[str, str]]

class DeleteResponse(BaseModel):
    status: str
    session_id: str

class HealthResponse(BaseModel):
    status: str
    model: str
    vectorstore: str

class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    status: str


# ---------------------------------------------------------------------------
# CRAG state schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question:          str
    original_question: str
    rewrite_count:     int
    chunks:            List[str]
    chunk_sources:     List[str]
    chunk_pages:       List[str]   # page numbers for sources response
    grades:            List[str]
    relevant_context:  str
    web_context:       str
    answer:            str


# ---------------------------------------------------------------------------
# Rate-limit-safe LLM helper
# ---------------------------------------------------------------------------
def _llm_invoke(messages: list):
    """Invoke LLM with 5s pre-call sleep and exponential backoff (max 3 retries)."""
    for attempt in range(3):
        try:
            time.sleep(5)
            return LLM.invoke(messages)
        except _groq.RateLimitError:
            if attempt == 2:
                raise
            wait = 10 * (2 ** attempt)
            print(f"  [RateLimit] Groq rate limit — waiting {wait}s …")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Vector store helpers
# ---------------------------------------------------------------------------
def _load_or_build_vectorstore() -> Chroma:
    global EMBEDDINGS
    EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("[VectorStore] Loading existing ChromaDB from", CHROMA_DIR)
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDINGS)

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

    store = Chroma.from_documents(chunks, EMBEDDINGS, persist_directory=CHROMA_DIR)
    print("[VectorStore] ChromaDB saved to", CHROMA_DIR)
    return store


# ---------------------------------------------------------------------------
# CRAG graph nodes
# ---------------------------------------------------------------------------
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


def _retrieve(state: AgentState) -> AgentState:
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)
    label = f"(rewrite #{rewrite_count})" if rewrite_count > 0 else ""
    print(f"\n[Node: retrieve] {label}  query={question!r}")

    results = VECTORSTORE.similarity_search(question, k=TOP_K)
    chunks  = [doc.page_content.strip() for doc in results]
    sources = [os.path.basename(doc.metadata.get("source", "unknown")) for doc in results]
    pages   = [str(doc.metadata.get("page", "?")) for doc in results]

    for i, (src, pg) in enumerate(zip(sources, pages), 1):
        print(f"  Chunk {i} [{src} p.{pg}]")

    return {
        **state,
        "chunks":          chunks,
        "chunk_sources":   sources,
        "chunk_pages":     pages,
        "grades":          [],
        "relevant_context": "",
        "web_context":     "",
    }


def _grade_chunks(state: AgentState) -> AgentState:
    question = state["question"]
    chunks   = state["chunks"]
    sources  = state["chunk_sources"]
    print(f"\n[Node: grade_chunks]  grading {len(chunks)} chunks")

    grades: List[str] = []
    for i, (chunk, src) in enumerate(zip(chunks, sources), 1):
        messages = [
            SystemMessage(content=GRADE_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nChunk:\n{chunk[:600]}"),
        ]
        response = _llm_invoke(messages)
        raw = response.content.strip().lower()
        if "irrelevant" in raw:
            grade = "irrelevant"
        elif "ambiguous" in raw:
            grade = "ambiguous"
        else:
            grade = "relevant"
        grades.append(grade)
        print(f"  Chunk {i} [{src}] → {grade.upper()}")

    return {**state, "grades": grades}


def _route_after_grading(state: AgentState) -> str:
    grades = state["grades"]
    rewrite_count = state.get("rewrite_count", 0)
    has_irrelevant = any(g == "irrelevant" for g in grades)
    all_ambiguous  = all(g == "ambiguous"  for g in grades)

    if all_ambiguous and rewrite_count < MAX_REWRITES:
        return "rewrite_query"
    if has_irrelevant or (all_ambiguous and rewrite_count >= MAX_REWRITES):
        return "web_search"
    return "answer"


def _web_search(state: AgentState) -> AgentState:
    question = state["original_question"]
    print(f"\n[Node: web_search]  query={question!r}")

    results = []
    try:
        ddgs = DDGS()
        for r in ddgs.text(question, max_results=3):
            results.append(r)
    except Exception as exc:
        print(f"  DuckDuckGo error: {exc}")

    if results:
        parts = [f"[Web {i}] {r.get('title', '')}\n{r.get('body', '')}"
                 for i, r in enumerate(results, 1)]
        web_context = "\n\n".join(parts)
        print(f"  Got {len(results)} web results.")
    else:
        web_context = ""
        print("  No web results.")

    return {**state, "web_context": web_context}


def _rewrite_query(state: AgentState) -> AgentState:
    original      = state["original_question"]
    current       = state["question"]
    rewrite_count = state.get("rewrite_count", 0)
    print(f"\n[Node: rewrite_query]  attempt {rewrite_count + 1}/{MAX_REWRITES}")

    messages = [
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=f"Original: {original}\nCurrent: {current}"),
    ]
    response = _llm_invoke(messages)
    new_question = response.content.strip().strip('"').strip("'")
    print(f"  Rewritten: {new_question!r}")

    return {**state, "question": new_question, "rewrite_count": rewrite_count + 1}


def _answer(state: AgentState) -> AgentState:
    question = state["original_question"]
    grades   = state["grades"]
    chunks   = state["chunks"]
    sources  = state["chunk_sources"]
    web_ctx  = state.get("web_context", "")

    kept_parts = [
        f"[Paper chunk {i} | {src}]\n{chunk.strip()}"
        for i, (chunk, src, grade) in enumerate(zip(chunks, sources, grades), 1)
        if grade != "irrelevant"
    ]
    rag_context = "\n\n".join(kept_parts)

    sections = []
    if rag_context:
        sections.append("=== Retrieved Paper Context ===\n" + rag_context)
    if web_ctx:
        sections.append("=== Web Search Results ===\n" + web_ctx)
    combined = "\n\n".join(sections)

    print(f"\n[Node: answer]  kept={len(kept_parts)}/{len(grades)}, web={'yes' if web_ctx else 'no'}")

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
    response = _llm_invoke(messages)
    return {**state, "answer": response.content.strip()}


# ---------------------------------------------------------------------------
# Build CRAG graph
# ---------------------------------------------------------------------------
def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve",      _retrieve)
    graph.add_node("grade_chunks",  _grade_chunks)
    graph.add_node("web_search",    _web_search)
    graph.add_node("rewrite_query", _rewrite_query)
    graph.add_node("answer",        _answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_chunks")
    graph.add_conditional_edges(
        "grade_chunks",
        _route_after_grading,
        {"answer": "answer", "web_search": "web_search", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("web_search",    "answer")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("answer",        END)
    return graph.compile()


# ---------------------------------------------------------------------------
# FastAPI lifespan — initialise everything on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global VECTORSTORE, LLM, CRAG_APP
    print("\n[Startup] Initialising AgentMind API …")
    VECTORSTORE = _load_or_build_vectorstore()
    LLM         = ChatGroq(model=GROQ_MODEL, temperature=0)
    CRAG_APP    = _build_graph()
    mlflow_tracker.init()
    sagemaker_tracker.init()
    print("[Startup] Ready.\n")
    yield
    print("[Shutdown] AgentMind API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AgentMind API",
    description="Corrective RAG pipeline exposed as a REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    session_id = req.session_id or str(uuid.uuid4())
    t0 = time.perf_counter()

    initial: AgentState = {
        "question":          req.question,
        "original_question": req.question,
        "rewrite_count":     0,
        "chunks":            [],
        "chunk_sources":     [],
        "chunk_pages":       [],
        "grades":            [],
        "relevant_context":  "",
        "web_context":       "",
        "answer":            "",
    }

    try:
        final = CRAG_APP.invoke(initial)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Build sources list: "filename:page"
    sources = list(dict.fromkeys(
        f"{src}:{pg}"
        for src, pg, grade in zip(
            final["chunk_sources"],
            final["chunk_pages"],
            final["grades"],
        )
        if grade != "irrelevant"
    ))

    # Determine tool_used
    has_rag = any(g != "irrelevant" for g in final["grades"])
    has_web = bool(final.get("web_context", ""))
    if has_rag and has_web:
        tool_used = "both"
    elif has_web:
        tool_used = "web_search"
    else:
        tool_used = "retrieval"

    chunk_grades = {
        "relevant":   final["grades"].count("relevant"),
        "irrelevant": final["grades"].count("irrelevant"),
        "ambiguous":  final["grades"].count("ambiguous"),
    }

    # Persist to session history
    SESSION_STORE.setdefault(session_id, []).append({
        "question": req.question,
        "answer":   final["answer"],
    })

    response = AskResponse(
        answer=final["answer"],
        sources=sources,
        tool_used=tool_used,
        chunk_grades=chunk_grades,
        session_id=session_id,
        latency_ms=latency_ms,
    )

    try:
        mlflow_tracker.log_ask_run(
            question=req.question,
            session_id=session_id,
            answer=response.answer,
            sources=response.sources,
            tool_used=response.tool_used,
            chunk_grades=response.chunk_grades,
            latency_ms=response.latency_ms,
        )
    except Exception:
        pass  # never break the API over tracking errors

    try:
        drift_monitor.log_query(
            question=req.question,
            answer=response.answer,
            latency_ms=response.latency_ms,
            tool_used=response.tool_used,
        )
    except Exception:
        pass  # never break the API over monitoring errors

    try:
        sagemaker_tracker.log_trial(
            question=req.question,
            answer=response.answer,
            tool_used=response.tool_used,
            chunk_grades=response.chunk_grades,
            latency_ms=response.latency_ms,
            model_name=GROQ_MODEL,
            retrieval_top_k=TOP_K,
        )
    except Exception:
        pass  # never break the API over tracking errors

    return response


# ---------------------------------------------------------------------------
# POST /ask/stream  (Server-Sent Events)
# ---------------------------------------------------------------------------
def _sse(data: dict) -> str:
    """Format a dict as a single SSE line."""
    return f"data: {json.dumps(data)}\n\n"


def _stream_crag_events(question: str, session_id: str) -> Generator[str, None, None]:
    """
    Sync generator that runs the full CRAG pipeline and yields SSE events:
      {"type": "status",  "message": "..."}
      {"type": "token",   "content": "..."}
      {"type": "done",    "sources": [...], "tool_used": "...",
                          "chunk_grades": {...}, "latency_ms": ...}
    """
    t0 = time.perf_counter()
    original_question = question
    current_question  = question
    rewrite_count     = 0
    chunks:  List[str] = []
    sources: List[str] = []
    pages:   List[str] = []
    grades:  List[str] = []
    web_context        = ""

    # ── Retrieval / grading loop (handles query rewrites) ──────────────────
    while True:
        yield _sse({"type": "status", "message": f"Retrieving chunks for: {current_question!r}"})
        results = VECTORSTORE.similarity_search(current_question, k=TOP_K)
        chunks  = [doc.page_content.strip() for doc in results]
        sources = [os.path.basename(doc.metadata.get("source", "unknown")) for doc in results]
        pages   = [str(doc.metadata.get("page", "?")) for doc in results]
        yield _sse({"type": "status", "message": f"Retrieved {len(chunks)} chunks"})

        # Grade each chunk
        yield _sse({"type": "status", "message": "Grading chunks..."})
        grades = []
        for i, (chunk, src) in enumerate(zip(chunks, sources), 1):
            yield _sse({"type": "status", "message": f"Grading chunk {i}/{len(chunks)} [{src}]..."})
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
            yield _sse({"type": "status", "message": f"Chunk {i} graded: {grade.upper()}"})

        # Routing decision
        has_irrelevant = any(g == "irrelevant" for g in grades)
        all_ambiguous  = all(g == "ambiguous"  for g in grades)

        if all_ambiguous and rewrite_count < MAX_REWRITES:
            yield _sse({"type": "status",
                        "message": f"All chunks ambiguous — rewriting query (attempt {rewrite_count + 1})..."})
            messages = [
                SystemMessage(content=REWRITE_PROMPT),
                HumanMessage(content=f"Original: {original_question}\nCurrent: {current_question}"),
            ]
            response = _llm_invoke(messages)
            current_question = response.content.strip().strip('"').strip("'")
            rewrite_count   += 1
            yield _sse({"type": "status", "message": f"Query rewritten to: {current_question!r}"})
            continue   # loop back to retrieve

        # Web search if needed
        web_context = ""
        if has_irrelevant or (all_ambiguous and rewrite_count >= MAX_REWRITES):
            yield _sse({"type": "status", "message": "Searching web (DuckDuckGo)..."})
            web_results: List[dict] = []
            try:
                ddgs = DDGS()
                for r in ddgs.text(original_question, max_results=3):
                    web_results.append(r)
            except Exception as exc:
                yield _sse({"type": "status", "message": f"Web search error: {exc}"})
            if web_results:
                web_context = "\n\n".join(
                    f"[Web {i}] {r.get('title','')}\n{r.get('body','')}"
                    for i, r in enumerate(web_results, 1)
                )
                yield _sse({"type": "status", "message": f"Got {len(web_results)} web results"})
            else:
                yield _sse({"type": "status", "message": "No web results returned"})

        break   # exit retrieval/grading loop

    # ── Build combined context ─────────────────────────────────────────────
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

    # ── Stream answer tokens ───────────────────────────────────────────────
    yield _sse({"type": "status", "message": "Generating answer..."})

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

    full_answer = ""
    time.sleep(5)   # rate-limit pre-call delay
    try:
        for chunk in LLM.stream(messages):
            token = chunk.content
            if token:
                full_answer += token
                yield _sse({"type": "token", "content": token})
    except _groq.RateLimitError:
        # Fall back to invoke with backoff
        response = _llm_invoke(messages)
        full_answer = response.content.strip()
        for word in full_answer.split(" "):
            yield _sse({"type": "token", "content": word + " "})

    # ── Persist session ────────────────────────────────────────────────────
    SESSION_STORE.setdefault(session_id, []).append({
        "question": original_question,
        "answer":   full_answer,
    })

    # ── Done event ─────────────────────────────────────────────────────────
    final_sources = list(dict.fromkeys(
        f"{src}:{pg}"
        for src, pg, grade in zip(sources, pages, grades)
        if grade != "irrelevant"
    ))
    has_rag   = any(g != "irrelevant" for g in grades)
    has_web   = bool(web_context)
    tool_used = "both" if (has_rag and has_web) else "web_search" if has_web else "retrieval"

    yield _sse({
        "type":        "done",
        "sources":     final_sources,
        "tool_used":   tool_used,
        "chunk_grades": {
            "relevant":   grades.count("relevant"),
            "irrelevant": grades.count("irrelevant"),
            "ambiguous":  grades.count("ambiguous"),
        },
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    })


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """
    Run the full CRAG pipeline and stream the response as Server-Sent Events.

    Event types:
      {"type": "status",  "message": "..."}        — pipeline progress
      {"type": "token",   "content": "..."}         — answer token
      {"type": "done",    "sources": [...], ...}    — final metadata
    """
    session_id = req.session_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_crag_events(req.question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /history/{session_id}
# ---------------------------------------------------------------------------
@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    exchanges = SESSION_STORE.get(session_id, [])
    return HistoryResponse(session_id=session_id, exchanges=exchanges)


# ---------------------------------------------------------------------------
# DELETE /history/{session_id}
# ---------------------------------------------------------------------------
@app.delete("/history/{session_id}", response_model=DeleteResponse)
def delete_history(session_id: str):
    SESSION_STORE.pop(session_id, None)
    return DeleteResponse(status="cleared", session_id=session_id)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model=GROQ_MODEL,
        vectorstore="chromadb",
    )


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save to a temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs   = loader.load()

        # Stamp the real filename in metadata so sources are readable
        for doc in docs:
            doc.metadata["source"] = file.filename

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks   = splitter.split_documents(docs)

        VECTORSTORE.add_documents(chunks)
        print(f"[Upload] Added {len(chunks)} chunks from {file.filename}")
    finally:
        os.unlink(tmp_path)

    return UploadResponse(
        filename=file.filename,
        chunks_added=len(chunks),
        status="ok",
    )
