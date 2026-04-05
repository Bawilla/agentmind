# AgentMind — Architecture

## System Architecture

```mermaid
graph TB
    User(["👤 User / Client"])

    subgraph Docker["🐳 Docker Container  (port 8000)"]
        subgraph API["FastAPI Layer  —  api.py"]
            EP1["POST /ask"]
            EP2["POST /ask/stream"]
            EP3["GET /history/{session_id}"]
            EP4["DELETE /history/{session_id}"]
            EP5["GET /health"]
            EP6["POST /upload"]
        end

        subgraph CRAG["LangGraph — Corrective RAG  (Day 4 / 7 / 8)"]
            CR1["retrieve\n(ChromaDB similarity search)"]
            CR2["grade_chunks\n(LLM grades each chunk:\nrelevant / irrelevant / ambiguous)"]
            CR3["web_search\n(DuckDuckGo fallback)"]
            CR4["rewrite_query\n(LLM rewrites on all-ambiguous)"]
            CR5["answer\n(Groq generates final answer)"]

            CR1 --> CR2
            CR2 -->|"any irrelevant"| CR3
            CR2 -->|"all ambiguous\n(max 2×)"| CR4
            CR2 -->|"has relevant"| CR5
            CR3 --> CR5
            CR4 --> CR1
        end

        subgraph SELFRAG["LangGraph — Self-RAG  (Day 5)"]
            SR1["retrieve\n(ChromaDB similarity search)"]
            SR2["generate\n(Groq initial answer)"]
            SR3["reflect_retrieval\n(supported / partial / unsupported)"]
            SR4["re_retrieve\n(rewrite + fetch, max 2×)"]
            SR5["reflect_answer\n(groundedness / relevance\n/ completeness  0–1)"]
            SR6["regenerate\n(targeted fix, max 2×)"]
            SR7["answer\n(final output)"]

            SR1 --> SR2
            SR2 --> SR3
            SR3 -->|"unsupported"| SR4
            SR3 -->|"supported / partial"| SR5
            SR4 --> SR2
            SR5 -->|"any score < 0.7"| SR6
            SR5 -->|"all scores ≥ 0.7"| SR7
            SR6 --> SR5
        end

        SS["💾 Session Store\n(in-memory dict)"]
        EP1 --> CRAG
        EP2 --> CRAG
        EP3 --> SS
        EP4 --> SS
        CRAG --> SS
    end

    subgraph External["External Services"]
        CHROMA["🗄️ ChromaDB\n(persisted volume)"]
        HF["🤗 HuggingFace\nall-MiniLM-L6-v2\n(embeddings)"]
        GROQ["⚡ Groq API\nllama-3.1-8b-instant\n(LLM inference)"]
        DDG["🦆 DuckDuckGo\n(web search)"]
        LS["🔭 LangSmith\n(tracing / observability)"]
    end

    User -->|"HTTP request"| API
    API -->|"SSE stream"| User

    CR1 -->|"similarity_search()"| CHROMA
    SR1 -->|"similarity_search()"| CHROMA
    EP6 -->|"add_documents()"| CHROMA
    HF -->|"embed chunks"| CHROMA

    CR2 -->|"LLM.invoke()"| GROQ
    CR4 -->|"LLM.invoke()"| GROQ
    CR5 -->|"LLM.invoke() / stream()"| GROQ
    SR2 -->|"LLM.invoke()"| GROQ
    SR3 -->|"LLM.invoke()"| GROQ
    SR5 -->|"LLM.invoke()"| GROQ
    SR6 -->|"LLM.invoke()"| GROQ

    CR3 -->|"ddgs.text()"| DDG

    CRAG -.->|"@traceable"| LS
    GROQ -.->|"token counts\n& latency"| LS
```

## Data Flow — Single Request

```mermaid
sequenceDiagram
    actor User
    participant API as FastAPI
    participant Graph as LangGraph (CRAG)
    participant DB as ChromaDB
    participant LLM as Groq LLM
    participant Web as DuckDuckGo
    participant LS as LangSmith

    User->>API: POST /ask  {"question": "..."}
    API->>Graph: invoke(initial_state)

    Graph->>DB: similarity_search(question, k=5)
    DB-->>Graph: 5 chunks + metadata

    loop Grade each chunk (5×)
        Graph->>LLM: grade prompt + chunk
        LLM-->>Graph: "relevant" | "irrelevant" | "ambiguous"
        Graph-)LS: log grade + latency
    end

    alt any irrelevant
        Graph->>Web: ddgs.text(question, max_results=3)
        Web-->>Graph: title + snippet × 3
    else all ambiguous (≤ 2 rewrites)
        Graph->>LLM: rewrite query
        LLM-->>Graph: rewritten question
        Graph->>DB: similarity_search(new_question, k=5)
    end

    Graph->>LLM: answer prompt (paper context + web context)
    LLM-->>Graph: final answer
    Graph-)LS: log full run

    Graph-->>API: final_state
    API-->>User: {"answer": "...", "sources": [...], "latency_ms": ...}
```

## Component Responsibilities

| Component | File | Role |
|-----------|------|------|
| FastAPI app | `api.py` | HTTP routing, session management, PDF upload |
| CRAG pipeline | `api.py`, `main4.py` | Corrective retrieval loop with chunk grading |
| Self-RAG pipeline | `main5.py` | Reflection-based answer quality scoring |
| Multi-tool agent | `main3.py` | Routes to retrieval / web search / calculator |
| Tracing utilities | `tracing.py` | LangSmith init, `@trace_step` decorator |
| Streaming | `main8.py` | Terminal spinner + `LLM.stream()` token output |
| Docker entry point | `main7.py` | Uvicorn startup banner |
| Vector store | `chroma_db_main4/` | Persisted ChromaDB index (5 papers, 738 chunks) |
