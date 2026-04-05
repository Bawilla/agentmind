# AgentMind — Agentic RAG with LangGraph

AgentMind is a production-grade agentic RAG system that **reasons before retrieving** — it decides what to look up, judges whether the retrieved context is good enough, and loops back to refine its search if it isn't. Built on LangGraph for explicit state-machine control, Groq for fast LLM inference, and ChromaDB for local vector storage.

## Tech Stack

| Component | Library / Model |
|-----------|----------------|
| Agent orchestration | LangGraph |
| LLM inference | Groq — `llama-3.1-8b-instant` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (persisted locally) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LangChain integrations | langchain, langchain-community, langchain-groq |

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/bawilla/agentmind.git
cd agentmind

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install langgraph langchain langchain-community langchain-groq \
            chromadb sentence-transformers groq python-dotenv

# 4. Set your Groq API key
# Copy .env and fill in your key (get one free at console.groq.com)
echo "GROQ_API_KEY=your_key_here" > .env
```

## Configuration

Create a `.env` file in the project root (it is gitignored):

```
GROQ_API_KEY=your_key_here
```

## Project Structure

```
agentmind/
  main1.py     # Day 1 — Minimal 3-node LangGraph pipeline (state machine basics)
  main2.py     # Day 2 — Basic agentic RAG with conditional retrieval routing
  main3.py     # Day 3 — Multi-tool agent (retrieval, web search, calculator)
  main4.py     # Day 4 — Corrective RAG (CRAG) with chunk grading + web fallback
  main5.py     # Day 5 — Self-RAG with reflection scoring and answer regeneration
  main6.py     # Day 6 — LangSmith tracing with full observability on every node
  main7.py     # Day 7 — FastAPI REST API entry point (runs api.py via uvicorn)
  api.py       # Day 7 — FastAPI app: CRAG pipeline as REST endpoints
  tracing.py   # LangSmith tracing utilities (used by main6.py)
  papers/      # Source PDFs indexed into ChromaDB
  .env         # API keys (gitignored)
  .gitignore
  README.md
```

## Running

```bash
# Day 1 — minimal pipeline
python main1.py

# Day 7 — start the REST API (http://localhost:8000)
python main7.py
```

## Day 7 — REST API

Start the server:

```bash
python main7.py
# Swagger UI → http://localhost:8000/docs
# ReDoc      → http://localhost:8000/redoc
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ask` | Run full CRAG pipeline |
| `GET` | `/history/{session_id}` | Retrieve session exchange history |
| `DELETE` | `/history/{session_id}` | Clear session history |
| `GET` | `/health` | Liveness check |
| `POST` | `/upload` | Upload a PDF and add it to ChromaDB |

### Example curl commands

```bash
# Health check
curl http://localhost:8000/health

# Ask a question (auto-generates session_id if omitted)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is corrective RAG?", "session_id": "test123"}'

# View session history
curl http://localhost:8000/history/test123

# Clear session history
curl -X DELETE http://localhost:8000/history/test123

# Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@papers/corrective_rag.pdf"
```

### POST /ask — Response schema

```json
{
  "answer":       "string",
  "sources":      ["filename.pdf:page", "..."],
  "tool_used":    "retrieval | web_search | both",
  "chunk_grades": {"relevant": 3, "irrelevant": 1, "ambiguous": 1},
  "session_id":   "string",
  "latency_ms":   1234.5
}
```

## Author

**Jamson Batista**
GitHub: [@bawilla](https://github.com/bawilla)
