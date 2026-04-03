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
  main2.py     # Day 2 — Add ChromaDB retrieval node + HuggingFace embeddings
  main3.py     # Day 3 — Query rewriting node before retrieval
  main4.py     # Day 4 — Self-reflection loop: judge answer quality, retry if poor
  main5.py     # Day 5 — Multi-step reasoning: decompose → retrieve → synthesize
  main6.py     # Day 6 — Streaming output with LangGraph async execution
  main7.py     # Day 7 — Add memory node: persist conversation history
  main8.py     # Day 8 — Tool-use node: web search fallback when docs insufficient
  main9.py     # Day 9 — FastAPI wrapper exposing the agent as a REST endpoint
  main10.py    # Day 10 — Docker + production hardening (retry, timeout, logging)
  .env         # API keys (gitignored)
  .gitignore
  README.md
```

## Running

```bash
# Day 1 — minimal pipeline
python main1.py
```

## Author

**Jamson Batista**
GitHub: [@bawilla](https://github.com/bawilla)
