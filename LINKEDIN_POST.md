# LinkedIn Post — AgentMind Launch

---

I built a production-grade agentic RAG system from scratch in 10 days. Here's what I learned. 🧵

---

Most RAG tutorials stop at "retrieve + answer." That's not a system — it's a prototype.

Real RAG needs to **know when its retrieval failed** and do something about it.

So I built AgentMind: a LangGraph-based agentic RAG pipeline that reasons before retrieving, grades what it finds, and loops back until the answer is good enough.

**10 days. 10 features. Here's the breakdown:**

🗓 **Day 1** — Minimal 3-node LangGraph pipeline. Got comfortable with StateGraph, TypedDict state, and conditional edges.

🗓 **Day 2** — Basic agentic RAG. The agent decides whether to retrieve or answer directly. First taste of conditional routing.

🗓 **Day 3** — Multi-tool agent. Routes between retrieval, DuckDuckGo web search, and a calculator. Tool selection is fully LLM-driven.

🗓 **Day 4** — **Corrective RAG (CRAG)**. The system now grades every retrieved chunk as `relevant`, `irrelevant`, or `ambiguous`. Any irrelevant chunk triggers web search. All-ambiguous triggers a query rewrite and a fresh retrieval.

🗓 **Day 5** — **Self-RAG with reflection scoring**. Every generated answer gets scored on three axes before it's returned:
- Groundedness (is it supported by context?)
- Relevance (does it answer the question?)
- Completeness (is it thorough?)

Any score below 0.7 triggers regeneration (max 2×). Real results from live runs:

| Question | G / R / C | Avg |
|----------|-----------|-----|
| How does Self-RAG decide when to retrieve? | 0.90 / 1.00 / 0.80 | **0.90** |
| RAG retrieval math formulation | 0.60 / 0.80 / 0.40 | **0.60** → regenerated twice |
| LLM agents multi-step reasoning | 0.80 / 1.00 / 0.80 | **0.87** |

🗓 **Day 6** — LangSmith tracing. Every node is `@traceable`. Full token counts, latency per step, and run URLs in the console.

🗓 **Day 7** — FastAPI REST API. CRAG as HTTP endpoints. Session management. PDF upload → auto-indexed into ChromaDB.

🗓 **Day 8** — Streaming. Terminal spinner during retrieval, then tokens stream live as they arrive from the LLM. Also added `POST /ask/stream` as a Server-Sent Events endpoint.

🗓 **Day 9** — Docker. One `docker-compose up --build` starts everything. ChromaDB index survives restarts via volume mounts.

🗓 **Day 10** — Polish. Architecture diagram, professional docs, this post.

---

**Stack:** LangGraph · Groq (llama-3.1-8b-instant) · ChromaDB · HuggingFace embeddings · FastAPI · DuckDuckGo · LangSmith · Docker

**What I'd do differently:** Start with the data layer and evaluation harness on Day 1. I was measuring answer quality with vibes until Day 5 — Self-RAG reflection scores changed that.

The full code (with every day as a standalone script) is on GitHub: [github.com/bawilla/agentmind]

---

If you're building RAG systems and want to go beyond the basic retrieve-and-answer loop, I'd start with CRAG (Day 4). It's the simplest upgrade with the biggest impact on answer quality.

Happy to answer questions in the comments.

---

#RAG #LangGraph #AI #GenerativeAI #MLEngineer #Python #LLM #VectorDatabase #FastAPI #Docker #MachineLearning #DataScience #OpenToWork
