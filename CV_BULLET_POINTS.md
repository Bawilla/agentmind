# CV Bullet Points — AgentMind

For ML Engineer / AI Engineer roles (Germany focus).

---

- **Architected** a production-grade agentic RAG system (AgentMind) using LangGraph, implementing Corrective RAG and Self-RAG pipelines with explicit state-machine control, chunk grading, and reflection-based answer scoring — reducing hallucination risk through iterative retrieval and regeneration loops

- **Engineered** a self-evaluating answer quality framework in which each LLM response is scored on groundedness, relevance, and completeness (0–1); scores below 0.7 trigger targeted regeneration (max 2×), achieving average reflection scores of 0.87–0.90 on domain-specific QA benchmarks

- **Built and deployed** a FastAPI REST service exposing CRAG as HTTP and Server-Sent Events (SSE) streaming endpoints with in-memory session management, PDF ingestion into ChromaDB, and LangSmith observability — Docker image hosted on AWS ECR, deployed on AWS ECS Fargate (eu-central-1), serving live RAG queries at scale via REST API (http://35.157.189.76:8000)

- **Integrated** a multi-stage retrieval strategy combining dense vector search (HuggingFace `all-MiniLM-L6-v2`, ChromaDB) with DuckDuckGo web fallback and LLM-driven query rewriting, cutting irrelevant-context responses by routing each request through a graded CRAG decision graph

- **Designed** a 10-day incremental build programme (state machine → agentic RAG → multi-tool agent → CRAG → Self-RAG → tracing → API → streaming → Docker → documentation), demonstrating end-to-end delivery of a research concept to a production-ready, observable, containerised service

---

**Keywords for ATS:** LangGraph, LangChain, RAG, agentic AI, LLM, FastAPI, ChromaDB, Docker, HuggingFace, Groq, LangSmith, Python, vector search, NLP, generative AI, MLOps
