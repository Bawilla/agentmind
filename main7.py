"""
AgentMind — Day 7: FastAPI REST API

Starts the AgentMind API server using uvicorn on port 8000.
All CRAG logic lives in api.py — this file is the entry point.

Endpoints:
  POST   /ask                    — run full CRAG pipeline
  GET    /history/{session_id}   — retrieve session exchange history
  DELETE /history/{session_id}   — clear session history
  GET    /health                 — liveness check
  POST   /upload                 — add a PDF to ChromaDB at runtime
  GET    /docs                   — Swagger UI (auto-generated)
  GET    /redoc                  — ReDoc UI (auto-generated)
"""

import uvicorn

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║            AgentMind API — Day 7: FastAPI Wrapper            ║
╠══════════════════════════════════════════════════════════════╣
║  Base URL : http://localhost:8000                            ║
║                                                              ║
║  Endpoints:                                                  ║
║    POST   /ask                  — run CRAG pipeline          ║
║    GET    /history/{session_id} — session exchange history   ║
║    DELETE /history/{session_id} — clear session history      ║
║    GET    /health               — liveness check             ║
║    POST   /upload               — add PDF to ChromaDB        ║
║                                                              ║
║  Interactive docs:                                           ║
║    Swagger UI  →  http://localhost:8000/docs                 ║
║    ReDoc       →  http://localhost:8000/redoc                ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
