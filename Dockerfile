# ── AgentMind — Production Dockerfile (ECS Fargate) ──────────────────────────
#
# Build:   docker build -t agentmind .
# Run:     docker run -p 8000:8000 --env-file .env agentmind
# Compose: docker-compose up
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source — all Python files
COPY *.py ./

# Copy papers folder (PDFs indexed into ChromaDB at startup)
COPY papers/ ./papers/

EXPOSE 8000

# Health check — hits /health every 30s; marks container unhealthy after 3 failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
