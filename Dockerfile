# ── AgentMind — Day 9: Docker containerization ──────────────────────────────
#
# Build:   docker build -t agentmind .
# Run:     docker run -p 8000:8000 --env-file .env agentmind
# Compose: docker-compose up
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY api.py tracing.py main7.py ./

# papers/ and chroma_db_main4/ are mounted as volumes at runtime — they are
# NOT baked into the image so the ChromaDB index persists across container
# restarts and PDFs can be added without rebuilding.

EXPOSE 8000

# Health check — hits /health every 30s; marks container unhealthy after 3 failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
