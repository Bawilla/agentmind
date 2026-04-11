"""
AgentMind smoke tests.

Uses FastAPI TestClient with all heavy dependencies mocked so no model
downloads or API calls are made in CI.  Mocked globals:
  - api.CRAG_APP   — .invoke() returns a fake AgentState dict
  - api.VECTORSTORE — replaced with MagicMock (never called directly in tests)
  - api.LLM        — replaced with MagicMock
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fake CRAG pipeline response
# ---------------------------------------------------------------------------
FAKE_STATE = {
    "question":          "What is RAG?",
    "original_question": "What is RAG?",
    "rewrite_count":     0,
    "chunks":            ["RAG stands for Retrieval-Augmented Generation."],
    "chunk_sources":     ["rag_original.pdf"],
    "chunk_pages":       ["1"],
    "grades":            ["relevant"],
    "relevant_context":  "RAG stands for Retrieval-Augmented Generation.",
    "web_context":       "",
    "answer":            "RAG (Retrieval-Augmented Generation) is a technique that "
                         "combines retrieval of relevant documents with LLM generation.",
}


# ---------------------------------------------------------------------------
# Fixture — patches everything heavy, returns TestClient
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_client():
    mock_vs  = MagicMock()
    mock_llm = MagicMock()
    mock_app = MagicMock()
    mock_app.invoke.return_value = FAKE_STATE

    with (
        patch("api.VECTORSTORE", mock_vs),
        patch("api.LLM",         mock_llm),
        patch("api.CRAG_APP",    mock_app),
    ):
        # Import app AFTER patching so lifespan sees the mocks
        import api
        client = TestClient(api.app, raise_server_exceptions=True)
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_health(mock_client):
    """GET /health → 200, body contains status: ok."""
    r = mock_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body
    assert "vectorstore" in body


def test_ask(mock_client):
    """POST /ask → 200, body contains answer field."""
    r = mock_client.post("/ask", json={"question": "What is RAG?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert len(body["answer"]) > 0
    assert "session_id" in body
    assert "tool_used" in body
    assert "chunk_grades" in body
    assert "latency_ms" in body


def test_ask_with_session(mock_client):
    """POST /ask with explicit session_id → response echoes same session_id."""
    sid = str(uuid.uuid4())
    r = mock_client.post("/ask", json={"question": "Explain ZNE", "session_id": sid})
    assert r.status_code == 200
    assert r.json()["session_id"] == sid


def test_history(mock_client):
    """GET /history/{session_id} → 200, body contains exchanges list."""
    sid = str(uuid.uuid4())
    # First create a session via /ask
    mock_client.post("/ask", json={"question": "What is RAG?", "session_id": sid})
    # Then fetch history
    r = mock_client.get(f"/history/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert isinstance(body["exchanges"], list)
    assert len(body["exchanges"]) == 1
    assert "question" in body["exchanges"][0]
    assert "answer" in body["exchanges"][0]


def test_history_empty(mock_client):
    """GET /history/{unknown_id} → 200, empty exchanges list."""
    r = mock_client.get(f"/history/{uuid.uuid4()}")
    assert r.status_code == 200
    assert r.json()["exchanges"] == []
