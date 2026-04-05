"""
AgentMind — tracing.py

LangSmith observability utilities.

Usage:
    from tracing import init_tracing, trace_step, get_run_url

    init_tracing()          # call once at startup

    @trace_step("my_node")
    def my_node(state):
        ...

    print(get_run_url())    # prints clickable LangSmith URL after a run
"""

import os
import time
import functools
from typing import Callable, Any, Optional

from dotenv import load_dotenv

# Load .env before anything else so env vars are present when langsmith imports
load_dotenv()

import langsmith
from langsmith import Client

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_client: Optional[Client] = None
_project: str = "agentmind"
_current_run_id: Optional[str] = None


# ---------------------------------------------------------------------------
# init_tracing
# ---------------------------------------------------------------------------
def init_tracing() -> None:
    """Initialise the LangSmith client and confirm connectivity."""
    global _client, _project

    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    _project = os.environ.get("LANGCHAIN_PROJECT", "agentmind")

    if not api_key:
        print("[Tracing] WARNING: LANGCHAIN_API_KEY not set — tracing disabled.")
        return

    _client = Client(api_key=api_key)

    # Verify connection
    try:
        projects = list(_client.list_projects())
        names = [p.name for p in projects]
        if _project not in names:
            _client.create_project(_project)
            print(f"[Tracing] Created new LangSmith project: {_project!r}")
        else:
            print(f"[Tracing] Connected to LangSmith project: {_project!r}")
    except Exception as exc:
        print(f"[Tracing] Could not verify LangSmith connection: {exc}")

    print(f"[Tracing] Dashboard → https://smith.langchain.com/o/~/projects/p/{_project}")


# ---------------------------------------------------------------------------
# trace_step decorator
# ---------------------------------------------------------------------------
def trace_step(name: str, metadata: Optional[dict] = None) -> Callable:
    """
    Decorator that wraps a node function and logs it to LangSmith as a
    child run of the current trace.

    Logs: inputs, outputs, latency, any metadata passed in.
    Works transparently when tracing is disabled (LANGCHAIN_TRACING_V2 not set).
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            tracing_on = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"

            if not tracing_on or _client is None:
                return fn(*args, **kwargs)

            # Capture a serialisable snapshot of the first positional arg (state dict)
            inputs: dict = {}
            if args:
                try:
                    state = args[0]
                    inputs = {
                        "question":   state.get("question", ""),
                        "regen_count": state.get("regen_count", 0),
                        "rewrite_count": state.get("rewrite_count", 0),
                        "retrieval_retries": state.get("retrieval_retries", 0),
                    }
                except Exception:
                    inputs = {}

            extra = metadata or {}

            run_id = None
            t0 = time.perf_counter()
            try:
                run = _client.create_run(
                    name=name,
                    run_type="chain",
                    inputs={**inputs, **extra},
                    project_name=_project,
                )
                run_id = run.id if hasattr(run, "id") else None
            except Exception:
                run_id = None

            try:
                result = fn(*args, **kwargs)
                latency_ms = int((time.perf_counter() - t0) * 1000)

                # Build a serialisable output snapshot
                outputs: dict = {"latency_ms": latency_ms}
                if isinstance(result, dict):
                    for key in ("answer", "grades", "retrieval_score", "groundedness",
                                "relevance", "completeness", "tool_selected",
                                "web_context", "retrieval_reason"):
                        if key in result:
                            outputs[key] = result[key]

                if run_id is not None:
                    try:
                        _client.update_run(run_id, outputs=outputs, end_time=None)
                    except Exception:
                        pass

                return result

            except Exception as exc:
                if run_id is not None:
                    try:
                        _client.update_run(run_id, error=str(exc))
                    except Exception:
                        pass
                raise

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# get_run_url
# ---------------------------------------------------------------------------
def get_run_url(question: Optional[str] = None) -> str:
    """
    Returns (and prints) a clickable LangSmith URL for the active project.
    LangSmith's SDK uses env vars to group all calls under the project
    automatically — this URL opens that project's run list.
    """
    project = os.environ.get("LANGCHAIN_PROJECT", "agentmind")
    # URL encodes spaces as %20
    encoded = project.replace(" ", "%20")
    url = f"https://smith.langchain.com/o/~/projects/p/{encoded}"
    if question:
        print(f"[Tracing] Run URL for {question!r}:")
    print(f"  → {url}")
    return url
