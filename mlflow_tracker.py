"""
AgentMind — mlflow_tracker.py

Wraps every /ask request to log experiment data to MLflow.
Saves to ./mlruns/ directory locally.

Usage:
    import mlflow_tracker
    mlflow_tracker.init()          # call once at startup
    mlflow_tracker.log_ask_run(question, session_id, response)
"""

EXPERIMENT_NAME = "agentmind-rag"
_initialised = False


def init() -> None:
    """Initialise MLflow with the agentmind-rag experiment."""
    global _initialised
    try:
        import mlflow
        mlflow.set_experiment(EXPERIMENT_NAME)
        _initialised = True
        print(f"[MLflow] Experiment '{EXPERIMENT_NAME}' ready — runs saved to ./mlruns/")
    except ImportError:
        print("[MLflow] mlflow not installed — tracking disabled.")
    except Exception as exc:
        print(f"[MLflow] Warning: could not initialise — {exc}")


def log_ask_run(
    question:   str,
    session_id: str,
    answer:     str,
    sources:    list,
    tool_used:  str,
    chunk_grades: dict,
    latency_ms: float,
) -> None:
    """
    Log a single /ask request as one MLflow run.

    Parameters match the fields of AskResponse (api.py:73-79) plus
    the raw question string.

    Logged params:
        session_id      — conversation session identifier
        question_length — character count of the question

    Logged metrics:
        latency_ms          — end-to-end request latency
        chunks_retrieved    — total chunks returned by the retriever (TOP_K)
        chunks_relevant     — chunks graded 'relevant'
        chunks_irrelevant   — chunks graded 'irrelevant'
        chunks_ambiguous    — chunks graded 'ambiguous'
        sources_count       — unique sources cited in the answer
        answer_length       — character count of the generated answer
        tool_used_retrieval — 1 if retrieval was used, else 0
        tool_used_web       — 1 if web search was used, else 0
    """
    if not _initialised:
        return  # silently skip if init() was never called

    try:
        import mlflow
        with mlflow.start_run():
            # Parameters (categorical / identifier fields)
            mlflow.log_params({
                "session_id":      session_id,
                "question_length": len(question),
                "tool_used":       tool_used,
            })

            # Metrics (numeric, comparable across runs)
            mlflow.log_metrics({
                "latency_ms":           latency_ms,
                "chunks_retrieved":     sum(chunk_grades.values()),
                "chunks_relevant":      chunk_grades.get("relevant",   0),
                "chunks_irrelevant":    chunk_grades.get("irrelevant", 0),
                "chunks_ambiguous":     chunk_grades.get("ambiguous",  0),
                "sources_count":        len(sources),
                "answer_length":        len(answer),
                "tool_used_retrieval":  int(tool_used in ("retrieval", "both")),
                "tool_used_web":        int(tool_used in ("web_search", "both")),
            })
    except Exception as exc:
        # Never crash the API over tracking errors
        print(f"[MLflow] Warning: failed to log run — {exc}")
