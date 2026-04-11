"""
AgentMind — sagemaker_tracker.py

Logs each /ask request as a SageMaker Experiment trial so all production
runs are visible in AWS SageMaker Studio.

Prerequisites:
  - AWS credentials configured (env vars or ~/.aws/credentials)
  - IAM permissions: sagemaker:CreateExperiment, CreateTrial, CreateTrialComponent,
                     sagemaker:AssociateTrialComponent

Usage:
    import sagemaker_tracker
    sagemaker_tracker.init()                    # call once at startup
    sagemaker_tracker.log_trial(question, response_dict)
"""

import uuid as _uuid

# Constants imported from api.py — referenced by name to avoid circular imports
# at module level. They are resolved at call time.
EXPERIMENT_NAME = "agentmind-production"

_initialised   = False
_sm_available  = False


def init() -> None:
    """Check SageMaker SDK availability and AWS credentials."""
    global _initialised, _sm_available
    try:
        import boto3
        from sagemaker.experiments.run import Run  # noqa: F401

        # Quick credential check — will raise if no credentials are configured
        boto3.client("sts").get_caller_identity()

        _sm_available = True
        print(f"[SageMaker] Experiment tracking ready — experiment='{EXPERIMENT_NAME}'")
    except ImportError:
        print("[SageMaker] sagemaker SDK not installed — tracking disabled.")
    except Exception as exc:
        print(f"[SageMaker] AWS credentials not available — tracking disabled. ({exc})")
    finally:
        _initialised = True


def log_trial(
    question:     str,
    answer:       str,
    tool_used:    str,
    chunk_grades: dict,
    latency_ms:   float,
    model_name:   str = "llama-3.1-8b-instant",
    retrieval_top_k: int = 5,
) -> None:
    """
    Log one /ask request as a SageMaker Experiment Run (trial).

    Logged parameters:
        model_name          — Groq LLM model identifier
        retrieval_top_k     — number of chunks retrieved per query

    Logged metrics (step=0):
        latency_ms          — end-to-end request latency
        chunks_retrieved    — total chunks returned by retriever
        answer_relevancy    — relevant / total chunks (proxy quality score)

    Each run is named ask-{short_uuid} and belongs to EXPERIMENT_NAME.
    """
    if not _initialised:
        init()
    if not _sm_available:
        return

    try:
        from sagemaker.experiments.run import Run

        total_chunks   = max(1, sum(chunk_grades.values()))
        relevant       = chunk_grades.get("relevant", 0)
        relevancy_score = relevant / total_chunks

        run_name = f"ask-{str(_uuid.uuid4())[:8]}"

        with Run(experiment_name=EXPERIMENT_NAME, run_name=run_name) as run:
            # Parameters (static per-request configuration)
            run.log_parameter("model_name",       model_name)
            run.log_parameter("retrieval_top_k",  retrieval_top_k)
            run.log_parameter("tool_used",        tool_used)

            # Metrics (numeric performance indicators)
            run.log_metric(name="latency_ms",        value=latency_ms,      step=0)
            run.log_metric(name="chunks_retrieved",  value=total_chunks,    step=0)
            run.log_metric(name="answer_relevancy",  value=relevancy_score, step=0)

        print(f"[SageMaker] Logged trial '{run_name}' to experiment '{EXPERIMENT_NAME}'")

    except Exception as exc:
        # Never crash the API over tracking errors
        print(f"[SageMaker] Warning: failed to log trial — {exc}")
