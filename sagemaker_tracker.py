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

import threading
import uuid as _uuid

# Constants imported from api.py — referenced by name to avoid circular imports
# at module level. They are resolved at call time.
EXPERIMENT_NAME = "agentmind-production"

_lock         = threading.Lock()   # guards _initialised / _sm_available writes
_initialised  = False
_sm_available = False
_sm_legacy    = False              # True → smexperiments SDK (sagemaker < 2.123)


def init() -> None:
    """Check SageMaker SDK availability and AWS credentials."""
    global _initialised, _sm_available, _sm_legacy

    with _lock:
        if _initialised:            # second thread arrives after first finishes
            return

        try:
            import boto3

            # sagemaker.experiments.run.Run arrived in sagemaker 2.123.
            # Fall back to the standalone smexperiments package for older installs.
            try:
                from sagemaker.experiments.run import Run  # noqa: F401
            except ImportError:
                import smexperiments  # noqa: F401  — raises ImportError if absent
                _sm_legacy = True

            # Isolate credential check: missing ~/.aws or absent env vars must
            # only disable tracking, never block API startup.
            try:
                boto3.client("sts").get_caller_identity()
            except Exception as cred_exc:
                print(f"[SageMaker] AWS credentials not available — tracking disabled. ({cred_exc})")
                return              # _sm_available stays False

            _sm_available = True
            mode = "smexperiments (legacy)" if _sm_legacy else "sagemaker>=2.123"
            print(f"[SageMaker] Experiment tracking ready — experiment='{EXPERIMENT_NAME}' ({mode})")

        except ImportError:
            print("[SageMaker] sagemaker SDK not installed — tracking disabled.")
        except Exception as exc:
            print(f"[SageMaker] Unexpected error during init — tracking disabled. ({exc})")
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
    if not _sm_available:      # confirmed correct — silent return, no raise
        return

    try:
        total_chunks    = max(1, sum(chunk_grades.values()))
        relevant        = chunk_grades.get("relevant", 0)
        relevancy_score = relevant / total_chunks
        run_name        = f"ask-{str(_uuid.uuid4())[:8]}"

        if _sm_legacy:
            # smexperiments SDK (sagemaker < 2.123): no Run context manager;
            # parameters are stored as trial tags (best available in legacy API).
            from smexperiments.trial import Trial
            trial = Trial.create(experiment_name=EXPERIMENT_NAME, trial_name=run_name)
            trial.add_tag({"Key": "model_name",      "Value": str(model_name)})
            trial.add_tag({"Key": "retrieval_top_k", "Value": str(retrieval_top_k)})
            trial.add_tag({"Key": "tool_used",       "Value": str(tool_used)})
        else:
            # sagemaker >= 2.123: full Run API with parameters and metrics
            from sagemaker.experiments.run import Run
            with Run(experiment_name=EXPERIMENT_NAME, run_name=run_name) as run:
                run.log_parameter("model_name",      model_name)
                run.log_parameter("retrieval_top_k", retrieval_top_k)
                run.log_parameter("tool_used",       tool_used)
                run.log_metric(name="latency_ms",       value=latency_ms,       step=0)
                run.log_metric(name="chunks_retrieved", value=total_chunks,     step=0)
                run.log_metric(name="answer_relevancy", value=relevancy_score,  step=0)

        print(f"[SageMaker] Logged trial '{run_name}' to experiment '{EXPERIMENT_NAME}'")

    except Exception as exc:
        # Never crash the API over tracking errors
        print(f"[SageMaker] Warning: failed to log trial — {exc}")
