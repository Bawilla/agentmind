"""
AgentMind — monitoring/drift_monitor.py

Records every /ask request to a rolling JSONL log and runs an Evidently
data-drift check every TRIGGER_EVERY queries.

Usage (called from api.py):
    from monitoring import drift_monitor
    drift_monitor.log_query(question, answer, latency_ms, tool_used)

Manual drift check:
    python -m monitoring.drift_monitor
"""

import json
import os
from datetime import datetime, timezone

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
QUERY_LOG   = os.path.join(_HERE, "query_log.jsonl")
REPORT_DIR  = os.path.join(_HERE, "reports")

# ── Thresholds ─────────────────────────────────────────────────────────────
DRIFT_THRESHOLD = 0.3    # alert if any feature drift score exceeds this
TRIGGER_EVERY   = 50     # run drift check every N queries
REFERENCE_SIZE  = 100    # rows used as reference window
CURRENT_SIZE    = 100    # rows used as current window


def log_query(
    question:   str,
    answer:     str,
    latency_ms: float,
    tool_used:  str,
) -> None:
    """
    Append one query record to query_log.jsonl.
    Triggers a drift check when the log hits a TRIGGER_EVERY boundary.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    record = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "question_length": len(question),
        "answer_length":   len(answer),
        "latency_ms":      latency_ms,
        "tool_used":       tool_used,
        "question":        question,
    }

    with open(QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Count lines to decide whether to trigger a drift check
    with open(QUERY_LOG, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    if total > 0 and total % TRIGGER_EVERY == 0:
        print(f"[DriftMonitor] {total} queries logged — running drift check …")
        run_drift_check()


def _load_log() -> list:
    """Return all records from query_log.jsonl as a list of dicts."""
    if not os.path.exists(QUERY_LOG):
        return []
    with open(QUERY_LOG, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_drift_check() -> None:
    """
    Run an Evidently DataDriftPreset report on the last 200 rows.

    Reference window : rows [-200 : -100]
    Current window   : rows [-100 : ]

    Saves HTML report to monitoring/reports/drift_report_{timestamp}.html.
    Prints an alert if any numeric feature drift score > DRIFT_THRESHOLD.
    """
    try:
        import pandas as pd
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        print("[DriftMonitor] evidently not installed — skipping drift check.")
        return

    records = _load_log()
    min_rows = REFERENCE_SIZE + CURRENT_SIZE
    if len(records) < min_rows:
        print(f"[DriftMonitor] Not enough data ({len(records)} rows, need {min_rows}) — skipping.")
        return

    features = ["question_length", "answer_length", "latency_ms"]
    df = pd.DataFrame(records)[features]

    reference = df.iloc[-(REFERENCE_SIZE + CURRENT_SIZE) : -CURRENT_SIZE].reset_index(drop=True)
    current   = df.iloc[-CURRENT_SIZE:].reset_index(drop=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(REPORT_DIR, f"drift_report_{ts}.html")
    report.save_html(out_path)
    print(f"[DriftMonitor] Report saved → {out_path}")

    # Extract drift scores and alert if any exceed threshold
    result = report.as_dict()
    try:
        drift_results = result["metrics"][0]["result"]["drift_by_columns"]
        alerted = False
        for col, stats in drift_results.items():
            score = stats.get("drift_score", 0.0)
            status = "DRIFT" if stats.get("drift_detected", False) else "ok"
            print(f"  {col:25s}  score={score:.3f}  [{status}]")
            if score > DRIFT_THRESHOLD:
                alerted = True
        if alerted:
            print(f"\n  [ALERT] Drift score exceeded {DRIFT_THRESHOLD} threshold!")
            print(f"  Open report: {out_path}\n")
        else:
            print(f"  All features within drift threshold ({DRIFT_THRESHOLD}).\n")
    except (KeyError, IndexError, TypeError):
        print("[DriftMonitor] Could not parse drift scores from report.")


# ── Run as script for manual checks ────────────────────────────────────────
if __name__ == "__main__":
    records = _load_log()
    print(f"[DriftMonitor] {len(records)} records in log.")
    run_drift_check()
