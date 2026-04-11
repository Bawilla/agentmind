"""
AgentMind — monitoring/dashboard.py

Terminal summary of query activity from monitoring/query_log.jsonl.

Usage:
    python monitoring/dashboard.py
    python -m monitoring.dashboard
"""

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone

_HERE     = os.path.dirname(os.path.abspath(__file__))
QUERY_LOG = os.path.join(_HERE, "query_log.jsonl")

# Common English stop words to exclude from topic extraction
STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "with", "this", "that", "what", "how", "why",
    "when", "where", "which", "who", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "i", "you", "we", "they", "he",
    "she", "my", "your", "their", "me", "us", "them", "about", "from",
}


def _load_log() -> list:
    if not os.path.exists(QUERY_LOG):
        return []
    with open(QUERY_LOG, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def main() -> None:
    records = _load_log()

    if not records:
        print("[Dashboard] No query log found. Make some /ask requests first.")
        return

    today = _today_str()
    today_records = [
        r for r in records
        if r.get("timestamp", "").startswith(today)
    ]

    # ── Latency ─────────────────────────────────────────────────────────────
    latencies   = [r["latency_ms"] for r in records if "latency_ms" in r]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # ── Tool usage ───────────────────────────────────────────────────────────
    tool_counts = Counter(r.get("tool_used", "unknown") for r in records)
    total_with_tool = sum(tool_counts.values())

    # ── Topic extraction ─────────────────────────────────────────────────────
    words: list[str] = []
    for r in records:
        q = r.get("question", "")
        tokens = re.findall(r"\b[a-zA-Z]{4,}\b", q.lower())
        words.extend(t for t in tokens if t not in STOP_WORDS)
    top_topics = Counter(words).most_common(5)

    # ── Print summary ────────────────────────────────────────────────────────
    sep = "─" * 50
    print(f"\n{sep}")
    print("  AgentMind Monitoring Dashboard")
    print(sep)
    print(f"  Total queries (all time) : {len(records)}")
    print(f"  Total queries (today)    : {len(today_records)}")
    print(f"  Average latency          : {avg_latency:,.1f} ms")

    print(f"\n  Tool usage breakdown ({total_with_tool} total):")
    for tool, count in sorted(tool_counts.items()):
        pct = 100 * count / total_with_tool if total_with_tool else 0
        bar = "█" * int(pct / 5)
        print(f"    {tool:15s}  {count:4d}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Top 5 question topics:")
    for i, (word, count) in enumerate(top_topics, 1):
        print(f"    {i}. {word}  ({count} mentions)")

    print(sep + "\n")


if __name__ == "__main__":
    main()
