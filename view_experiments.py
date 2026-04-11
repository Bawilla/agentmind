"""
Launch the MLflow UI to browse agentmind-rag experiment runs.

Usage:
    python view_experiments.py

Then open: http://localhost:5001
"""

import subprocess
import sys

PORT = 5001

print(f"Starting MLflow UI on http://localhost:{PORT}")
print("Press Ctrl+C to stop.\n")

subprocess.run([
    sys.executable, "-m", "mlflow", "ui",
    "--port", str(PORT),
    "--host", "0.0.0.0",
])
