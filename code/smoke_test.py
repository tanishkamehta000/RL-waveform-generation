"""Minimal smoke test to validate the pipeline runs.

Run from repo root:
  python -m code.smoke_test
"""
from pathlib import Path
from code.run_analyze import main as run

if __name__ == "__main__":
    # Use small limits for faster check
    import sys
    sys.argv = [
        sys.argv[0],
        "--background", "./benign/background/location1",
        "--floor", "./benign/floor",
        "--out", "./outputs_smoke",
        "--max-bg-files", "10",
        "--max-floor-files", "5",
    ]
    run()
