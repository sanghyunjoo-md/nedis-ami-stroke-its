#!/usr/bin/env python3
"""Run the parent analysis followed by the analysis extension."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PIPELINES = [
    ROOT / "analysis" / "run_all.py",
    ROOT / "analysis_extension" / "run_all.py",
]


def main() -> None:
    for pipeline in PIPELINES:
        print(f"Running {pipeline.relative_to(ROOT)}", flush=True)
        subprocess.run([sys.executable, str(pipeline)], check=True)


if __name__ == "__main__":
    main()
