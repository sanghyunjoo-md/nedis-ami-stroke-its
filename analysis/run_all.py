#!/usr/bin/env python3
"""Run the complete analysis in the order prespecified by the locked SAP."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "01_inventory.py",
    "02_ingest_raw.py",
    "03_build_cohort.py",
    "04_primary_its.py",
    "05_sensitivity.py",
    "06_secondary_exploratory.py",
    "07_adjusted_and_stratified.py",
    "08_tables.py",
    "09_figures.py",
    "10_validate.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(ANALYSIS_DIR / script)], check=True)


if __name__ == "__main__":
    main()

