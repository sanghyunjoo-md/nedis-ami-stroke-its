#!/usr/bin/env python3
"""Run the complete secondary/exploratory analysis extension."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent / "code"
SCRIPTS = [
    "01_derive_extension.py",
    "02_fit_extension.py",
    "03_sensitivities.py",
    "04_case_mix_adjusted.py",
    "05_validate_and_figures.py",
    "06_extension_figures.py",
    "07_prepare_workbook_data.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"Running extension {script}", flush=True)
        subprocess.run([sys.executable, str(CODE_DIR / script)], check=True)


if __name__ == "__main__":
    main()
