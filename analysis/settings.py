#!/usr/bin/env python3
"""Local paths and optional integrity checks for the public code release.

Restricted NEDIS data, signed governance files, and their checksums are supplied
locally by an authorized researcher and are never stored in this repository.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _configured_path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


RAW_XLSX = _configured_path(
    "NEDIS_RAW_XLSX",
    ROOT / "data" / "raw" / "NEDIS_custom_extract.xlsx",
)
SAP_DOCX = _configured_path(
    "NEDIS_LOCKED_SAP",
    ROOT / "private" / "SAP_signed_locked.docx",
)

# Set these locally to verify an authorized extract and locked SAP. Do not
# commit restricted-file checksums unless the data provider permits it.
EXPECTED_RAW_SHA256 = os.environ.get("NEDIS_EXPECTED_RAW_SHA256")
EXPECTED_SAP_SHA256 = os.environ.get("NEDIS_EXPECTED_SAP_SHA256")

