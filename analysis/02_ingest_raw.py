#!/usr/bin/env python3
"""Read the unmodified NEDIS XLSX once and create a loss-minimizing local cache.

No cohort filters or outcome models are applied in this step.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from settings import RAW_XLSX, ROOT

OUT = ROOT / "analysis_outputs" / "01_raw_cache"
KEY_CODE_COLUMNS = [
    "ptmiemcl",
    "ptmibrtd",
    "ptmisexx",
    "ptmiiukd",
    "ptmidgkd",
    "ptmiinrt",
    "ptmiinmn",
    "ptmikts1",
    "ptmiemrt",
    "ptmidcrt",
    "ptmiintp",
    "ptmidctp",
]


def norm_value_counts(s: pd.Series) -> dict[str, int]:
    x = s.fillna("").astype(str).str.strip()
    x = x.replace("", "<blank>")
    return {str(k): int(v) for k, v in x.value_counts(dropna=False).sort_index().items()}


def main() -> None:
    if not RAW_XLSX.is_file():
        raise FileNotFoundError(
            f"Restricted NEDIS workbook not found: {RAW_XLSX}. "
            "Set NEDIS_RAW_XLSX or follow data/README.md."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    df = pd.read_excel(
        RAW_XLSX,
        sheet_name="EMIHPTMI",
        dtype=str,
        keep_default_na=False,
        engine="openpyxl",
    )
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].fillna("").astype(str).str.strip()

    cache = OUT / "EMIHPTMI_raw_strings.pkl"
    df.to_pickle(cache)

    profile = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "read_started_utc": started.isoformat(),
        "source_rows": int(len(df)),
        "source_columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "exact_duplicate_rows_all_fields": int(df.duplicated(keep="first").sum()),
        "unique_facility_tokens": int(df["ptmiemnm"].replace("", pd.NA).nunique(dropna=True)),
        "key_code_frequencies": {c: norm_value_counts(df[c]) for c in KEY_CODE_COLUMNS},
        "blank_counts": {c: int((df[c] == "").sum()) for c in df.columns},
        "cache_path": str(cache.relative_to(ROOT)),
        "cache_size_bytes": int(cache.stat().st_size),
    }
    (OUT / "raw_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "source_rows": profile["source_rows"],
        "source_columns": profile["source_columns"],
        "exact_duplicate_rows_all_fields": profile["exact_duplicate_rows_all_fields"],
        "unique_facility_tokens": profile["unique_facility_tokens"],
        "cache_size_bytes": profile["cache_size_bytes"],
        "key_code_frequencies": profile["key_code_frequencies"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
