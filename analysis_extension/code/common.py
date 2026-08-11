#!/usr/bin/env python3
"""Shared utilities for the prespecified NEDIS analysis extension.

Restricted inputs, governance documents, and their expected checksums are
configured locally and are never stored in the public repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_extension_outputs"
CONFIG_PATH = ROOT / "analysis_extension" / "config" / "extension_config_v1.0.json"


def _configured_path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


COHORT_PATH = _configured_path(
    "NEDIS_EXTENSION_COHORT_PKL",
    ROOT / "analysis_outputs" / "02_cohort" / "cohort_primary_weeks.pkl",
)
ADDENDUM_PATH = _configured_path(
    "NEDIS_EXTENSION_ADDENDUM",
    ROOT / "private" / "Analysis_Extension_Addendum_signed_locked.docx",
)
EXPECTED_COHORT_SHA256 = os.environ.get("NEDIS_EXPECTED_EXTENSION_COHORT_SHA256")
EXPECTED_ADDENDUM_SHA256 = os.environ.get("NEDIS_EXPECTED_EXTENSION_ADDENDUM_SHA256")
DISEASES = ["AMI", "Ischemic stroke", "Hemorrhagic stroke"]
FIRST_COMPLETE_WEEK = pd.Timestamp("2023-01-02")
TRANSITION_WEEK = pd.Timestamp("2024-02-19")
FIRST_POST_WEEK = pd.Timestamp("2024-02-26")
LAST_COMPLETE_WEEK = pd.Timestamp("2024-12-23")
PERIOD_WEEKS = 52.18
BASE_TERMS = ["intercept", "time", "post", "post_time", "sin1", "cos1"]

sys.path.insert(0, str(ROOT / "analysis"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cleaned_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(cleaned_string(series), errors="coerce")


def parse_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    dates = cleaned_string(date_series)
    times = cleaned_string(time_series)
    valid = dates.str.fullmatch(r"\d{8}", na=False) & times.str.fullmatch(r"\d{4}", na=False)
    text = (dates.where(valid) + times.where(valid)).where(valid)
    return pd.to_datetime(text, format="%Y%m%d%H%M", errors="coerce")


def week_skeleton() -> pd.DataFrame:
    weeks = pd.date_range(FIRST_COMPLETE_WEEK, LAST_COMPLETE_WEEK, freq="7D")
    weeks = weeks[weeks != TRANSITION_WEEK]
    return pd.MultiIndex.from_product([DISEASES, weeks], names=["disease", "week_start"]).to_frame(index=False)


def add_time_terms(frame: pd.DataFrame, first_post: pd.Timestamp = FIRST_POST_WEEK) -> pd.DataFrame:
    out = frame.copy()
    out["week_start"] = pd.to_datetime(out["week_start"])
    out["calendar_time"] = ((out["week_start"] - FIRST_COMPLETE_WEEK).dt.days // 7).astype(int)
    out["post"] = out["week_start"].ge(first_post).astype(int)
    first_post_index = int((first_post - FIRST_COMPLETE_WEEK).days // 7)
    out["post_time"] = np.where(out["post"].eq(1), out["calendar_time"] - first_post_index, 0).astype(int)
    out["sin1"] = np.sin(2 * np.pi * out["calendar_time"] / PERIOD_WEEKS)
    out["cos1"] = np.cos(2 * np.pi * out["calendar_time"] / PERIOD_WEEKS)
    out["sin2"] = np.sin(4 * np.pi * out["calendar_time"] / PERIOD_WEEKS)
    out["cos2"] = np.cos(4 * np.pi * out["calendar_time"] / PERIOD_WEEKS)
    return out


def design(frame: pd.DataFrame, second_harmonic: bool = False) -> tuple[np.ndarray, list[str]]:
    terms = BASE_TERMS + (["sin2", "cos2"] if second_harmonic else [])
    columns = [np.ones(len(frame))]
    source_columns = {"time": "calendar_time"}
    columns.extend(frame[source_columns.get(name, name)].to_numpy(float) for name in terms[1:])
    return np.column_stack(columns), terms


def wald_test(beta: np.ndarray, cov: np.ndarray, indices: list[int]) -> dict[str, float]:
    values = beta[indices]
    vcov = cov[np.ix_(indices, indices)]
    statistic = float(values @ np.linalg.pinv(vcov) @ values)
    df = int(np.linalg.matrix_rank(vcov))
    return {
        "wald_chi2": statistic,
        "df": df,
        "p_raw": float(chi2.sf(statistic, df)) if df > 0 else 1.0,
    }


def period_label(post: pd.Series) -> pd.Series:
    return post.map({0: "Preinterruption", 1: "Postinterruption"}).astype("string")
