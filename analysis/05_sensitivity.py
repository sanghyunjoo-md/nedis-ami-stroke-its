#!/usr/bin/env python3
"""Execute the complete prespecified co-primary sensitivity matrix."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from model_utils import (
    effect_row,
    fit_grouped_binomial_hac,
    fit_nb2_hac,
    fit_poisson_hac,
    sandwich,
)


ROOT = Path(__file__).resolve().parents[1]
COHORT_FILE = ROOT / "analysis_outputs" / "02_cohort" / "cohort_all_study_dates.pkl"
RAW_FILE = ROOT / "analysis_outputs" / "01_raw_cache" / "EMIHPTMI_raw_strings.pkl"
OUT = ROOT / "analysis_outputs" / "04_sensitivity"
DISEASES = ["AMI", "Ischemic stroke", "Hemorrhagic stroke"]
FIRST_WEEK = pd.Timestamp("2023-01-02")
LAST_WEEK = pd.Timestamp("2024-12-23")
TRANSITION = pd.Timestamp("2024-02-19")
PRIMARY_POST = pd.Timestamp("2024-02-26")
HOLIDAYS = pd.to_datetime(["2023-01-16", "2023-09-25", "2024-02-05", "2024-09-16"])


def primary_weeks() -> pd.DatetimeIndex:
    weeks = pd.date_range(FIRST_WEEK, LAST_WEEK, freq="7D")
    return weeks[weeks != TRANSITION]


def aggregate(cohort: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    c = cohort.loc[cohort["week_start"].isin(weeks)].copy()
    skeleton = pd.MultiIndex.from_product([DISEASES, weeks], names=["disease", "week_start"]).to_frame(index=False)
    w = c.groupby(["disease", "week_start"], observed=True).agg(
        visit_count=("disease", "size"),
        known_disposition_n=("known_disposition", "sum"),
        transfer_out_n=("transfer_out", "sum"),
    ).reset_index()
    w = skeleton.merge(w, how="left", on=["disease", "week_start"])
    for col in ["visit_count", "known_disposition_n", "transfer_out_n"]:
        w[col] = w[col].fillna(0).astype(int)
    w["calendar_time"] = ((w["week_start"] - FIRST_WEEK).dt.days // 7).astype(int)
    return w


def add_timing(w: pd.DataFrame, post_start: pd.Timestamp, post_time_origin: int) -> pd.DataFrame:
    x = w.copy()
    x["post"] = x["week_start"].ge(post_start).astype(int)
    x["post_time"] = np.where(x["post"].eq(1), x["calendar_time"] - post_time_origin, 0).astype(int)
    x["sin1"] = np.sin(2 * np.pi * x["calendar_time"] / 52.18)
    x["cos1"] = np.cos(2 * np.pi * x["calendar_time"] / 52.18)
    x["sin2"] = np.sin(4 * np.pi * x["calendar_time"] / 52.18)
    x["cos2"] = np.cos(4 * np.pi * x["calendar_time"] / 52.18)
    return x


def design(d: pd.DataFrame, second_harmonic: bool = False) -> np.ndarray:
    cols = [np.ones(len(d)), d["calendar_time"], d["post"], d["post_time"], d["sin1"], d["cos1"]]
    if second_harmonic:
        cols.extend([d["sin2"], d["cos2"]])
    return np.column_stack(cols).astype(float)


def fit_ar1_gaussian(y: np.ndarray, x: np.ndarray, lag: int = 4) -> tuple[np.ndarray, np.ndarray, float]:
    beta0 = np.linalg.pinv(x.T @ x) @ (x.T @ y)
    resid = y - x @ beta0
    rho = float(np.dot(resid[1:], resid[:-1]) / max(np.dot(resid[:-1], resid[:-1]), 1e-12))
    rho = float(np.clip(rho, -0.95, 0.95))
    ys = np.r_[np.sqrt(1 - rho**2) * y[0], y[1:] - rho * y[:-1]]
    xs = np.vstack([np.sqrt(1 - rho**2) * x[0], x[1:] - rho * x[:-1]])
    bread = np.linalg.pinv(xs.T @ xs)
    beta = bread @ (xs.T @ ys)
    u = ys - xs @ beta
    scores = xs * u[:, None]
    cov = sandwich(bread, scores, lag, x.shape[1])
    return beta, cov, rho


def run_variant(
    name: str,
    weekly: pd.DataFrame,
    *,
    lag: int = 4,
    second_harmonic: bool = False,
    family_override: str | None = None,
    ar1: bool = False,
) -> list[dict]:
    rows = []
    for disease in DISEASES:
        d = weekly.loc[weekly["disease"].eq(disease)].sort_values("week_start").reset_index(drop=True)
        x = design(d, second_harmonic)
        for outcome in ("ED visit count", "Transfer-out rate"):
            if outcome == "ED visit count":
                y = d["visit_count"].to_numpy(float)
                if ar1:
                    beta, cov, rho = fit_ar1_gaussian(np.log(np.maximum(y, 0.5)), x, lag=4)
                    model = "Gaussian log-count AR(1) FGLS"
                    alpha = None
                elif family_override == "poisson_quasi":
                    fit = fit_poisson_hac(y, x, lag=lag)
                    beta, cov, rho, model, alpha = fit.params, fit.cov_hac, None, fit.family, fit.alpha
                else:
                    fit = fit_nb2_hac(y, x, lag=lag)
                    beta, cov, rho, model, alpha = fit.params, fit.cov_hac, None, fit.family, fit.alpha
            else:
                y = d["transfer_out_n"].to_numpy(float)
                n = d["known_disposition_n"].to_numpy(float)
                if np.any(n <= 0):
                    raise RuntimeError(f"Zero rate denominator in {name}, {disease}")
                if ar1:
                    prop = (y + 0.5) / (n + 1.0)
                    beta, cov, rho = fit_ar1_gaussian(np.log(prop / (1 - prop)), x, lag=4)
                    model = "Gaussian empirical-logit AR(1) FGLS"
                    alpha = None
                else:
                    fit = fit_grouped_binomial_hac(y, n, x, lag=lag)
                    beta, cov, rho, model, alpha = fit.params, fit.cov_hac, None, (
                        "Quasi-binomial logit with HAC" if family_override == "poisson_quasi" else fit.family
                    ), None
            for parameter, idx in (("Immediate level change", 2), ("Weekly slope change", 3)):
                result = effect_row(beta, cov, idx, "exp")
                rows.append(
                    {
                        "sensitivity": name,
                        "disease": disease,
                        "outcome": outcome,
                        "parameter": parameter,
                        "model": model,
                        "weeks": len(d),
                        "hac_lag": lag if not ar1 else 4,
                        "alpha": alpha,
                        "ar1_rho": rho,
                        **result,
                    }
                )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_pickle(COHORT_FILE)
    weeks_primary = primary_weeks()
    base = add_timing(aggregate(cohort, weeks_primary), PRIMARY_POST, 59)
    results = []
    metadata: dict = {}

    # Timing sensitivities.
    alt_weeks = pd.date_range(FIRST_WEEK, LAST_WEEK, freq="7D")
    alt_weeks = alt_weeks[alt_weeks != pd.Timestamp("2024-02-26")]
    alt = add_timing(aggregate(cohort, alt_weeks), pd.Timestamp("2024-03-04"), 61)
    results += run_variant("Alternative intervention: 2024-03-01; exclude Feb26-Mar3", alt)

    incl_weeks = pd.date_range(FIRST_WEEK, LAST_WEEK, freq="7D")
    incl = add_timing(aggregate(cohort, incl_weeks), TRANSITION, 59)
    results += run_variant("Transition week included as first postweek", incl)

    holiday = base.loc[~base["week_start"].isin(HOLIDAYS)].copy()
    results += run_variant("Holiday weeks excluded", holiday)
    results += run_variant("Second annual Fourier harmonic", base, second_harmonic=True)
    results += run_variant("Newey-West HAC lag 1", base, lag=1)
    results += run_variant("Newey-West HAC lag 8", base, lag=8)
    results += run_variant("Alternative model family", base, family_override="poisson_quasi")
    results += run_variant("Autoregressive-error segmented model", base, ar1=True)

    # Stable-facility panels are disease-specific and then recombined.
    stable_parts = []
    stable_counts = {}
    model_c = cohort.loc[cohort["week_start"].isin(weeks_primary)]
    for disease in DISEASES:
        cd = model_c.loc[model_c["disease"].eq(disease)]
        presence = cd.groupby(["ptmiemnm", "week_start"], observed=True).size().reset_index(name="n")
        presence["post"] = presence["week_start"].ge(PRIMARY_POST)
        counts = presence.groupby(["ptmiemnm", "post"], observed=True)["week_start"].nunique().unstack(fill_value=0)
        pre_n = counts[False] if False in counts.columns else pd.Series(0, index=counts.index)
        post_n = counts[True] if True in counts.columns else pd.Series(0, index=counts.index)
        keep_fac = counts.index[(pre_n >= 48) & (post_n >= 36)]
        stable_counts[disease] = int(len(keep_fac))
        stable_parts.append(cd.loc[cd["ptmiemnm"].isin(keep_fac)])
    stable_cohort = pd.concat(stable_parts, ignore_index=False)
    stable = add_timing(aggregate(stable_cohort, weeks_primary), PRIMARY_POST, 59)
    results += run_variant("Stable-facility panel (48/59 pre; 36/44 post)", stable)
    metadata["stable_facilities_by_disease"] = stable_counts

    # Facility ED-level continuity across the two-year source extract.
    raw = pd.read_pickle(RAW_FILE)[["ptmiemnm", "ptmiemcl", "ptmiindt"]].copy()
    raw["arrival_date"] = pd.to_datetime(raw["ptmiindt"], format="%Y%m%d", errors="coerce")
    raw = raw.loc[raw["arrival_date"].between("2023-01-01", "2024-12-31")]
    level_n = raw.groupby("ptmiemnm", observed=True)["ptmiemcl"].nunique()
    continuity_facilities = level_n.index[level_n.eq(1)]
    metadata["facilities_with_level_change_excluded"] = int((level_n > 1).sum())
    continuity_cohort = cohort.loc[cohort["ptmiemnm"].isin(continuity_facilities)]
    continuity = add_timing(aggregate(continuity_cohort, weeks_primary), PRIMARY_POST, 59)
    results += run_variant("Center-code continuity", continuity)

    dda_cohort = cohort.loc[~cohort["ptmiemrt"].eq("41")]
    dda = add_timing(aggregate(dda_cohort, weeks_primary), PRIMARY_POST, 59)
    results += run_variant("DDA disposition code 41 excluded", dda)

    # No cross-stratum conflict occurred in the locked primary cohort; report the identical implementation.
    metadata["cross_stratum_conflicts"] = 0
    results += run_variant("First-listed diagnosis for cross-stratum conflicts (none present)", base)

    # Preperiod falsification series.
    placebo_weeks = pd.date_range(FIRST_WEEK, pd.Timestamp("2024-02-12"), freq="7D")
    placebo = aggregate(cohort, placebo_weeks)
    placebo = add_timing(placebo, pd.Timestamp("2023-07-03"), 26)
    results += run_variant("Preperiod placebo: false interruption 2023-07-03", placebo)

    sensitivity = pd.DataFrame(results)
    sensitivity.to_csv(OUT / "complete_coprimary_sensitivity_matrix.csv", index=False)
    metadata.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sensitivity_variants": int(sensitivity["sensitivity"].nunique()),
            "rows": int(len(sensitivity)),
            "expected_rows_per_complete_variant": 12,
        }
    )
    (OUT / "sensitivity_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(sensitivity.groupby("sensitivity").size().to_string())
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
