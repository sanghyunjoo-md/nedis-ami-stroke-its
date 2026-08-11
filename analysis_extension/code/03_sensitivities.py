#!/usr/bin/env python3
"""Run every sensitivity analysis fixed in Analysis Extension Addendum v1.0."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import DISEASES, OUT, add_time_terms, design, week_skeleton, write_json
from model_utils import effect_row


MODULE_PATH = Path(__file__).with_name("02_fit_extension.py")
MODULE_SPEC = importlib.util.spec_from_file_location("extension_fit_module", MODULE_PATH)
FIT_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(FIT_MODULE)
OUTCOME_SPECS = FIT_MODULE.OUTCOME_SPECS
fit_binomial_stable = FIT_MODULE.fit_binomial_stable

WEEKLY_PATH = OUT / "01_derived" / "weekly_extension_inputs.csv"
PATIENT_PATH = OUT / "01_derived" / "extension_patient_derived.pkl"
SENS_OUT = OUT / "03_sensitivity"


def fit_one(frame: pd.DataFrame, key: str, lag: int = 4, second_harmonic: bool = False):
    spec = OUTCOME_SPECS[key]
    if spec["type"] == "continuous":
        d = frame.loc[frame[spec["median"]].gt(0) & frame[spec["median"]].notna()].reset_index(drop=True)
        x, terms = design(d, second_harmonic=second_harmonic)
        from model_utils import fit_ols_hac
        fit = fit_ols_hac(np.log(d[spec["median"]].to_numpy(float)), x, lag=lag)
    else:
        d = frame.loc[frame[f"{key}_valid_n"].gt(0)].reset_index(drop=True)
        x, terms = design(d, second_harmonic=second_harmonic)
        fit = fit_binomial_stable(d[f"{key}_event_n"].to_numpy(float), d[f"{key}_valid_n"].to_numpy(float), x, lag=lag)
    return d, terms, fit


def append_variant(rows: list[dict], weekly: pd.DataFrame, variant: str, outcomes: list[str], diseases: list[str] = DISEASES, lag: int = 4, second_harmonic: bool = False) -> list[dict]:
    diagnostics: list[dict] = []
    for key in outcomes:
        for disease in diseases:
            base = weekly.loc[weekly["disease"].eq(disease)].sort_values("week_start").reset_index(drop=True)
            d, terms, fit = fit_one(base, key, lag=lag, second_harmonic=second_harmonic)
            for parameter, term in [("Immediate level change", "post"), ("Weekly slope change", "post_time")]:
                value = effect_row(fit.params, fit.cov_hac, terms.index(term), "exp")
                rows.append({"sensitivity": variant, "outcome_key": key, "outcome": OUTCOME_SPECS[key]["label"], "disease": disease, "parameter": parameter, "effect_scale": OUTCOME_SPECS[key]["scale"], **value})
            diagnostics.append({"sensitivity": variant, "outcome_key": key, "disease": disease, "converged": bool(fit.converged), "n_weeks": int(len(d)), "message": fit.message})
    return diagnostics


def onset_weekly(patient: pd.DataFrame, upper_minutes: int) -> pd.DataFrame:
    valid = patient["onset_minutes_raw"].between(0, upper_minutes, inclusive="both")
    temp = patient.assign(onset_variant=patient["onset_minutes_raw"].where(valid))
    grouped = temp.groupby(["disease", "week_start"], observed=True)["onset_variant"].agg([("onset_valid_n", "count"), ("onset_median_minutes", "median")]).reset_index()
    weekly = week_skeleton().merge(grouped, on=["disease", "week_start"], how="left")
    weekly["onset_valid_n"] = weekly["onset_valid_n"].fillna(0).astype(int)
    return add_time_terms(weekly)


def binary_weekly(patient: pd.DataFrame, key: str, valid: pd.Series, event: pd.Series) -> pd.DataFrame:
    temp = patient[["disease", "week_start"]].copy()
    temp[f"{key}_valid_n"] = valid.fillna(False).astype(int)
    temp[f"{key}_event_n"] = (event.fillna(False) & valid.fillna(False)).astype(int)
    grouped = temp.groupby(["disease", "week_start"], observed=True)[[f"{key}_valid_n", f"{key}_event_n"]].sum().reset_index()
    weekly = week_skeleton().merge(grouped, on=["disease", "week_start"], how="left")
    weekly[[f"{key}_valid_n", f"{key}_event_n"]] = weekly[[f"{key}_valid_n", f"{key}_event_n"]].fillna(0).astype(int)
    return add_time_terms(weekly)


def main() -> None:
    SENS_OUT.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_csv(WEEKLY_PATH, parse_dates=["week_start"])
    patient = pd.read_pickle(PATIENT_PATH)
    rows: list[dict] = []
    diagnostics: list[dict] = []
    all_outcomes = list(OUTCOME_SPECS)

    alternative = weekly.loc[~weekly["week_start"].eq(pd.Timestamp("2024-02-26"))].copy()
    alternative = add_time_terms(alternative, first_post=pd.Timestamp("2024-03-04"))
    diagnostics += append_variant(rows, alternative, "Alternative interruption: first post week 2024-03-04", all_outcomes)
    diagnostics += append_variant(rows, weekly, "Second annual Fourier harmonic", all_outcomes, second_harmonic=True)
    diagnostics += append_variant(rows, weekly, "Newey-West HAC lag 1", all_outcomes, lag=1)
    diagnostics += append_variant(rows, weekly, "Newey-West HAC lag 8", all_outcomes, lag=8)

    for upper, label in [(7 * 24 * 60, "Onset interval upper bound 7 days"), (24 * 60, "Onset interval upper bound 24 hours")]:
        diagnostics += append_variant(rows, onset_weekly(patient, upper), label, ["onset_to_arrival"])

    non_doa = ~patient["doa"]
    for key in ["shock_range_sbp", "hypoxemia"]:
        variant_weekly = binary_weekly(patient, key, patient[f"{key}_valid"] & non_doa, patient[f"{key}_event"])
        diagnostics += append_variant(rows, variant_weekly, "Exclude ED disposition code 41 (DOA)", [key])

    strict_event = pd.Series(False, index=patient.index)
    strict_event.loc[patient["disease"].eq("AMI")] = patient.loc[patient["disease"].eq("AMI"), "department_code"].eq("AA")
    strict_event.loc[patient["disease"].eq("Ischemic stroke")] = patient.loc[patient["disease"].eq("Ischemic stroke"), "department_code"].eq("FA")
    strict_event.loc[patient["disease"].eq("Hemorrhagic stroke")] = patient.loc[patient["disease"].eq("Hemorrhagic stroke"), "department_code"].eq("BB")
    strict_weekly = binary_weekly(patient, "concordant_department", patient["concordant_department_valid"], strict_event)
    diagnostics += append_variant(rows, strict_weekly, "Strict stroke-concordant department", ["concordant_department"], diseases=["Ischemic stroke", "Hemorrhagic stroke"])

    results = pd.DataFrame(rows)
    results.to_csv(SENS_OUT / "extension_sensitivity_estimates.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(SENS_OUT / "extension_sensitivity_diagnostics.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimate_rows": int(len(results)),
        "model_runs": int(len(diagnostics)),
        "sensitivity_variants": sorted(results["sensitivity"].unique().tolist()),
        "all_models_converged": bool(pd.DataFrame(diagnostics)["converged"].all()),
    }
    write_json(SENS_OUT / "sensitivity_run_summary.json", summary)
    print(results.groupby("sensitivity").size().to_string())
    print(summary)


if __name__ == "__main__":
    main()
