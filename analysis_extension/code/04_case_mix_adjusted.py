#!/usr/bin/env python3
"""Supportive encounter-level case-mix-adjusted care-pathway models."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit

from common import COHORT_PATH, DISEASES, OUT, PERIOD_WEEKS, write_json
from model_utils import linear_combination


PATIENT_PATH = OUT / "01_derived" / "extension_patient_derived.pkl"
ADJ_OUT = OUT / "04_case_mix_adjusted"
OUTCOMES = {
    "ktas_escalation": "KTAS escalation",
    "specialist_involvement": "Consulting-specialist involvement",
    "concordant_department": "Condition-concordant department",
    "high_acuity_area": "High-acuity final treatment area",
}


def dummy_block(series: pd.Series, prefix: str) -> tuple[sparse.csr_matrix, list[str]]:
    text = series.astype("string")
    levels = sorted(str(value) for value in text.dropna().unique())
    if len(levels) <= 1:
        return sparse.csr_matrix((len(series), 0)), []
    columns = [(text == level).fillna(False).astype(float).to_numpy() for level in levels[1:]]
    return sparse.csr_matrix(np.column_stack(columns)), [f"{prefix}[{level}]" for level in levels[1:]]


def build_design(frame: pd.DataFrame, include_ed_level: bool) -> tuple[sparse.csr_matrix, list[str]]:
    base = np.column_stack(
        [
            np.ones(len(frame)),
            frame["calendar_time"].to_numpy(float) / PERIOD_WEEKS,
            frame["post"].to_numpy(float),
            frame["post_time"].to_numpy(float) / PERIOD_WEEKS,
            np.sin(2 * np.pi * frame["calendar_time"].to_numpy(float) / PERIOD_WEEKS),
            np.cos(2 * np.pi * frame["calendar_time"].to_numpy(float) / PERIOD_WEEKS),
            frame["sex"].eq("F").to_numpy(float),
            frame["arrival_119"].to_numpy(float),
            frame["transfer_in"].to_numpy(float),
        ]
    )
    names = ["intercept", "time_per_year", "post", "post_time_per_year", "sin1", "cos1", "female", "arrival_119", "transfer_in"]
    blocks = [sparse.csr_matrix(base)]
    for series, prefix in [(frame["age_group"], "age_group"), (frame["initial_ktas"].astype("Int64").astype("string"), "initial_KTAS"), (frame["avpu"], "AVPU")]:
        block, block_names = dummy_block(series, prefix)
        blocks.append(block)
        names.extend(block_names)
    if include_ed_level:
        # ED level is time invariant within facility and therefore is absorbed by
        # facility fixed effects. It is not added as a redundant design column.
        pass
    facility_block, facility_names = dummy_block(frame["facility"], "facility")
    blocks.append(facility_block)
    names.extend(facility_names)
    return sparse.hstack(blocks, format="csr"), names


def group_meat(x: sparse.csr_matrix, residual: np.ndarray, groups: pd.Series) -> tuple[np.ndarray, int]:
    codes, levels = pd.factorize(groups.astype("string"), sort=True)
    group_count = len(levels)
    aggregator = sparse.csr_matrix((residual, (codes, np.arange(len(residual)))), shape=(group_count, len(residual)))
    scores = (aggregator @ x).toarray()
    return scores.T @ scores, group_count


def fit_logistic_two_way(y: np.ndarray, x: sparse.csr_matrix, facility: pd.Series, week: pd.Series) -> dict:
    p = x.shape[1]
    beta0 = np.zeros(p)
    mean = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    beta0[0] = np.log(mean / (1 - mean))

    def objective(beta: np.ndarray) -> float:
        eta = np.clip(np.asarray(x @ beta).ravel(), -30, 30)
        return float(np.sum(np.logaddexp(0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        probability = expit(np.clip(np.asarray(x @ beta).ravel(), -30, 30))
        return np.asarray(x.T @ (probability - y)).ravel()

    opt = minimize(objective, beta0, jac=gradient, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-6, "maxls": 50})
    beta = opt.x
    probability = expit(np.clip(np.asarray(x @ beta).ravel(), -30, 30))
    weights = probability * (1 - probability)
    info = (x.T @ x.multiply(weights[:, None])).toarray()
    rank = int(np.linalg.matrix_rank(info))
    condition = float(np.linalg.cond(info))
    bread = np.linalg.pinv(info)
    score_residual = y - probability
    meat_facility, n_facilities = group_meat(x, score_residual, facility)
    meat_week, n_weeks = group_meat(x, score_residual, week)
    intersections = facility.astype("string") + "|" + pd.to_datetime(week).astype("string")
    meat_intersection, n_intersections = group_meat(x, score_residual, intersections)
    n = len(y)

    def correction(groups: int) -> float:
        return (groups / max(groups - 1, 1)) * ((n - 1) / max(n - p, 1))

    meat = correction(n_facilities) * meat_facility + correction(n_weeks) * meat_week - correction(n_intersections) * meat_intersection
    covariance = bread @ meat @ bread
    covariance = (covariance + covariance.T) / 2
    gradient_inf = float(np.linalg.norm(gradient(beta), ord=np.inf))
    separation = bool(np.max(np.abs(beta)) > 25 or np.mean(weights < 1e-10) > 0.01)
    estimable = bool((opt.success or gradient_inf < 1e-4) and rank == p and np.isfinite(condition) and condition < 1e12 and not separation)
    reason_parts = []
    if not (opt.success or gradient_inf < 1e-4):
        reason_parts.append(f"nonconvergence: {opt.message}; gradient_inf={gradient_inf:.3g}")
    if rank < p:
        reason_parts.append(f"singular information matrix rank {rank}/{p}")
    if not np.isfinite(condition) or condition >= 1e12:
        reason_parts.append(f"ill-conditioned information matrix condition={condition:.3g}")
    if separation:
        reason_parts.append("complete or quasi-complete separation diagnostic")
    return {
        "beta": beta,
        "covariance": covariance,
        "estimable": estimable,
        "reason": "; ".join(reason_parts) if reason_parts else "estimable",
        "optimizer_success": bool(opt.success),
        "optimizer_message": str(opt.message),
        "gradient_inf": gradient_inf,
        "information_rank": rank,
        "design_columns": p,
        "information_condition_number": condition,
        "n_facilities": n_facilities,
        "n_weeks": n_weeks,
        "n_facility_week_cells": n_intersections,
        "max_abs_coefficient": float(np.max(np.abs(beta))),
        "near_boundary_probability_pct": float(100 * np.mean(weights < 1e-10)),
    }


def main() -> None:
    ADJ_OUT.mkdir(parents=True, exist_ok=True)
    patient = pd.read_pickle(PATIENT_PATH)
    source = pd.read_pickle(COHORT_PATH)
    patient["arrival_method_code"] = source.loc[patient.index, "ptmiinmn"].astype("string").str.strip()
    patient["arrival_route_code"] = source.loc[patient.index, "ptmiinrt"].astype("string").str.strip()
    patient["age_numeric"] = pd.to_numeric(patient["age_group"], errors="coerce")
    common_valid = (
        patient["age_numeric"].ge(5)
        & patient["sex"].isin(["M", "F"])
        & patient["initial_ktas"].between(1, 5)
        & patient["avpu"].isin(["A", "V", "P", "U"])
        & patient["arrival_method_code"].isin(["1", "2", "3", "4", "5", "6", "7", "8"])
        & patient["arrival_route_code"].isin(["1", "2", "3", "8"])
        & patient["facility"].notna()
        & patient["facility"].ne("")
    )

    estimate_rows: list[dict] = []
    diagnostics: dict[str, dict] = {}
    for key, label in OUTCOMES.items():
        for disease in DISEASES:
            valid = common_valid & patient["disease"].eq(disease) & patient[f"{key}_valid"]
            frame = patient.loc[valid].copy().reset_index(drop=True)
            y = frame[f"{key}_event"].astype(int).to_numpy(float)
            include_ed_level = key != "high_acuity_area"
            x, names = build_design(frame, include_ed_level=include_ed_level)
            fit = fit_logistic_two_way(y, x, frame["facility"], frame["week_start"])
            model_key = f"{key}|{disease}"
            diagnostics[model_key] = {
                "outcome": label,
                "disease": disease,
                "n_encounters": int(len(frame)),
                "events": int(y.sum()),
                "event_pct": float(100 * y.mean()),
                "ed_level_handling": "absorbed by facility fixed effects" if include_ed_level else "constant regional-ED restriction",
                **{k: v for k, v in fit.items() if k not in ["beta", "covariance"]},
            }
            effects = [("Immediate level change", 0), ("Weekly slope change", None), ("Combined postinterruption effect", 13), ("Combined postinterruption effect", 26), ("Combined postinterruption effect", 43)]
            for parameter, post_week in effects:
                base = {"outcome_key": key, "outcome": label, "disease": disease, "parameter": parameter, "post_week": post_week if post_week else np.nan, "effect_scale": "adjusted OR", "status": "estimable" if fit["estimable"] else "NE", "reason": fit["reason"], "n_encounters": len(frame), "events": int(y.sum())}
                if not fit["estimable"]:
                    estimate_rows.append({**base, "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "z": np.nan, "p_raw": np.nan})
                    continue
                contrast = np.zeros(len(names))
                if parameter == "Immediate level change":
                    contrast[names.index("post")] = 1
                elif parameter == "Weekly slope change":
                    contrast[names.index("post_time_per_year")] = 1 / PERIOD_WEEKS
                else:
                    contrast[names.index("post")] = 1
                    contrast[names.index("post_time_per_year")] = float(post_week) / PERIOD_WEEKS
                estimate_rows.append({**base, **linear_combination(fit["beta"], fit["covariance"], contrast, "exp")})

    estimates = pd.DataFrame(estimate_rows)
    estimates.to_csv(ADJ_OUT / "case_mix_adjusted_estimates.csv", index=False)
    write_json(ADJ_OUT / "case_mix_adjusted_diagnostics.json", diagnostics)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": len(diagnostics),
        "estimable_models": int(sum(value["estimable"] for value in diagnostics.values())),
        "nonestimable_models": int(sum(not value["estimable"] for value in diagnostics.values())),
        "estimate_rows": int(len(estimates)),
    }
    write_json(ADJ_OUT / "case_mix_adjusted_summary.json", summary)
    print(estimates.groupby(["status", "outcome"]).size().to_string())
    print(summary)


if __name__ == "__main__":
    main()
