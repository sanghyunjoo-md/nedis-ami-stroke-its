#!/usr/bin/env python3
"""Fit all locked disease-specific and disease-heterogeneity extension models."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.special import expit

from common import BASE_TERMS, DISEASES, OUT, design, wald_test, write_json
from model_utils import ModelFit, bh_adjust, effect_row, fit_grouped_binomial_hac, fit_ols_hac, linear_combination, residual_diagnostics, sandwich


INPUT = OUT / "01_derived" / "weekly_extension_inputs.csv"
MODEL_OUT = OUT / "02_models"
OUTCOME_SPECS = {
    "onset_to_arrival": {"label": "Onset-to-arrival time", "domain": "Access", "type": "continuous", "median": "onset_median_minutes", "scale": "Ratio of medians"},
    "impaired_consciousness": {"label": "Impaired consciousness", "domain": "Presenting severity", "type": "binary", "scale": "OR"},
    "shock_range_sbp": {"label": "Shock-range SBP", "domain": "Presenting severity", "type": "binary", "scale": "OR"},
    "hypoxemia": {"label": "Hypoxemia", "domain": "Presenting severity", "type": "binary", "scale": "OR"},
    "ktas_escalation": {"label": "KTAS escalation", "domain": "Care pathways", "type": "binary", "scale": "OR"},
    "specialist_involvement": {"label": "Consulting-specialist involvement", "domain": "Care pathways", "type": "binary", "scale": "OR"},
    "concordant_department": {"label": "Condition-concordant department", "domain": "Care pathways", "type": "binary", "scale": "OR"},
    "high_acuity_area": {"label": "High-acuity final treatment area", "domain": "Care pathways", "type": "binary", "scale": "OR"},
}


def fit_binomial_stable(success: np.ndarray, total: np.ndarray, x: np.ndarray, lag: int) -> ModelFit:
    fit = fit_grouped_binomial_hac(success, total, x, lag=lag)
    if fit.converged:
        return fit

    def scores(beta: np.ndarray) -> np.ndarray:
        probability = expit(np.clip(x @ beta, -30, 30))
        return x * (success - total * probability)[:, None]

    refined = root(lambda beta: scores(beta).sum(axis=0), fit.params, method="hybr", options={"xtol": 1e-11, "maxfev": 20000})
    beta = refined.x if refined.success and np.all(np.isfinite(refined.x)) else fit.params
    probability = expit(np.clip(x @ beta, -30, 30))
    weights = total * probability * (1 - probability)
    bread = np.linalg.pinv(x.T @ (weights[:, None] * x))
    score_obs = scores(beta)
    covariance = sandwich(bread, score_obs, lag, len(beta))
    gradient_inf = float(np.linalg.norm(score_obs.sum(axis=0), ord=np.inf))
    return ModelFit(
        family="Grouped binomial logit",
        params=beta,
        cov_hac=covariance,
        cov_model=bread,
        fitted=probability,
        score_obs=score_obs,
        converged=bool(refined.success or gradient_inf < 1e-4),
        message=f"score-equation refinement: {refined.message}; gradient_inf={gradient_inf:.3g}",
        alpha=None,
        loglik=float(np.sum(success * np.log(np.maximum(probability, 1e-12)) + (total - success) * np.log(np.maximum(1 - probability, 1e-12)))),
    )


def fit_outcome(frame: pd.DataFrame, key: str, lag: int = 4, second_harmonic: bool = False):
    spec = OUTCOME_SPECS[key]
    x, terms = design(frame, second_harmonic=second_harmonic)
    if spec["type"] == "continuous":
        keep = frame[spec["median"]].gt(0) & frame[spec["median"]].notna()
        d = frame.loc[keep].reset_index(drop=True)
        x, terms = design(d, second_harmonic=second_harmonic)
        y = np.log(d[spec["median"]].to_numpy(float))
        total = None
        fit = fit_ols_hac(y, x, lag=lag)
    else:
        keep = frame[f"{key}_valid_n"].gt(0)
        d = frame.loc[keep].reset_index(drop=True)
        x, terms = design(d, second_harmonic=second_harmonic)
        y = d[f"{key}_event_n"].to_numpy(float)
        total = d[f"{key}_valid_n"].to_numpy(float)
        fit = fit_binomial_stable(y, total, x, lag=lag)
    return d, x, terms, y, total, fit


def diagnostic(frame: pd.DataFrame, x: np.ndarray, y: np.ndarray, total: np.ndarray | None, fit) -> dict:
    if total is None:
        residual = y - fit.fitted
        hat = np.einsum("ij,jk,ik->i", x, np.linalg.pinv(x.T @ x), x)
        mse = float(np.sum(residual**2) / max(len(y) - x.shape[1], 1))
        cook = residual**2 * hat / np.maximum(x.shape[1] * mse * (1 - hat) ** 2, 1e-12)
        dispersion = mse
    else:
        variance = np.maximum(total * fit.fitted * (1 - fit.fitted), 1e-12)
        residual = (y - total * fit.fitted) / np.sqrt(variance)
        weight = variance
        hat = weight * np.einsum("ij,jk,ik->i", x, fit.cov_model, x)
        cook = residual**2 * hat / np.maximum(x.shape[1] * (1 - hat) ** 2, 1e-12)
        dispersion = float(np.sum(residual**2) / max(len(y) - x.shape[1], 1))
    info = np.linalg.pinv(fit.cov_model)
    return {
        "model_family": fit.family,
        "converged": bool(fit.converged),
        "message": fit.message,
        "n_weeks": int(len(frame)),
        "design_columns": int(x.shape[1]),
        "information_rank": int(np.linalg.matrix_rank(info)),
        "information_condition_number": float(np.linalg.cond(info)),
        "dispersion": dispersion,
        "max_leverage": float(np.nanmax(hat)),
        "max_cook": float(np.nanmax(cook)),
        "denominator_min": float(np.min(total)) if total is not None else None,
        "denominator_median": float(np.median(total)) if total is not None else None,
        "zero_event_weeks": int(np.sum(y == 0)) if total is not None else None,
        **residual_diagnostics(residual, max_lag=12),
    }


def stacked_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    ischemic = frame["disease"].eq("Ischemic stroke").to_numpy(float)
    hemorrhagic = frame["disease"].eq("Hemorrhagic stroke").to_numpy(float)
    variables = ["calendar_time", "post", "post_time", "sin1", "cos1"]
    columns = [np.ones(len(frame)), ischemic, hemorrhagic]
    names = ["intercept", "ischemic", "hemorrhagic"]
    for variable in variables:
        value = frame[variable].to_numpy(float)
        columns.extend([value, ischemic * value, hemorrhagic * value])
        names.extend([variable, f"ischemic:{variable}", f"hemorrhagic:{variable}"])
    return np.column_stack(columns), names


def aggregate_week_scores(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    weeks = pd.Index(sorted(frame["week_start"].unique()))
    return np.vstack([scores[frame["week_start"].eq(week).to_numpy()].sum(axis=0) for week in weeks])


def main() -> None:
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_csv(INPUT, parse_dates=["week_start"])
    estimate_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    diagnostics: dict[str, dict] = {}

    for key, spec in OUTCOME_SPECS.items():
        for disease in DISEASES:
            base = weekly.loc[weekly["disease"].eq(disease)].sort_values("week_start").reset_index(drop=True)
            d, x, terms, y, total, fit = fit_outcome(base, key)
            model_key = f"{key}|{disease}"
            diagnostics[model_key] = diagnostic(d, x, y, total, fit)
            for idx, term in enumerate(terms):
                coefficient_rows.append({"outcome_key": key, "outcome": spec["label"], "domain": spec["domain"], "disease": disease, "term": term, "coefficient": float(fit.params[idx]), "se_hac": float(np.sqrt(max(fit.cov_hac[idx, idx], 0))), "model_family": fit.family})
            for parameter, idx in [("Immediate level change", terms.index("post")), ("Weekly slope change", terms.index("post_time"))]:
                estimate_rows.append({"outcome_key": key, "outcome": spec["label"], "domain": spec["domain"], "disease": disease, "parameter": parameter, "effect_scale": spec["scale"], "post_week": np.nan, **effect_row(fit.params, fit.cov_hac, idx, "exp")})
            for post_week in [13, 26, 43]:
                contrast = np.zeros(len(terms))
                contrast[terms.index("post")] = 1
                contrast[terms.index("post_time")] = post_week
                estimate_rows.append({"outcome_key": key, "outcome": spec["label"], "domain": spec["domain"], "disease": disease, "parameter": "Combined postinterruption effect", "effect_scale": spec["scale"], "post_week": post_week, **linear_combination(fit.params, fit.cov_hac, contrast, "exp")})

    estimates = pd.DataFrame(estimate_rows)
    estimates["q_domain"] = np.nan
    central_tests = estimates["parameter"].isin(["Immediate level change", "Weekly slope change"])
    for domain, idx in estimates.loc[central_tests].groupby("domain").groups.items():
        estimates.loc[idx, "q_domain"] = bh_adjust(estimates.loc[idx, "p_raw"].to_numpy())
    estimates["q_signal_0_05"] = estimates["q_domain"].lt(0.05).fillna(False)
    estimates.to_csv(MODEL_OUT / "central_extension_estimates.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(MODEL_OUT / "central_extension_coefficients.csv", index=False)
    write_json(MODEL_OUT / "central_extension_diagnostics.json", diagnostics)

    heterogeneity_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    heterogeneity_diagnostics: dict[str, dict] = {}
    for key, spec in OUTCOME_SPECS.items():
        stacked = weekly.sort_values(["week_start", "disease"]).reset_index(drop=True)
        if spec["type"] == "continuous":
            keep = stacked[spec["median"]].gt(0) & stacked[spec["median"]].notna()
            d = stacked.loc[keep].reset_index(drop=True)
            x, names = stacked_design(d)
            y = np.log(d[spec["median"]].to_numpy(float))
            fit = fit_ols_hac(y, x, lag=0)
            residual = y - fit.fitted
        else:
            keep = stacked[f"{key}_valid_n"].gt(0)
            d = stacked.loc[keep].reset_index(drop=True)
            x, names = stacked_design(d)
            y = d[f"{key}_event_n"].to_numpy(float)
            total = d[f"{key}_valid_n"].to_numpy(float)
            fit = fit_binomial_stable(y, total, x, lag=0)
            residual = (y - total * fit.fitted) / np.sqrt(np.maximum(total * fit.fitted * (1 - fit.fitted), 1e-12))
        week_scores = aggregate_week_scores(d, fit.score_obs)
        bread = np.linalg.pinv(x.T @ x) if spec["type"] == "continuous" else fit.cov_model
        cov_week_hac = sandwich(bread, week_scores, 4, x.shape[1])
        for parameter, term in [("Immediate level change", "post"), ("Weekly slope change", "post_time")]:
            indices = [names.index(f"ischemic:{term}"), names.index(f"hemorrhagic:{term}")]
            heterogeneity_rows.append({"outcome_key": key, "outcome": spec["label"], "domain": "Disease heterogeneity", "parameter": parameter, **wald_test(fit.params, cov_week_hac, indices)})
            contrast_specs = [
                ("Ischemic stroke vs AMI", {f"ischemic:{term}": 1.0}),
                ("Hemorrhagic stroke vs AMI", {f"hemorrhagic:{term}": 1.0}),
                ("Hemorrhagic vs ischemic stroke", {f"hemorrhagic:{term}": 1.0, f"ischemic:{term}": -1.0}),
            ]
            for contrast_name, weights in contrast_specs:
                contrast = np.zeros(len(names))
                for name, weight in weights.items():
                    contrast[names.index(name)] = weight
                pairwise_rows.append({"outcome_key": key, "outcome": spec["label"], "parameter": parameter, "contrast": contrast_name, "effect_scale": spec["scale"], **linear_combination(fit.params, cov_week_hac, contrast, "exp")})
        info = np.linalg.pinv(fit.cov_model)
        heterogeneity_diagnostics[key] = {
            "model_family": fit.family,
            "converged": bool(fit.converged),
            "n_rows": int(len(d)),
            "n_calendar_weeks": int(d["week_start"].nunique()),
            "design_columns": int(x.shape[1]),
            "information_rank": int(np.linalg.matrix_rank(info)),
            "information_condition_number": float(np.linalg.cond(info)),
            **residual_diagnostics(residual, max_lag=12),
        }

    heterogeneity = pd.DataFrame(heterogeneity_rows)
    heterogeneity["q_domain"] = bh_adjust(heterogeneity["p_raw"].to_numpy())
    heterogeneity["q_signal_0_05"] = heterogeneity["q_domain"].lt(0.05)
    heterogeneity.to_csv(MODEL_OUT / "disease_heterogeneity_tests.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(MODEL_OUT / "disease_pairwise_contrasts.csv", index=False)
    write_json(MODEL_OUT / "disease_heterogeneity_diagnostics.json", heterogeneity_diagnostics)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "disease_specific_models": len(OUTCOME_SPECS) * len(DISEASES),
        "central_immediate_slope_tests": int(central_tests.sum()),
        "combined_effect_estimates": int((estimates["parameter"] == "Combined postinterruption effect").sum()),
        "heterogeneity_models": len(OUTCOME_SPECS),
        "heterogeneity_tests": int(len(heterogeneity)),
        "all_disease_specific_models_converged": bool(all(v["converged"] for v in diagnostics.values())),
        "all_heterogeneity_models_converged": bool(all(v["converged"] for v in heterogeneity_diagnostics.values())),
        "extension_q_signals": int(estimates["q_signal_0_05"].sum()),
        "heterogeneity_q_signals": int(heterogeneity["q_signal_0_05"].sum()),
    }
    write_json(MODEL_OUT / "model_run_summary.json", summary)
    print(estimates.loc[central_tests, ["disease", "outcome", "parameter", "estimate", "ci_low", "ci_high", "p_raw", "q_domain"]].to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
