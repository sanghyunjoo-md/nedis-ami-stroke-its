#!/usr/bin/env python3
"""Run all locked confirmatory ITS models and diagnostics in one execution."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from model_utils import (
    effect_row,
    fit_grouped_binomial_hac,
    fit_nb2_hac,
    holm_adjust,
    influence_measures,
    linear_combination,
    residual_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis_outputs" / "02_cohort" / "weekly_primary_inputs.csv"
OUT = ROOT / "analysis_outputs" / "03_primary_its"
DISEASES = ["AMI", "Ischemic stroke", "Hemorrhagic stroke"]
TERMS = ["intercept", "time", "post", "post_time", "sin1", "cos1"]


def design(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(d)),
            d["calendar_time"].to_numpy(float),
            d["post"].to_numpy(float),
            d["post_time"].to_numpy(float),
            d["sin1"].to_numpy(float),
            d["cos1"].to_numpy(float),
        ]
    )


def response_band(x: np.ndarray, beta: np.ndarray, cov: np.ndarray, family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta = x @ beta
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", x, cov, x), 0.0))
    if family == "count":
        return np.exp(eta), np.exp(eta - 1.96 * se), np.exp(eta + 1.96 * se)
    return expit(eta), expit(eta - 1.96 * se), expit(eta + 1.96 * se)


def fit_one(d: pd.DataFrame, outcome: str):
    x = design(d)
    if outcome == "ED visit count":
        y = d["visit_count"].to_numpy(float)
        total = None
        fit = fit_nb2_hac(y, x, lag=4)
        family = "count"
        observed = y
    else:
        y = d["transfer_out_n"].to_numpy(float)
        total = d["known_disposition_n"].to_numpy(float)
        fit = fit_grouped_binomial_hac(y, total, x, lag=4)
        family = "rate"
        observed = y / total
    return x, y, total, fit, family, observed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_csv(INPUT, parse_dates=["week_start"])
    primary_rows = []
    coefficient_rows = []
    combined_rows = []
    prediction_frames = []
    diagnostics = {}
    loo_rows = []

    for disease in DISEASES:
        d = weekly.loc[weekly["disease"].eq(disease)].sort_values("week_start").reset_index(drop=True)
        for outcome in ("ED visit count", "Transfer-out rate"):
            x, y, total, fit, family, observed = fit_one(d, outcome)
            key = f"{disease} | {outcome}"
            if not fit.converged:
                raise RuntimeError(f"Model did not converge: {key}: {fit.message}")

            for j, term in enumerate(TERMS):
                row = effect_row(fit.params, fit.cov_hac, j, "exp")
                coefficient_rows.append(
                    {
                        "disease": disease,
                        "outcome": outcome,
                        "model_family": fit.family,
                        "term": term,
                        "coefficient": float(fit.params[j]),
                        "se_hac": float(np.sqrt(max(fit.cov_hac[j, j], 0.0))),
                        **row,
                    }
                )

            for parameter, idx in (("Immediate level change", 2), ("Weekly slope change", 3)):
                row = effect_row(fit.params, fit.cov_hac, idx, "exp")
                primary_rows.append(
                    {
                        "disease": disease,
                        "outcome": outcome,
                        "parameter": parameter,
                        "effect_scale": "IRR" if outcome == "ED visit count" else "OR",
                        "model_family": fit.family,
                        "alpha": fit.alpha,
                        **row,
                    }
                )

            for k in (13, 26, 43):
                contrast = np.zeros(len(TERMS))
                contrast[2] = 1.0
                contrast[3] = float(k)
                row = linear_combination(fit.params, fit.cov_hac, contrast, "exp")
                combined_rows.append(
                    {
                        "disease": disease,
                        "outcome": outcome,
                        "post_week": k,
                        "effect_scale": "IRR" if outcome == "ED visit count" else "OR",
                        **row,
                    }
                )

            x_counter = x.copy()
            x_counter[:, 2] = 0.0
            x_counter[:, 3] = 0.0
            fitted, fit_lo, fit_hi = response_band(x, fit.params, fit.cov_hac, family)
            counter, counter_lo, counter_hi = response_band(x_counter, fit.params, fit.cov_hac, family)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "disease": disease,
                        "outcome": outcome,
                        "week_start": d["week_start"],
                        "observed": observed,
                        "fitted": fitted,
                        "fitted_ci_low": fit_lo,
                        "fitted_ci_high": fit_hi,
                        "counterfactual": counter,
                        "counterfactual_ci_low": counter_lo,
                        "counterfactual_ci_high": counter_hi,
                        "post": d["post"],
                    }
                )
            )

            influence = influence_measures(x, fit, y, total)
            diag = residual_diagnostics(influence["pearson"], max_lag=12)
            p = x.shape[1]
            n = x.shape[0]
            flagged = (
                (influence["leverage"] > 2 * p / n)
                | (influence["cook"] > 4 / n)
                | (np.max(np.abs(influence["dfbeta"]), axis=1) > 2 / np.sqrt(n))
            )
            diag.update(
                {
                    "model_family": fit.family,
                    "converged": fit.converged,
                    "message": fit.message,
                    "alpha": fit.alpha,
                    "loglik": fit.loglik,
                    "pearson_summary": {
                        "mean": float(np.mean(influence["pearson"])),
                        "sd": float(np.std(influence["pearson"], ddof=1)),
                        "min": float(np.min(influence["pearson"])),
                        "max": float(np.max(influence["pearson"])),
                    },
                    "max_leverage": float(np.max(influence["leverage"])),
                    "max_cook": float(np.max(influence["cook"])),
                    "max_abs_dfbeta": float(np.max(np.abs(influence["dfbeta"]))),
                    "flagged_week_count": int(flagged.sum()),
                    "flagged_weeks": [str(x.date()) for x in d.loc[flagged, "week_start"]],
                }
            )
            diagnostics[key] = diag

            base_post = float(fit.params[2])
            base_slope = float(fit.params[3])
            for idx in np.flatnonzero(flagged):
                keep = np.ones(n, dtype=bool)
                keep[idx] = False
                dd = d.loc[keep].reset_index(drop=True)
                _, _, _, refit, _, _ = fit_one(dd, outcome)
                loo_rows.append(
                    {
                        "disease": disease,
                        "outcome": outcome,
                        "omitted_week": str(d.loc[idx, "week_start"].date()),
                        "refit_family": refit.family,
                        "refit_converged": refit.converged,
                        "base_beta_post": base_post,
                        "refit_beta_post": float(refit.params[2]),
                        "base_beta_post_time": base_slope,
                        "refit_beta_post_time": float(refit.params[3]),
                        "post_direction_preserved": bool(np.sign(base_post) == np.sign(refit.params[2])),
                        "slope_direction_preserved": bool(np.sign(base_slope) == np.sign(refit.params[3])),
                    }
                )

    primary = pd.DataFrame(primary_rows)
    primary["p_holm"] = holm_adjust(primary["p_raw"].to_numpy())
    primary["holm_significant_0_05"] = primary["p_holm"] < 0.05
    primary.to_csv(OUT / "confirmatory_estimates.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(OUT / "all_primary_model_coefficients.csv", index=False)
    pd.DataFrame(combined_rows).to_csv(OUT / "combined_effects_post_weeks_13_26_43.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(OUT / "primary_predictions_with_95CI.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(OUT / "leave_one_flagged_week_out.csv", index=False)
    (OUT / "primary_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": 6,
        "confirmatory_tests": 12,
        "all_models_converged": True,
        "holm_significant_tests": int(primary["holm_significant_0_05"].sum()),
        "flagged_week_refits": int(len(loo_rows)),
    }
    (OUT / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(primary.to_string(index=False))
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
